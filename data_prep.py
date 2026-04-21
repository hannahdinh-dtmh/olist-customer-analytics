#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Olist Customer Analytics — Data Preparation Pipeline
=====================================================
Merges all 9 Olist CSV tables, computes:
  - Delivery performance metrics (delay days, on-time flag)
  - RFM scores + behavioural segment labels
  - Logistic regression churn model (P(churn) per customer)

Outputs:
  data/master_df.csv          — order-level enriched dataset
  data/rfm_df.csv             — customer-level RFM + churn predictions
  data/churn_coefficients.csv — model feature importances

Usage:
  python data_prep.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_olist():
    """Load all 9 Olist CSV files from the data/ directory."""
    print("Loading Olist datasets...")

    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    items    = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
    custs    = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
    payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv")
    reviews  = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
    products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
    sellers  = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
    cat_map  = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

    print(f"  Orders loaded:    {len(orders):,}")
    print(f"  Customers loaded: {len(custs):,}")
    return orders, items, custs, payments, reviews, products, sellers, cat_map


# ── 2. Merge ───────────────────────────────────────────────────────────────────

def merge_tables(orders, items, custs, payments, reviews, products, sellers, cat_map):
    print("\nMerging tables...")

    # Items: aggregate per order
    items_agg = (items.groupby("order_id")
                 .agg(
                     item_count     = ("order_item_id",   "count"),
                     total_price    = ("price",           "sum"),
                     total_freight  = ("freight_value",   "sum"),
                     product_id     = ("product_id",      "first"),
                     seller_id      = ("seller_id",       "first"),
                 ).reset_index())

    # Payments: aggregate per order
    payments_agg = (payments.groupby("order_id")
                    .agg(
                        payment_value        = ("payment_value",        "sum"),
                        payment_type         = ("payment_type",         lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
                        payment_installments = ("payment_installments", "mean"),
                    ).reset_index())

    # Reviews: keep most recent per order
    reviews_agg = (reviews
                   .sort_values("review_answer_timestamp")
                   .groupby("order_id")
                   .agg(review_score=("review_score", "last"))
                   .reset_index())

    # Products: translate category names
    products = products.merge(cat_map, on="product_category_name", how="left")
    products["category"] = (products["product_category_name_english"]
                            .fillna(products["product_category_name"])
                            .fillna("unknown")
                            .str.replace("_", " ").str.title())

    # Build master join
    df = (orders
          .merge(custs,                                  on="customer_id",  how="left")
          .merge(items_agg,                              on="order_id",     how="left")
          .merge(payments_agg,                           on="order_id",     how="left")
          .merge(reviews_agg,                            on="order_id",     how="left")
          .merge(products[["product_id", "category"]],  on="product_id",   how="left")
          .merge(sellers[["seller_id", "seller_state"]], on="seller_id",   how="left"))

    print(f"  Master shape: {df.shape}")
    return df


# ── 3. Delivery Metrics ────────────────────────────────────────────────────────

def compute_delivery_metrics(df):
    print("\nComputing delivery metrics...")

    d = df[df["order_status"] == "delivered"].copy()

    d["delay_days"] = (
        d["order_delivered_customer_date"] - d["order_estimated_delivery_date"]
    ).dt.days

    d["is_late"]  = d["delay_days"] > 0
    d["is_early"] = d["delay_days"] < 0
    d["delivery_days_actual"] = (
        d["order_delivered_customer_date"] - d["order_purchase_timestamp"]
    ).dt.days

    df = df.merge(
        d[["order_id", "delay_days", "is_late", "is_early", "delivery_days_actual"]],
        on="order_id", how="left"
    )

    on_time_pct = (~d["is_late"]).mean() * 100
    avg_delay   = d.loc[d["is_late"], "delay_days"].mean()
    print(f"  On-time rate:     {on_time_pct:.1f}%")
    print(f"  Avg delay (late): {avg_delay:.1f} days")
    return df


# ── 4. RFM ─────────────────────────────────────────────────────────────────────

SEGMENT_RULES = [
    ("Champions",          lambda r, f, m: r >= 4 and f >= 4 and m >= 4),
    ("Loyal Customers",    lambda r, f, m: r >= 3 and f >= 3 and m >= 3),
    ("New Customers",      lambda r, f, m: r >= 4 and f <= 1),
    ("Potential Loyalists",lambda r, f, m: r >= 3 and f <= 2 and m <= 3),
    ("Can't Lose Them",    lambda r, f, m: r <= 2 and f >= 4 and m >= 4),
    ("At Risk",            lambda r, f, m: r <= 2 and f >= 3 and m >= 3),
    ("Hibernating",        lambda r, f, m: r <= 3 and f <= 2 and m <= 3),
    ("Lost",               lambda r, f, m: r == 1 and f <= 2 and m <= 2),
]

def assign_segment(row):
    r, f, m = row["R"], row["F"], row["M"]
    for label, rule in SEGMENT_RULES:
        if rule(r, f, m):
            return label
    return "Hibernating"

def compute_rfm(df):
    print("\nComputing RFM scores...")

    base = df[df["order_status"] == "delivered"].copy()
    snapshot = base["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    print(f"  Snapshot date: {snapshot.date()}")

    rfm = (base.groupby("customer_unique_id")
           .agg(
               last_order_date    = ("order_purchase_timestamp", "max"),
               frequency          = ("order_id",                 "nunique"),
               monetary           = ("payment_value",            "sum"),
               avg_review_score   = ("review_score",             "mean"),
               late_pct           = ("is_late",  lambda x: x.mean() * 100),
               avg_delay_days     = ("delay_days",               "mean"),
               customer_state     = ("customer_state",           "first"),
               payment_type       = ("payment_type",
                                     lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
           ).reset_index())

    rfm["recency_days"] = (snapshot - rfm["last_order_date"]).dt.days

    # Quintile scoring (5 = best for all three)
    rfm["R"] = pd.qcut(rfm["recency_days"],  q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["F"] = pd.cut(rfm["frequency"],
                      bins=[0,1,2,3,5, rfm["frequency"].max()+1],
                      labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"], q=5, labels=[1,2,3,4,5]).astype(int)

    rfm["RFM_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["RFM_total"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["Segment"]   = rfm.apply(assign_segment, axis=1)

    print("  Segment distribution:")
    print(rfm["Segment"].value_counts().to_string())
    return rfm, snapshot


# ── 5. Churn Model ─────────────────────────────────────────────────────────────

def compute_churn_model(rfm):
    """
    Define churn as: no purchase in the last 180 days before snapshot.
    Features: recency, frequency, monetary, review score, late delivery %, avg delay.
    Model: Logistic Regression with StandardScaler.
    """
    print("\nTraining churn model (Logistic Regression)...")

    # Churn label
    rfm["is_churned"] = (rfm["recency_days"] > 180).astype(int)
    churn_rate = rfm["is_churned"].mean() * 100
    print(f"  Churn rate (180-day threshold): {churn_rate:.1f}%")

    FEATURES = ["recency_days", "frequency", "monetary",
                "avg_review_score", "late_pct", "avg_delay_days"]

    model_df = rfm[FEATURES + ["is_churned"]].dropna()
    X, y = model_df[FEATURES], model_df["is_churned"]

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print(f"  ROC-AUC: {auc:.3f}")
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))

    # Predict for all customers
    X_all = scaler.transform(rfm.loc[model_df.index, FEATURES])
    rfm.loc[model_df.index, "churn_probability"] = clf.predict_proba(X_all)[:, 1]
    rfm["churn_probability"] = rfm["churn_probability"].fillna(0.5)

    # Feature importances (absolute coefficients)
    coef_df = pd.DataFrame({
        "Feature":     [f.replace("_", " ").title() for f in FEATURES],
        "Coefficient": clf.coef_[0],
        "Impact":      ["Increases churn risk" if c > 0 else "Reduces churn risk"
                        for c in clf.coef_[0]],
    }).sort_values("Coefficient", key=abs, ascending=True)

    return rfm, coef_df, auc


# ── 6. Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    orders, items, custs, payments, reviews, products, sellers, cat_map = load_olist()
    master_df = merge_tables(orders, items, custs, payments, reviews, products, sellers, cat_map)
    master_df = compute_delivery_metrics(master_df)
    rfm_df, snapshot = compute_rfm(master_df)
    rfm_df, coef_df, auc = compute_churn_model(rfm_df)

    master_path = os.path.join(DATA_DIR, "master_df.csv")
    rfm_path    = os.path.join(DATA_DIR, "rfm_df.csv")
    coef_path   = os.path.join(DATA_DIR, "churn_coefficients.csv")

    master_df.to_csv(master_path, index=False)
    rfm_df.to_csv(rfm_path,       index=False)
    coef_df.to_csv(coef_path,     index=False)

    print(f"\n✅ Done — files saved to {DATA_DIR}/")
    print(f"   master_df.csv         {len(master_df):,} rows")
    print(f"   rfm_df.csv            {len(rfm_df):,} customers")
    print(f"   churn_coefficients.csv {len(coef_df)} features")
    print(f"\n   Model AUC: {auc:.3f}  |  Snapshot: {snapshot.date()}")


if __name__ == "__main__":
    main()
