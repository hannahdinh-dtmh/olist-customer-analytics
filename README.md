# Olist Customer Analytics — Delivery Performance & RFM Analysis

A portfolio-grade analytics project built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It combines RFM customer segmentation with delivery performance analysis and a logistic regression churn model to answer a question most analysts miss: **does late delivery drive customer churn, and which segments are most at risk?**

---

## Project Structure

```
Olist delivery & RFM/
├── data_prep.py          # Data pipeline: merge → RFM → delivery → churn model
├── app.py                # Streamlit dashboard (5 tabs)
├── requirements.txt      # Python dependencies
└── data/
    │   ── Raw inputs (download from Kaggle, not tracked in git) ──
    ├── olist_orders_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_customers_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    ├── product_category_name_translation.csv
    │   ── Generated outputs (committed to git) ──
    ├── master_df.csv         # Order-level enriched dataset
    ├── rfm_df.csv            # Customer-level RFM + churn predictions
    └── churn_coefficients.csv  # Model feature importances
```

---

## Dataset

**Source:** [Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**Coverage:** 99,441 orders from 2016–2018 across multiple Brazilian marketplaces, with 8 linked tables covering customers, orders, payments, reviews, products, sellers, and product categories. (The geolocation file from the Kaggle download is not used in this analysis.)

---

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/hannahdinh-dtmh/olist-customer-analytics.git
cd olist-customer-analytics
pip install -r requirements.txt
```

**2. Download the Olist dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), unzip, and place the 8 CSV files listed above into the `data/` folder. (You can skip `olist_geolocation_dataset.csv` — it is not used.)

**3. Run the data pipeline**
```bash
python data_prep.py
```

This merges all 9 tables, computes RFM scores, delivery metrics, and trains the churn model. Outputs are saved to `data/`.

**4. Launch the dashboard**
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Dashboard — 5 Tabs

### 🏠 Customer Overview
High-level market snapshot: total customers, orders, revenue, average order value, review score, and on-time delivery rate. Includes monthly order volume & revenue trend, top states by revenue, and payment type breakdown.

### 🎯 RFM Segmentation
Customers scored on Recency, Frequency, and Monetary value (quintile 1–5), then assigned to 8 behavioural segments: Champions, Loyal Customers, New Customers, Potential Loyalists, At Risk, Can't Lose Them, Hibernating, and Lost. Visualised as a treemap (hero chart), bubble scatter, and segment profile table.

### 🚚 Delivery Performance
On-time rate (93.2%), average delay distribution, monthly on-time trend, worst-performing states, and average delay by product category. Key finding: northern states (AM, RR, AP) consistently underperform due to logistics infrastructure gaps.

### ⚠️ Churn Risk
The core novel analysis. A logistic regression model predicts each customer's probability of churning (defined as 180+ days of inactivity). Includes a **segment × churn rate scatter** showing that segments with higher late delivery rates have significantly higher churn probabilities — direct evidence that delivery experience drives retention.

### 🗺️ Geographic Intelligence
State-level bar charts for revenue and on-time delivery rate, plus a full state summary table combining orders, revenue, review scores, delivery performance, RFM scores, and churn risk.

---

## Methodology

### RFM Scoring
- **Recency:** days since last order (lower = better → score 5 to 1)
- **Frequency:** number of distinct orders (custom bins: 1 / 2 / 3 / 4–5 / 6+)
- **Monetary:** total payment value (quintiles, higher = better → score 1 to 5)

Segment labels are assigned by rule-based logic on the R, F, M scores (e.g. Champions = R≥4, F≥4, M≥4).

### Churn Definition
A customer is labelled **churned** if their most recent order was more than **180 days** before the dataset snapshot date (2018-08-30). This reflects the typical re-purchase window for Brazilian marketplace shoppers.

### Churn Model
**Algorithm:** Logistic Regression with `class_weight="balanced"` (to handle the ~77% churn rate imbalance).

**Features:**

| Feature | Description |
|---|---|
| `recency_days` | Days since last order |
| `frequency` | Number of orders |
| `monetary` | Total spend (R$) |
| `avg_review_score` | Mean review rating (1–5) |
| `late_pct` | % of orders delivered late |
| `avg_delay_days` | Average delivery delay (days) |

The model outputs a churn probability (0–1) for each customer, used to populate the high-risk customer table and segment-level churn analysis in Tab 4.

---

## Key Findings

- **93.2% on-time delivery rate** overall, but northern states fall as low as 75–80%
- **Late delivery is a statistically significant churn predictor** — the two strongest non-RFM features in the logistic regression are `late_pct` and `avg_delay_days`
- **At Risk and Hibernating segments** have both higher late delivery rates and higher churn probabilities than Champions and Loyal Customers in the same recency bands
- **São Paulo accounts for ~40% of total revenue**, but its high on-time rate suggests infrastructure quality follows economic concentration

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data merging and feature engineering |
| `scikit-learn` | Logistic Regression, StandardScaler, train/test split |
| `plotly` | Interactive charts throughout the dashboard |
| `streamlit` | Dashboard framework |

---

## Related Projects

- [eBay Product Analytics Dashboard](https://github.com/hannahdinh-dtmh/ebay-product-analytics-dashboard) — eBay electronics scraper + dark-mode Streamlit dashboard with product family segmentation and outlier detection
