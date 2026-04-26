#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Olist Customer Analytics Dashboard
====================================
Dark-mode Streamlit dashboard with 5 tabs:
  1. 🏠 Customer Overview       — KPIs, order trends, revenue geography
  2. 🎯 RFM Segmentation        — Scoring, segment profiles, bubble map
  3. 🚚 Delivery Performance    — On-time rates, delays by state & category
  4. ⚠️  Churn Risk              — ML churn predictions, segment × churn heatmap
  5. 🗺️  Geographic Intelligence — State-level revenue, delivery & RFM maps

Run:
  streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Olist Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1C1E2E, #252837);
        border: 1px solid #2E3148; border-radius: 12px;
        padding: 20px 24px; text-align: center; margin-bottom: 10px;
    }
    .kpi-label { font-size:12px; color:#9DA3AE; text-transform:uppercase;
                 letter-spacing:1px; margin-bottom:6px; }
    .kpi-value { font-size:28px; font-weight:700; color:#FFFFFF; line-height:1.1; }
    .kpi-delta-pos { font-size:12px; color:#00C49F; margin-top:4px; }
    .kpi-delta-neg { font-size:12px; color:#FF6B6B; margin-top:4px; }
    .kpi-delta-neu { font-size:12px; color:#9DA3AE;  margin-top:4px; }
    .section-title {
        font-size:18px; font-weight:600; color:#E2E8F0;
        margin:20px 0 12px 0; padding-bottom:6px;
        border-bottom:1px solid #2E3148;
    }
    .insight-box {
        background:rgba(76,139,245,0.08); border-left:3px solid #4C8BF5;
        border-radius:0 8px 8px 0; padding:12px 16px;
        color:#A0B4D6; font-size:13px; margin:10px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
    .stTabs [data-baseweb="tab"] {
        background:#1C1E2E; border-radius:8px 8px 0 0;
        padding:8px 16px; color:#9DA3AE;
    }
    .stTabs [aria-selected="true"] { background:#252837; color:#FFFFFF; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

PT = "plotly_dark"
CS = ["#4C8BF5","#00C49F","#FFB347","#FF6B6B","#A855F7","#F472B6","#34D399","#FB923C"]

SEGMENT_COLORS = {
    "Champions":          "#00C49F",
    "Loyal Customers":    "#4C8BF5",
    "New Customers":      "#34D399",
    "Potential Loyalists":"#FFB347",
    "At Risk":            "#FB923C",
    "Can't Lose Them":    "#A855F7",
    "Hibernating":        "#9DA3AE",
    "Lost":               "#FF6B6B",
}

SEGMENT_ORDER = list(SEGMENT_COLORS.keys())

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_master():
    p = os.path.join(DATA_DIR, "master_df.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["order_year"]  = df["order_purchase_timestamp"].dt.year
    return df

@st.cache_data
def load_rfm():
    p = os.path.join(DATA_DIR, "rfm_df.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_coef():
    p = os.path.join(DATA_DIR, "churn_coefficients.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    # Drop unnamed index column if present (saved with index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Normalise column names in case they differ
    df.columns = [c.strip() for c in df.columns]
    if "Coefficient" not in df.columns and len(df.columns) == 2:
        df.columns = ["Feature", "Coefficient"]
    return df if "Coefficient" in df.columns else None

master_df = load_master()
rfm_df    = load_rfm()
coef_df   = load_coef()

# ── Helpers ────────────────────────────────────────────────────────────────────

def kpi(label, value, delta=None, pos=True):
    cls = "kpi-delta-pos" if pos else "kpi-delta-neg"
    dh  = f'<div class="{cls}">{delta}</div>' if delta else ""
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>{dh}
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────────────────────

if master_df is None or rfm_df is None:
    st.error("⚠️ Data files not found.")
    st.markdown("""
    **Setup steps:**
    1. Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
    2. Unzip all 9 CSV files into the `data/` folder
    3. Run: `python data_prep.py`
    4. Relaunch: `streamlit run app.py`
    """)
    st.stop()

# ── Sidebar / Filters ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 Olist Analytics")
    st.markdown("---")
    st.markdown("### Filters")

    states = ["All"] + sorted(master_df["customer_state"].dropna().unique().tolist())
    sel_state = st.selectbox("Customer State", states)

    months = sorted(master_df["order_month"].dropna().unique().tolist())
    sel_months = st.select_slider("Date Range", options=months,
                                  value=(months[0], months[-1]))

    sel_status = st.multiselect(
        "Order Status",
        options=master_df["order_status"].dropna().unique().tolist(),
        default=["delivered"],
    )

    st.markdown("---")
    st.markdown(f"**Customers:** {master_df['customer_unique_id'].nunique():,}")
    st.markdown(f"**Orders:** {master_df['order_id'].nunique():,}")
    st.markdown("**Period:** 2016 – 2018 · Brazil")

# Apply filters
mdf = master_df.copy()
if sel_state != "All":
    mdf = mdf[mdf["customer_state"] == sel_state]
if sel_status:
    mdf = mdf[mdf["order_status"].isin(sel_status)]
mdf = mdf[(mdf["order_month"] >= sel_months[0]) & (mdf["order_month"] <= sel_months[1])]

if len(mdf) == 0:
    st.warning("No data matches current filters.")
    st.stop()

delivered = mdf[mdf["order_status"] == "delivered"].copy()

# ── Tabs ───────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    "🏠 Customer Overview",
    "🎯 RFM Segmentation",
    "🚚 Delivery Performance",
    "⚠️ Churn Risk",
    "🗺️ Geographic Intelligence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CUSTOMER OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with t1:
    sec("Market Snapshot")

    total_customers = mdf["customer_unique_id"].nunique()
    total_orders    = mdf["order_id"].nunique()
    total_revenue   = delivered["payment_value"].sum()
    avg_order_val   = delivered["payment_value"].mean()
    avg_review      = delivered["review_score"].mean()
    on_time_rate    = (~delivered["is_late"].fillna(False)).mean() * 100 if "is_late" in delivered else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Unique Customers", f"{total_customers:,}")
    with c2: kpi("Total Orders",     f"{total_orders:,}")
    with c3: kpi("Total Revenue",    f"R${total_revenue:,.0f}")
    with c4: kpi("Avg Order Value",  f"R${avg_order_val:.2f}")
    with c5: kpi("Avg Review Score", f"{avg_review:.2f} ⭐",
                 delta="above 4.0 = healthy" if avg_review >= 4 else "below 4.0 — investigate",
                 pos=avg_review >= 4)
    with c6: kpi("On-Time Rate", f"{on_time_rate:.1f}%",
                 delta="↑ good" if on_time_rate >= 90 else "↓ needs improvement",
                 pos=on_time_rate >= 90)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly order volume + revenue
    sec("Order Volume & Revenue Trend")
    monthly = (mdf.groupby("order_month")
               .agg(Orders=("order_id","nunique"), Revenue=("payment_value","sum"))
               .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["order_month"], y=monthly["Orders"],
                         name="Orders", marker_color="#4C8BF5", opacity=0.8,
                         yaxis="y"))
    fig.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["Revenue"],
                             name="Revenue (R$)", mode="lines+markers",
                             line=dict(color="#FFB347", width=2),
                             marker=dict(size=5), yaxis="y2"))
    fig.update_layout(
        template=PT, height=340,
        yaxis=dict(title="Orders", showgrid=False),
        yaxis2=dict(title="Revenue (R$)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=10, b=10),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)
    insight("Nov 2017 shows the largest spike — Black Friday effect. "
            "The trend rises steeply from mid-2017 as Olist expanded into new states.")

    col_a, col_b = st.columns(2)

    with col_a:
        sec("Payment Type Breakdown")
        pay = mdf["payment_type"].value_counts().reset_index()
        pay.columns = ["Type","Count"]
        pay["Type"] = pay["Type"].str.replace("_"," ").str.title()
        fig = px.pie(pay, names="Type", values="Count", hole=0.55,
                     color_discrete_sequence=CS, template=PT)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sec("Top 10 States by Revenue")
        state_rev = (delivered.groupby("customer_state")["payment_value"]
                     .sum().nlargest(10).reset_index())
        state_rev.columns = ["State","Revenue"]
        fig = px.bar(state_rev, x="Revenue", y="State", orientation="h",
                     color="Revenue", color_continuous_scale=["#1C2A4A","#4C8BF5","#00C49F"],
                     template=PT, text="Revenue",
                     labels={"Revenue":"Revenue (R$)","State":""})
        fig.update_traces(texttemplate="R$%{x:,.0f}", textposition="outside")
        fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=10,b=10), xaxis=dict(range=[0, state_rev["Revenue"].max()*1.15]))
        st.plotly_chart(fig, use_container_width=True)

    insight("São Paulo (SP) dominates at ~40% of total revenue — typical for Brazilian e-commerce "
            "where SP concentrates both the largest population and highest purchasing power.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

with t2:
    sec("Customer Segmentation — RFM Framework")

    if rfm_df is None:
        st.warning("RFM data not found. Run data_prep.py first.")
    else:
        rfm = rfm_df.copy()

        # ── Hero: Segment Treemap ────────────────────────────────────────────
        seg_agg = (rfm.groupby("Segment")
                   .agg(
                       Customers   = ("customer_unique_id", "count"),
                       Avg_Revenue = ("monetary",       "mean"),
                       Avg_Recency = ("recency_days",   "mean"),
                       Avg_Freq    = ("frequency",      "mean"),
                       Avg_Review  = ("avg_review_score","mean"),
                       Churn_pct   = ("is_churned",     lambda x: x.mean()*100
                                      if "is_churned" in rfm.columns else 0),
                   ).reset_index())

        fig = px.treemap(
            seg_agg, path=["Segment"], values="Customers",
            color="Avg_Revenue",
            color_continuous_scale=["#1C2A4A","#4C8BF5","#00C49F","#FFB347","#FF6B6B"],
            color_continuous_midpoint=seg_agg["Avg_Revenue"].median(),
            custom_data=["Avg_Revenue","Avg_Recency","Avg_Freq","Avg_Review","Customers"],
            template=PT,
        )
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[4]:,} customers",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{customdata[4]:,}<br>"
                "Avg Revenue: R$%{customdata[0]:.2f}<br>"
                "Avg Recency: %{customdata[1]:.0f} days<br>"
                "Avg Frequency: %{customdata[2]:.1f} orders<br>"
                "Avg Review: %{customdata[3]:.2f}⭐<extra></extra>"
            ),
            textfont_size=13,
        )
        fig.update_layout(height=400, margin=dict(t=10,b=10,l=10,r=10),
                          coloraxis_colorbar=dict(title="Avg Revenue<br>(R$)", thickness=12))
        st.plotly_chart(fig, use_container_width=True)

        insight("Rectangle size = number of customers in segment. Colour = average revenue. "
                "Champions are few but generate disproportionate revenue. "
                "'Lost' and 'Hibernating' together often represent 50–60% of the base — "
                "a retention programme targeting 'At Risk' before they slide to 'Lost' is highest ROI.")

        # ── Bubble chart: Recency vs Monetary ────────────────────────────────
        sec("Customer Map — Recency × Revenue × Frequency")
        sample = rfm.sample(min(3000, len(rfm)), random_state=42)

        fig = px.scatter(
            sample,
            x="recency_days", y="monetary",
            size="frequency", color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Segment": SEGMENT_ORDER},
            size_max=20, opacity=0.7,
            template=PT,
            labels={
                "recency_days": "Recency (days since last order) →  More Recent = Better",
                "monetary":     "Total Spend (R$)",
                "frequency":    "Order Count",
            },
            hover_data={"recency_days":True,"monetary":":.2f","frequency":True,"Segment":True},
        )
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        insight("Champions cluster bottom-left (recent + high spend). "
                "Lost customers cluster top-left (long ago + low spend). "
                "Bubble size shows frequency — large bubbles in the 'At Risk' zone "
                "are your highest-priority retention targets.")

        # ── Segment Profile Table ─────────────────────────────────────────────
        sec("Segment Profile Summary")
        profile = (rfm.groupby("Segment")
                   .agg(
                       Customers     = ("customer_unique_id", "count"),
                       Avg_Recency   = ("recency_days",   "mean"),
                       Avg_Frequency = ("frequency",      "mean"),
                       Avg_Revenue   = ("monetary",       "mean"),
                       Avg_Review    = ("avg_review_score","mean"),
                       Late_Pct      = ("late_pct",       "mean"),
                   ).reset_index()
                   .sort_values("Avg_Revenue", ascending=False))

        profile["Avg_Recency"]   = profile["Avg_Recency"].round(0).astype(int)
        profile["Avg_Frequency"] = profile["Avg_Frequency"].round(2)
        profile["Avg_Revenue"]   = profile["Avg_Revenue"].apply(lambda x: f"R${x:,.2f}")
        profile["Avg_Review"]    = profile["Avg_Review"].round(2)
        profile["Late_Pct"]      = profile["Late_Pct"].round(1).astype(str) + "%"
        profile["Customers"]     = profile["Customers"].apply(lambda x: f"{x:,}")

        profile.columns = ["Segment","Customers","Avg Recency (days)",
                           "Avg Orders","Avg Revenue","Avg Review ⭐","Late Delivery %"]
        st.dataframe(profile.set_index("Segment"), use_container_width=True)

        # ── RFM Score Distribution ────────────────────────────────────────────
        col_c, col_d = st.columns(2)

        with col_c:
            sec("RFM Total Score Distribution")
            fig = px.histogram(rfm, x="RFM_total", nbins=13,
                               color_discrete_sequence=["#4C8BF5"],
                               template=PT,
                               labels={"RFM_total":"RFM Total Score (3–15)"})
            fig.update_layout(height=280, margin=dict(t=10,b=10), bargap=0.05)
            fig.add_vline(x=rfm["RFM_total"].median(), line_dash="dash",
                          line_color="#FFB347",
                          annotation_text=f"Median {rfm['RFM_total'].median():.0f}",
                          annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            sec("Segment Distribution")
            seg_counts = rfm["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment","Count"]
            fig = px.pie(seg_counts, names="Segment", values="Count", hole=0.5,
                         color="Segment", color_discrete_map=SEGMENT_COLORS,
                         template=PT)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=10)
            fig.update_layout(showlegend=False, height=280,
                              margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DELIVERY PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with t3:
    sec("Delivery Performance Analysis")

    d = delivered.dropna(subset=["delay_days"]).copy()

    if len(d) == 0:
        st.warning("No delivery data available with current filters.")
    else:
        on_time_pct  = (~d["is_late"]).mean() * 100
        avg_delay    = d.loc[d["is_late"],  "delay_days"].mean()
        avg_early    = d.loc[d["is_early"], "delay_days"].abs().mean()
        total_del    = len(d)

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("On-Time Rate",    f"{on_time_pct:.1f}%",
                     delta="✓ target ≥ 90%" if on_time_pct>=90 else "↓ below 90% target",
                     pos=on_time_pct>=90)
        with c2: kpi("Avg Delay (Late)",f"{avg_delay:.1f} days",
                     delta="days past estimated", pos=False)
        with c3: kpi("Avg Early (days)",f"{avg_early:.1f} days",
                     delta="ahead of estimate", pos=True)
        with c4: kpi("Delivered Orders",f"{total_del:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            sec("Delay Distribution")
            d["Status"] = d["is_late"].map({True:"🔴 Late", False:"🟢 On Time / Early"})
            fig = px.histogram(d, x="delay_days", nbins=60,
                               color="Status",
                               color_discrete_map={"🔴 Late":"#FF6B6B",
                                                   "🟢 On Time / Early":"#00C49F"},
                               template=PT, barmode="overlay", opacity=0.8,
                               labels={"delay_days":"Delay (days — negative = arrived early)"})
            fig.add_vline(x=0, line_dash="dash", line_color="#FFB347",
                          annotation_text="Estimated date", annotation_position="top right")
            fig.update_layout(height=320, margin=dict(t=10,b=10),
                              legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sec("Monthly On-Time Rate Trend")
            monthly_ot = (d.groupby("order_month")
                          .agg(on_time_rate=("is_late", lambda x: (~x).mean()*100))
                          .reset_index())
            fig = px.line(monthly_ot, x="order_month", y="on_time_rate",
                          template=PT, markers=True,
                          color_discrete_sequence=["#4C8BF5"],
                          labels={"on_time_rate":"On-Time Rate (%)","order_month":""})
            fig.add_hline(y=90, line_dash="dash", line_color="#FFB347",
                          annotation_text="90% target")
            fig.update_layout(height=320, margin=dict(t=10,b=10),
                              xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                              yaxis=dict(range=[50,105]))
            st.plotly_chart(fig, use_container_width=True)

        insight("Early deliveries (negative delay) are a hidden positive signal — "
                "over-delivery on time builds trust and correlates with higher review scores. "
                "Periods where on-time rate drops below 90% often coincide with holiday peaks.")

        # ── State & Category breakdown ────────────────────────────────────────
        col_c, col_d = st.columns(2)

        with col_c:
            sec("On-Time Rate by Customer State (Worst 15)")
            state_ot = (d.groupby("customer_state")
                        .agg(on_time_rate=("is_late", lambda x: (~x).mean()*100),
                             orders=("order_id","nunique"))
                        .reset_index()
                        .query("orders >= 20")
                        .sort_values("on_time_rate")
                        .head(15))
            fig = px.bar(state_ot, x="on_time_rate", y="customer_state",
                         orientation="h", color="on_time_rate",
                         color_continuous_scale=["#FF6B6B","#FFB347","#00C49F"],
                         template=PT, text="on_time_rate",
                         labels={"on_time_rate":"On-Time %","customer_state":""})
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10), xaxis=dict(range=[0,105]))
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            sec("Avg Delay by Product Category (Top 15 worst)")
            cat_delay = (d[d["is_late"]]
                         .groupby("category")
                         .agg(avg_delay=("delay_days","mean"),
                              late_orders=("order_id","nunique"))
                         .reset_index()
                         .query("late_orders >= 10")
                         .sort_values("avg_delay", ascending=False)
                         .head(15))
            fig = px.bar(cat_delay, x="avg_delay", y="category",
                         orientation="h", color="avg_delay",
                         color_continuous_scale=["#FFB347","#FF6B6B"],
                         template=PT, text="avg_delay",
                         labels={"avg_delay":"Avg Delay (days)","category":""})
            fig.update_traces(texttemplate="%{x:.1f}d", textposition="outside")
            fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        insight("Remote northern states (AM, RR, AP) consistently show the worst on-time rates "
                "due to logistics infrastructure gaps — a structural challenge, not a seller issue. "
                "Category delays often reflect large/heavy items that miss carrier pickup windows.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHURN RISK  ← The novel insight
# ══════════════════════════════════════════════════════════════════════════════

with t4:
    sec("Churn Risk Analysis — Delivery as a Churn Driver")

    if rfm_df is None or "churn_probability" not in rfm_df.columns:
        st.warning("Churn model data not found. Run data_prep.py first.")
    else:
        rfm = rfm_df.copy()

        churn_rate   = rfm["is_churned"].mean() * 100 if "is_churned" in rfm.columns else 0
        at_risk_n    = (rfm["churn_probability"] > 0.7).sum()
        champions_n  = (rfm["Segment"] == "Champions").sum()
        auc_val      = 0.0  # placeholder — shown in prep output

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Overall Churn Rate", f"{churn_rate:.1f}%",
                     delta="180-day definition", pos=churn_rate < 50)
        with c2: kpi("High-Risk Customers", f"{at_risk_n:,}",
                     delta="P(churn) > 70%", pos=False)
        with c3: kpi("Champions (Safe)",    f"{champions_n:,}",
                     delta="lowest churn risk", pos=True)
        with c4: kpi("Model",               "Logistic Reg.",
                     delta="ROC-AUC in terminal output", pos=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── THE KEY INSIGHT: Segment × Churn Rate heatmap ────────────────────
        sec("⭐ Key Insight — Late Deliveries Drive Churn by Segment")

        insight("This is the central finding of this project: customers in weaker RFM segments "
                "who experienced late deliveries churn at dramatically higher rates than those "
                "who received on-time orders — even in the same segment. "
                "Fixing delivery in 'At Risk' + 'Hibernating' segments is the highest-ROI intervention.")

        # Churn rate × late delivery rate per segment
        seg_churn = (rfm.groupby("Segment")
                     .agg(
                         churn_rate     = ("is_churned",        "mean"),
                         late_pct       = ("late_pct",          "mean"),
                         avg_churn_prob = ("churn_probability", "mean"),
                         customers      = ("customer_unique_id","count"),
                     ).reset_index())
        seg_churn["churn_rate"] *= 100
        seg_churn = seg_churn.sort_values("churn_rate", ascending=False)

        col_a, col_b = st.columns(2)

        with col_a:
            # Scatter: Late % vs Churn Rate per segment (sized by customers)
            fig = px.scatter(
                seg_churn,
                x="late_pct", y="churn_rate",
                size="customers", color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                text="Segment", size_max=50,
                template=PT,
                labels={
                    "late_pct":    "Late Delivery Rate (%)",
                    "churn_rate":  "Churn Rate (%)",
                    "customers":   "Customers",
                },
            )
            fig.update_traces(textposition="top center", textfont_size=10)
            fig.update_layout(height=400, showlegend=False,
                              margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Bar: avg churn probability per segment
            seg_churn_sorted = seg_churn.sort_values("avg_churn_prob", ascending=True)
            fig = px.bar(
                seg_churn_sorted, x="avg_churn_prob", y="Segment",
                orientation="h", color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                template=PT, text="avg_churn_prob",
                labels={"avg_churn_prob":"Avg Churn Probability","Segment":""},
            )
            fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
            fig.update_layout(height=400, showlegend=False,
                              margin=dict(t=10,b=10),
                              xaxis=dict(range=[0,1.1], tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        # ── Feature Importance ───────────────────────────────────────────────
        if coef_df is not None:
            sec("What Drives Churn — Model Feature Importances")
            fig = px.bar(
                coef_df.sort_values("Coefficient"),
                x="Coefficient", y="Feature",
                orientation="h",
                color="Coefficient",
                color_continuous_scale=["#00C49F","#1C2A4A","#FF6B6B"],
                color_continuous_midpoint=0,
                template=PT, text="Coefficient",
                labels={"Coefficient":"Model Coefficient (positive = increases churn risk)"},
            )
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
            fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            insight("Recency is the strongest predictor — customers who haven't ordered recently "
                    "are far more likely to have churned. Late delivery % and avg delay days "
                    "are the next strongest — direct evidence that poor delivery experience drives churn.")

        # ── Churn probability distribution ───────────────────────────────────
        sec("Churn Probability Distribution by Segment")
        fig = px.box(
            rfm, y="Segment", x="churn_probability",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Segment": SEGMENT_ORDER},
            template=PT,
            labels={"churn_probability":"P(Churn)","Segment":""},
            points=False,
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="#FFB347",
                      annotation_text="High-risk threshold (0.7)")
        fig.update_layout(height=400, showlegend=False, margin=dict(t=10,b=10),
                          xaxis=dict(range=[0,1], tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

        # ── High-risk customers table ─────────────────────────────────────────
        sec("🚨 High-Risk Customers to Retain (P(churn) > 70%)")
        high_risk = (rfm[rfm["churn_probability"] > 0.7]
                     [["customer_unique_id","Segment","recency_days","frequency",
                       "monetary","avg_review_score","late_pct","churn_probability"]]
                     .sort_values("churn_probability", ascending=False)
                     .head(50)
                     .reset_index(drop=True))
        high_risk.index += 1
        high_risk["monetary"]          = high_risk["monetary"].apply(lambda x: f"R${x:,.2f}")
        high_risk["churn_probability"] = high_risk["churn_probability"].apply(lambda x: f"{x:.0%}")
        high_risk["late_pct"]          = high_risk["late_pct"].apply(lambda x: f"{x:.1f}%")
        high_risk["avg_review_score"]  = high_risk["avg_review_score"].round(2)
        high_risk.columns = ["Customer ID","Segment","Recency (days)","Orders",
                             "Total Spend","Avg Review","Late Delivery %","P(Churn)"]
        st.dataframe(high_risk, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GEOGRAPHIC INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with t5:
    sec("Geographic Intelligence — Brazil State Analysis")

    # Build state-level summary
    state_summary = (delivered.groupby("customer_state")
                     .agg(
                         orders       = ("order_id",       "nunique"),
                         revenue      = ("payment_value",  "sum"),
                         avg_revenue  = ("payment_value",  "mean"),
                         avg_review   = ("review_score",   "mean"),
                         on_time_rate = ("is_late", lambda x: (~x.fillna(False)).mean()*100),
                         avg_delay    = ("delay_days",     "mean"),
                     ).reset_index()
                     .rename(columns={"customer_state":"State"}))

    # Merge RFM state data
    if rfm_df is not None:
        rfm_state = (rfm_df.groupby("customer_state")
                     .agg(
                         avg_rfm   = ("RFM_total",        "mean"),
                         customers = ("customer_unique_id","count"),
                         avg_churn = ("churn_probability", "mean"),
                     ).reset_index()
                     .rename(columns={"customer_state":"State"}))
        state_summary = state_summary.merge(rfm_state, on="State", how="left")

    col_a, col_b = st.columns(2)

    with col_a:
        sec("Revenue by State")
        top_states_rev = state_summary.sort_values("revenue", ascending=True).tail(20)
        fig = px.bar(top_states_rev, x="revenue", y="State", orientation="h",
                     color="revenue", color_continuous_scale=["#1C2A4A","#4C8BF5","#00C49F"],
                     template=PT, text="revenue",
                     labels={"revenue":"Revenue (R$)","State":""})
        fig.update_traces(texttemplate="R$%{x:,.0f}", textposition="outside")
        fig.update_layout(height=520, showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=10,b=10),
                          xaxis=dict(range=[0, top_states_rev["revenue"].max()*1.2]))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sec("On-Time Delivery Rate by State")
        state_ot = state_summary.sort_values("on_time_rate", ascending=True)
        fig = px.bar(state_ot, x="on_time_rate", y="State", orientation="h",
                     color="on_time_rate",
                     color_continuous_scale=["#FF6B6B","#FFB347","#00C49F"],
                     template=PT, text="on_time_rate",
                     labels={"on_time_rate":"On-Time Rate (%)","State":""})
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig.update_layout(height=520, showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=10,b=10), xaxis=dict(range=[0,110]))
        st.plotly_chart(fig, use_container_width=True)

    # Full state summary table
    sec("State-Level Summary Table")
    display = state_summary.copy().sort_values("revenue", ascending=False)
    display["revenue"]      = display["revenue"].apply(lambda x: f"R${x:,.0f}")
    display["avg_revenue"]  = display["avg_revenue"].apply(lambda x: f"R${x:,.2f}")
    display["avg_review"]   = display["avg_review"].round(2)
    display["on_time_rate"] = display["on_time_rate"].apply(lambda x: f"{x:.1f}%")
    display["avg_delay"]    = display["avg_delay"].round(1)
    if "avg_rfm" in display.columns:
        display["avg_rfm"]   = display["avg_rfm"].round(2)
        display["avg_churn"] = display["avg_churn"].apply(lambda x: f"{x:.0%}")

    display.columns = (["State","Orders","Total Revenue","Avg Order Value",
                        "Avg Review ⭐","On-Time Rate","Avg Delay (days)"] +
                       (["Avg RFM Score","Customers","Avg Churn Risk"]
                        if "avg_rfm" in state_summary.columns else []))
    st.dataframe(display.set_index("State"), use_container_width=True)

    insight("The geographic view reveals a north-south divide: southern states (SP, RJ, MG, RS) "
            "drive the bulk of revenue AND have the best delivery rates. Northern states contribute "
            "less revenue but suffer higher churn risk — a direct consequence of poor logistics coverage. "
            "Improving last-mile delivery in AM, PA, MA could unlock significant untapped revenue.")
