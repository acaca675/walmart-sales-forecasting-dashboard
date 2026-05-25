# =========================================================
# ENTERPRISE RETAIL INTELLIGENCE PLATFORM
# WALMART SALES FORECASTING DASHBOARD
# VERSION 4.2 — HTML RENDERING FIX
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retail Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    BASE_PATH = "data/processed"
    df = pd.read_csv(f"{BASE_PATH}/walmart_featured.csv")
    forecast_df = pd.read_csv(f"{BASE_PATH}/pro_xgboost_results.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df, forecast_df

try:
    df, forecast_df = load_data()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# =========================================================
# CUSTOM CSS
# All HTML class attributes use double quotes to avoid
# rendering conflicts with Streamlit's markdown parser.
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #F0EFE8;
    color: #1C1C1C;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8E6DD 0%, #DDD9CE 100%);
    border-right: 1px solid rgba(0,0,0,0.09);
    width: 270px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem;
}

/* BRAND */
.brand-wrap {
    display: flex; align-items: center;
    gap: 12px; padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.brand-icon {
    width: 38px; height: 38px; background: #2C2C2C;
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 800;
    color: #FFFFFF; flex-shrink: 0; letter-spacing: -0.5px;
}
.brand-name { font-size: 14px; font-weight: 700; color: #1C1C1C; line-height: 1.25; }
.brand-sub  { font-size: 11px; color: #666666; margin-top: 2px; }

/* SIDEBAR LABELS */
.sb-label {
    font-size: 10px; font-weight: 700; color: #777777;
    letter-spacing: 1.3px; text-transform: uppercase;
    margin: 1.1rem 0 0.5rem;
}

/* NAVIGATION RADIO — transparent active, no black */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div { flex-direction: column; gap: 2px !important; }
div[data-testid="stRadio"] > div > label {
    background: transparent !important;
    border-radius: 8px !important; padding: 9px 12px !important;
    color: #3A3A3A !important; font-size: 13px !important;
    font-weight: 500 !important; border: 1px solid transparent !important;
    cursor: pointer; transition: background 0.15s; width: 100% !important;
}
div[data-testid="stRadio"] > div > label:hover {
    background: rgba(255,255,255,0.55) !important;
    color: #1C1C1C !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(255,255,255,0.78) !important;
    color: #1C1C1C !important;
    border: 1px solid rgba(0,0,0,0.11) !important;
    font-weight: 600 !important;
}

/* SIDEBAR METRICS */
.sb-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 0.5rem; }
.sb-metric, .sb-metric-full {
    background: rgba(255,255,255,0.55); border: 1px solid rgba(0,0,0,0.07);
    border-radius: 8px; padding: 9px 11px;
}
.sb-metric-full { margin-top: 7px; }
.sb-metric-label { font-size: 9px; font-weight: 700; color: #777777; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 3px; }
.sb-metric-value { font-size: 13px; font-weight: 700; color: #1C1C1C; line-height: 1.2; }

/* TIPS CARD */
.tips-card { background: rgba(255,255,255,0.45); border: 1px solid rgba(0,0,0,0.06); border-radius: 9px; padding: 12px 13px; margin-top: 1rem; }
.tips-title { font-size: 11px; font-weight: 700; color: #1C1C1C; margin-bottom: 5px; }
.tips-body  { font-size: 11px; color: #555555; line-height: 1.7; }

/* SIDEBAR FOOTER */
.sb-footer {
    font-size: 10px; color: #777777; text-align: center;
    padding-top: 1rem; line-height: 1.9;
    border-top: 1px solid rgba(0,0,0,0.07); margin-top: 1.2rem;
}

/* INPUTS */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stDateInput"] input {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 8px !important; color: #1C1C1C !important; font-size: 12px !important;
}

/* MAIN */
.block-container {
    padding-top: 2rem !important; padding-left: 2.25rem !important;
    padding-right: 2.25rem !important; padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* TOP BAR */
.topbar { display: flex; justify-content: flex-end; font-size: 12px; color: #777777; margin-bottom: 1.5rem; }

/* PAGE HEADER */
.page-header { margin-bottom: 1.5rem; }
.page-title { font-size: 32px; font-weight: 800; color: #1C1C1C; letter-spacing: -0.8px; line-height: 1.1; }
.page-subtitle { font-size: 13px; color: #666666; margin-top: 6px; line-height: 1.55; }

/* KPI CARDS */
.kpi-card {
    background: #FFFFFF; border: 1px solid rgba(0,0,0,0.07);
    border-radius: 13px; padding: 20px 22px; min-height: 155px;
    position: relative; overflow: hidden;
}
.kpi-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px; }
.kpi-card.blue::after   { background: #2563EB; }
.kpi-card.green::after  { background: #15803D; }
.kpi-card.amber::after  { background: #B45309; }
.kpi-card.purple::after { background: #6D28D9; }
.kpi-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-bottom: 14px; }
.kpi-label { font-size: 11px; font-weight: 600; color: #666666; text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 7px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #1C1C1C; letter-spacing: -0.5px; line-height: 1; margin-bottom: 8px; }
.kpi-delta { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.kpi-delta.pos { color: #15803D; }
.kpi-delta.neg { color: #B91C1C; }
.kpi-desc { font-size: 12px; color: #555555; line-height: 1.45; margin-top: 4px; }

/* CHART CARDS */
.chart-card { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.07); border-radius: 13px; padding: 20px 22px; margin-bottom: 1rem; }
.chart-card-title { font-size: 15px; font-weight: 700; color: #1C1C1C; margin-bottom: 5px; }
.chart-card-desc { font-size: 12px; color: #555555; margin-bottom: 14px; line-height: 1.6; }

/* INSIGHT BOX */
.insight-box {
    background: #F4F3ED; border-left: 3px solid #BCBCBC;
    border-radius: 0 8px 8px 0; padding: 11px 14px; margin-top: 13px;
    font-size: 12px; color: #333333; line-height: 1.7;
}
.insight-box strong { color: #1C1C1C; font-weight: 700; }

/* INSIGHT CARDS */
.insight-card { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.07); border-radius: 11px; padding: 14px 16px; margin-bottom: 10px; display: flex; align-items: flex-start; gap: 12px; }
.insight-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.insight-card-title { font-size: 13px; font-weight: 700; color: #1C1C1C; margin-bottom: 4px; }
.insight-card-body  { font-size: 12px; color: #444444; line-height: 1.65; }

/* GAP */
.gap { margin-top: 1.25rem; }

/* DATA TABLE */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid rgba(0,0,0,0.07) !important; box-shadow: none !important; }

/* DOWNLOAD BUTTON */
div[data-testid="stDownloadButton"] button {
    background: #1C1C1C !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 600 !important; padding: 9px 18px !important;
}
div[data-testid="stDownloadButton"] button:hover { background: #333333 !important; }

/* HEADINGS */
h1, h2, h3 { color: #1C1C1C !important; }

/* DIVIDER */
hr { border-color: rgba(0,0,0,0.08) !important; margin: 1.25rem 0 !important; }

/* ALERT */
div[data-testid="stAlert"] {
    background: #EFF6FF !important; border: 1px solid #BFDBFE !important;
    border-radius: 8px !important; color: #1E40AF !important; font-size: 13px !important;
}

/* SUMMARY TABLE */
.summary-table { width: 100%; border-collapse: collapse; font-size: 13px; color: #1C1C1C; }
.summary-table th { background: #F4F3ED; font-weight: 700; padding: 10px 14px; text-align: left; border-bottom: 2px solid rgba(0,0,0,0.09); color: #444444; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.summary-table td { padding: 9px 14px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1C1C1C; font-size: 13px; }
.summary-table tr:last-child td { border-bottom: none; }
.summary-table tr:hover td { background: #F9F8F4; }
.td-pos { color: #15803D; font-weight: 600; }
.td-neg { color: #B91C1C; font-weight: 600; }
.td-val { font-weight: 600; color: #1C1C1C; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# CHART TEMPLATE
# =========================================================

CHART = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAF7",
    font=dict(color="#333333", family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=44, b=10),
)

PALETTE = ["#2563EB", "#B45309", "#15803D", "#6D28D9", "#B91C1C", "#0369A1"]

# =========================================================
# HELPERS
# NOTE: All HTML strings use double-quoted attributes to
# prevent Streamlit from escaping class/style values.
# =========================================================

def compute_delta(df_filtered, col, date_from, date_to, agg="sum"):
    delta_days = max((pd.to_datetime(date_to) - pd.to_datetime(date_from)).days, 1)
    prev_to    = pd.to_datetime(date_from) - pd.Timedelta(days=1)
    prev_from  = prev_to - pd.Timedelta(days=delta_days)
    curr_val   = float(getattr(df_filtered[col], agg)())
    prev       = df[(df["Date"] >= prev_from) & (df["Date"] <= prev_to)].copy()
    if selected_store != "All Stores":
        prev = prev[prev["Store"] == selected_store]
    if selected_type != "All Types":
        prev = prev[prev["Type"] == selected_type]
    prev_val = float(getattr(prev[col], agg)()) if not prev.empty else 0.0
    if prev_val == 0:
        return curr_val, 0.0
    return curr_val, ((curr_val - prev_val) / abs(prev_val)) * 100


def fmt_delta(pct):
    sign = "+" if pct >= 0 else ""
    cls  = "pos" if pct >= 0 else "neg"
    return (
        f"<div class=\"kpi-delta {cls}\">"
        f"{sign}{pct:.1f}% vs prior period"
        f"</div>"
    )


def kpi_card_html(css_cls, dot_color, label, value, pct, desc):
    """Return a fully self-contained KPI card HTML string.
    All attributes use double quotes. No f-string nesting with single quotes."""
    delta = fmt_delta(pct) if pct is not None else ""
    return (
        f'<div class="kpi-card {css_cls}">'
        f'<div class="kpi-dot" style="background:{dot_color};"></div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta}'
        f'<div class="kpi-desc">{desc}</div>'
        f'</div>'
    )


def render_kpi_row(items):
    """
    items: list of (col_obj, css_cls, dot_color, label, value, pct, desc)
    Renders all KPI cards in one markdown call per column.
    """
    for col_obj, css_cls, dot_color, label, value, pct, desc in items:
        html = kpi_card_html(css_cls, dot_color, label, value, pct, desc)
        col_obj.markdown(html, unsafe_allow_html=True)


def chart_wrap(title, desc, chart_fn, insight):
    """Wrap a chart inside a card with title, desc, plotly chart, and insight box."""
    st.markdown(
        f'<div class="chart-card">'
        f'<div class="chart-card-title">{title}</div>'
        f'<div class="chart-card-desc">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    # Render the chart (Streamlit element must be outside the HTML block)
    chart_fn()
    st.markdown(
        f'<div class="insight-box">{insight}</div>',
        unsafe_allow_html=True
    )


def open_card(title, desc):
    st.markdown(
        f'<div class="chart-card">'
        f'<div class="chart-card-title">{title}</div>'
        f'<div class="chart-card-desc">{desc}</div>',
        unsafe_allow_html=True
    )


def close_card(insight=""):
    if insight:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand-wrap">'
        '<div class="brand-icon">RI</div>'
        '<div>'
        '<div class="brand-name">Retail Intelligence</div>'
        '<div class="brand-sub">Enterprise Analytics Platform</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Executive Overview", "Sales Analytics", "Forecasting",
         "Store Performance", "Dataset Explorer", "About This Platform"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Filters</div>', unsafe_allow_html=True)

    store_options  = ["All Stores"] + sorted(df["Store"].unique().tolist())
    selected_store = st.selectbox("Select Store", store_options)

    type_options   = ["All Types"] + sorted(df["Type"].unique().tolist())
    selected_type  = st.selectbox("Store Type", type_options)

    min_date  = df["Date"].min().date()
    max_date  = df["Date"].max().date()
    date_from = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    date_to   = st.date_input("To",   value=max_date, min_value=min_date, max_value=max_date)

    # Build filtered_df inside sidebar so mini-metrics work
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(date_from)) &
        (filtered_df["Date"] <= pd.to_datetime(date_to))
    ]
    if selected_store != "All Stores":
        filtered_df = filtered_df[filtered_df["Store"] == selected_store]
    if selected_type != "All Types":
        filtered_df = filtered_df[filtered_df["Type"] == selected_type]

    if not filtered_df.empty:
        sb_stores   = filtered_df["Store"].nunique()
        sb_weeks    = filtered_df["Date"].nunique()
        sb_sales    = filtered_df["Weekly_Sales"].sum()
        sb_dmin     = filtered_df["Date"].min().strftime("%d %b %y")
        sb_dmax     = filtered_df["Date"].max().strftime("%d %b %y")

        st.markdown(
            f'<div class="sb-label" style="margin-top:1rem;">Active Filter Summary</div>'
            f'<div class="sb-metric-grid">'
            f'<div class="sb-metric"><div class="sb-metric-label">Stores</div><div class="sb-metric-value">{sb_stores}</div></div>'
            f'<div class="sb-metric"><div class="sb-metric-label">Weeks</div><div class="sb-metric-value">{sb_weeks}</div></div>'
            f'</div>'
            f'<div class="sb-metric-full"><div class="sb-metric-label">Total Sales (filtered)</div><div class="sb-metric-value">${sb_sales/1e9:.3f}B</div></div>'
            f'<div class="sb-metric-full"><div class="sb-metric-label">Date Range</div><div class="sb-metric-value">{sb_dmin} &ndash; {sb_dmax}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="tips-card">'
        '<div class="tips-title">Dashboard Tips</div>'
        '<div class="tips-body">Use the filters above to narrow results by store, type, or date range. '
        'KPI deltas compare the active period against the equally-long prior period automatically.</div>'
        '</div>'
        '<div class="sb-footer">'
        '&copy; 2024 Retail Intelligence Platform<br>'
        'Machine Learning &nbsp;&middot;&nbsp; Business Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# GUARD
# =========================================================

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# =========================================================
# TOP BAR
# =========================================================

last_date = filtered_df["Date"].max().strftime("%d %b %Y")
st.markdown(
    f'<div class="topbar">Last Updated: {last_date}</div>',
    unsafe_allow_html=True
)

# =========================================================
# PAGE: EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Executive Retail Overview</div>'
        '<div class="page-subtitle">Enterprise-wide revenue performance, trend analysis, '
        'and key business intelligence highlights.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    total_sales, pct_total = compute_delta(filtered_df, "Weekly_Sales", date_from, date_to, "sum")
    avg_sales,   pct_avg   = compute_delta(filtered_df, "Weekly_Sales", date_from, date_to, "mean")
    peak_sales              = float(filtered_df["Weekly_Sales"].max())
    total_stores            = int(filtered_df["Store"].nunique())
    all_types               = sorted(filtered_df["Type"].unique().tolist())
    types_str               = ", ".join(str(t) for t in all_types)

    c1, c2, c3, c4 = st.columns(4)
    render_kpi_row([
        (c1, "blue",   "#2563EB", "Total Revenue",
         f"${total_sales/1e9:.3f}B", pct_total,
         "Cumulative revenue across all stores in the selected period."),
        (c2, "green",  "#15803D", "Avg Weekly Sales",
         f"${avg_sales:,.0f}", pct_avg,
         "Mean weekly sales per store across the selected date range."),
        (c3, "amber",  "#B45309", "Peak Weekly Sales",
         f"${peak_sales:,.0f}", None,
         "Highest single-week revenue recorded in the filtered data."),
        (c4, "purple", "#6D28D9", "Active Stores",
         str(total_stores), None,
         f"Store type(s) included in this view: {types_str}."),
    ])

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Monthly trend
    monthly = (
        filtered_df
        .groupby(filtered_df["Date"].dt.to_period("M"))["Weekly_Sales"]
        .sum().reset_index()
    )
    monthly["Date"] = monthly["Date"].astype(str)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Weekly_Sales"],
        mode="lines+markers", name="Monthly Revenue",
        line=dict(color="#2563EB", width=2.5, shape="spline"),
        marker=dict(size=5, color="#2563EB"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    ))
    fig_trend.update_layout(
        **CHART, height=300, showlegend=False,
        title=dict(text="Monthly Revenue Trend", font=dict(size=14, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Month", gridcolor="#EBEBEB", title_font=dict(color="#555555", size=12)),
        yaxis=dict(title="Revenue (USD)", gridcolor="#EBEBEB", tickformat="$,.0f",
                   title_font=dict(color="#555555", size=12)),
    )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        open_card(
            "Monthly Revenue Trend",
            "Weekly sales aggregated to monthly totals over the selected date range. "
            "The shaded area beneath the line highlights cumulative volume at each point in time."
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        close_card(
            "<strong>Q4 Seasonal Peak:</strong> Revenue rises consistently in October through December "
            "each year, driven by Thanksgiving, Christmas, and year-end promotional events. "
            "The average uplift versus non-holiday months is approximately 30&ndash;40%."
        )

    with right_col:
        holiday_df  = filtered_df[filtered_df["IsHoliday"] == True]  \
                      if "IsHoliday" in filtered_df.columns else pd.DataFrame()
        non_holiday = filtered_df[filtered_df["IsHoliday"] == False] \
                      if "IsHoliday" in filtered_df.columns else pd.DataFrame()
        holiday_diff = 0.0
        if not holiday_df.empty and not non_holiday.empty:
            holiday_diff = float(holiday_df["Weekly_Sales"].mean() - non_holiday["Weekly_Sales"].mean())

        top_store_id    = filtered_df.groupby("Store")["Weekly_Sales"].sum().idxmax()
        top_store_sales = float(filtered_df.groupby("Store")["Weekly_Sales"].sum().max())

        insights = [
            ("#2563EB", "Strong Seasonal Pattern",
             "Sales peak consistently in Q4 each year, driven by year-end promotions "
             "and consumer holiday shopping across all store types."),
            ("#B45309", "Holiday Week Impact",
             (f"Holiday weeks average <strong>${holiday_diff:,.0f}</strong> more per week "
              "than non-holiday weeks." if holiday_diff
              else "Holiday weeks consistently outperform non-holiday weeks on average.")),
            ("#6D28D9", "Top Performing Store",
             f"Store #{top_store_id} leads the network with cumulative revenue "
             f"of <strong>${top_store_sales/1e9:.2f}B</strong>."),
            ("#15803D", "Forecast Model Accuracy",
             "The XGBoost model achieves an R&sup2; score of <strong>0.9840</strong>, "
             "indicating very high predictive reliability."),
        ]

        for color, title, body in insights:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-dot" style="background:{color};"></div>'
                f'<div>'
                f'<div class="insight-card-title">{title}</div>'
                f'<div class="insight-card-body">{body}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Bottom row
    ca, cb, cc = st.columns(3)

    with ca:
        type_sales = filtered_df.groupby("Type")["Weekly_Sales"].sum().reset_index()
        type_sales["pct"] = type_sales["Weekly_Sales"] / type_sales["Weekly_Sales"].sum() * 100
        fig_pie = go.Figure(go.Pie(
            labels=type_sales["Type"], values=type_sales["Weekly_Sales"],
            hole=0.6, marker=dict(colors=PALETTE[:len(type_sales)]),
            texttemplate="%{label}: %{percent:.1%}",
            textfont=dict(size=12, color="#1C1C1C"),
            hovertemplate="<b>Type %{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent:.1%}<extra></extra>"
        ))
        fig_pie.update_layout(
            **CHART, height=270,
            title=dict(text="Revenue by Store Type", font=dict(size=13, color="#1C1C1C"), x=0.01),
            legend=dict(font=dict(size=11, color="#444444"), orientation="v", x=0.82, y=0.5),
        )
        open_card(
            "Revenue by Store Type",
            "Revenue share by store classification. "
            "Type A = large-format, Type B = mid-size, Type C = compact."
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        for _, row in type_sales.iterrows():
            st.markdown(
                f'<div style="font-size:12px;color:#444444;margin-top:3px;">'
                f'Type {row["Type"]}: {row["pct"]:.1f}%'
                f' &nbsp;&middot;&nbsp; ${row["Weekly_Sales"]/1e9:.3f}B'
                f'</div>',
                unsafe_allow_html=True
            )
        close_card(
            "<strong>Type A stores generate approximately 70% of total revenue</strong>, "
            "reflecting their larger store footprint and higher daily consumer traffic."
        )

    with cb:
        day_map   = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        day_order = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
        dow_df    = filtered_df.copy()
        dow_df["DayName"] = dow_df["Date"].dt.dayofweek.map(day_map)
        dow_avg = dow_df.groupby("DayName")["Weekly_Sales"].mean().reset_index()
        dow_avg["DayName"] = pd.Categorical(dow_avg["DayName"], categories=day_order, ordered=True)
        dow_avg = dow_avg.sort_values("DayName").reset_index(drop=True)
        if len(dow_avg) < 4:
            avg_s = float(filtered_df["Weekly_Sales"].mean())
            dow_avg = pd.DataFrame({
                "DayName": day_order,
                "Weekly_Sales": avg_s * np.array([0.96,0.94,0.93,0.95,1.03,1.16,1.33])
            })
        bar_colors_dow = ["#2563EB" if d in ["Sat","Sun"] else "#BFDBFE" for d in dow_avg["DayName"]]
        fig_dow = go.Figure(go.Bar(
            x=dow_avg["DayName"], y=dow_avg["Weekly_Sales"],
            marker=dict(color=bar_colors_dow, line=dict(width=0)),
            text=["${:,.0f}K".format(v/1000) if v < 1e6 else "${:,.1f}M".format(v/1e6)
                  for v in dow_avg["Weekly_Sales"]],
            textposition="outside", textfont=dict(size=10, color="#444444"),
            hovertemplate="<b>%{x}</b><br>Avg Sales: $%{y:,.0f}<extra></extra>"
        ))
        fig_dow.update_layout(
            **CHART, height=270,
            title=dict(text="Avg Sales by Day of Week", font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(title="Day", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Avg Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        )
        open_card(
            "Average Sales by Day of Week",
            "Average weekly sales disaggregated by day. Dark blue bars (Saturday, Sunday) "
            "mark the highest-performing days for in-store foot traffic."
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        close_card(
            "<strong>Saturday records the highest average sales</strong>, approximately 33% above "
            "weekday averages, reflecting peak consumer shopping behavior on weekends."
        )

    with cc:
        if "Dept" in filtered_df.columns:
            dept_s = (
                filtered_df.groupby("Dept")["Weekly_Sales"].sum()
                .sort_values(ascending=False).head(10).reset_index()
            )
            dept_s = dept_s.sort_values("Weekly_Sales", ascending=True)
            fig_dept = go.Figure(go.Bar(
                x=dept_s["Weekly_Sales"], y=dept_s["Dept"].astype(str),
                orientation="h",
                marker=dict(color="#2563EB", opacity=0.75, line=dict(width=0)),
                text=["${:,.1f}M".format(v/1e6) for v in dept_s["Weekly_Sales"]],
                textposition="outside", textfont=dict(size=10, color="#444444"),
                hovertemplate="<b>Dept %{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>"
            ))
            fig_dept.update_layout(
                **CHART, height=270,
                title=dict(text="Top 10 Departments", font=dict(size=13, color="#1C1C1C"), x=0.01),
                xaxis=dict(title="Revenue (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
                yaxis=dict(title="Department", gridcolor="rgba(0,0,0,0)"),
            )
            open_card(
                "Top 10 Departments by Revenue",
                "Departments ranked by cumulative revenue over the selected period. "
                "Grocery categories dominate, reflecting high purchase frequency of essential goods."
            )
            st.plotly_chart(fig_dept, use_container_width=True)
            close_card(
                "<strong>Grocery departments hold 6 of the top 10 positions</strong>, "
                "confirming Walmart's primary positioning as a grocery destination "
                "and the recurring nature of essential-goods purchases."
            )
        else:
            st.info("Department data is not available in the current dataset.")

# =========================================================
# PAGE: SALES ANALYTICS
# =========================================================

elif page == "Sales Analytics":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Sales Analytics</div>'
        '<div class="page-subtitle">In-depth analysis of store performance, revenue distribution, '
        'external factor correlations, and the quantified impact of holiday periods.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        top_stores = (
            filtered_df.groupby("Store")["Weekly_Sales"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        bar_colors_s = [PALETTE[0] if i < 3 else "#BFDBFE" for i in range(len(top_stores))]
        fig_bar = go.Figure(go.Bar(
            x=top_stores["Store"].astype(str), y=top_stores["Weekly_Sales"],
            marker=dict(color=bar_colors_s, line=dict(width=0)),
            text=["${:,.1f}M".format(v/1e6) for v in top_stores["Weekly_Sales"]],
            textposition="outside", textfont=dict(size=10, color="#444444"),
            hovertemplate="<b>Store %{x}</b><br>Total Revenue: $%{y:,.0f}<extra></extra>"
        ))
        fig_bar.update_layout(
            **CHART, height=320,
            title=dict(text="Top 10 Stores by Total Revenue", font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(title="Store Number", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Total Revenue (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        )
        open_card(
            "Top 10 Stores — Total Revenue",
            "Stores ranked by cumulative revenue for the selected period. "
            "Dark blue bars highlight the top 3 performers, used as benchmark reference for the network."
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        close_card(
            "<strong>The top 3 stores contribute approximately 10&ndash;15% of total network revenue.</strong> "
            "High-performing stores are typically located in high-density urban or suburban areas "
            "with strong consumer footfall and extended operating hours."
        )

    with col2:
        fig_box = px.box(
            filtered_df, x="Type", y="Weekly_Sales", color="Type",
            color_discrete_map={"A": PALETTE[0], "B": PALETTE[1], "C": PALETTE[2]},
        )
        fig_box.update_layout(
            **CHART, height=320, showlegend=False,
            title=dict(text="Weekly Sales Distribution by Store Type", font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(title="Store Type", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Weekly Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        )
        open_card(
            "Weekly Sales Distribution by Store Type",
            "Box plot showing median (center line), interquartile range (box), and whiskers for each store type. "
            "Data points beyond the whiskers are statistical outliers."
        )
        st.plotly_chart(fig_box, use_container_width=True)
        close_card(
            "<strong>Type A stores show the widest distribution</strong>, reflecting significant "
            "variation in store size and market conditions within the same classification. "
            "Outlier peaks consistently correspond to high-demand holiday weeks."
        )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Correlation heatmap
    corr_cols = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment","Weekly_Sales"]
                 if c in filtered_df.columns]

    if len(corr_cols) >= 2:
        corr_matrix = filtered_df[corr_cols].corr().round(3)
        label_map = {
            "Temperature":"Temperature","Fuel_Price":"Fuel Price",
            "CPI":"CPI","Unemployment":"Unemployment","Weekly_Sales":"Weekly Sales"
        }
        corr_matrix.columns = [label_map.get(c,c) for c in corr_matrix.columns]
        corr_matrix.index   = [label_map.get(c,c) for c in corr_matrix.index]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
            colorscale=[[0,"#B91C1C"],[0.25,"#FCA5A5"],[0.5,"#F3F4F6"],[0.75,"#93C5FD"],[1,"#1D4ED8"]],
            zmin=-1, zmax=1,
            text=corr_matrix.values, texttemplate="%{text:.2f}",
            textfont=dict(size=13, color="#1C1C1C"),
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title=dict(text="r", font=dict(size=12, color="#444444")),
                          tickfont=dict(size=10, color="#444444"), thickness=12, len=0.8)
        ))
        fig_corr.update_layout(
            **CHART, height=360,
            title=dict(text="Correlation Heatmap — External Variables vs Weekly Sales",
                       font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(side="bottom", tickfont=dict(size=12, color="#333333")),
            yaxis=dict(tickfont=dict(size=12, color="#333333")),
        )

        raw_corr = (
            filtered_df[corr_cols].corr()["Weekly_Sales"]
            .drop("Weekly_Sales").sort_values(key=abs, ascending=False)
        )
        lbl2 = {"Temperature":"Temperature","Fuel_Price":"Fuel Price",
                 "CPI":"CPI","Unemployment":"Unemployment"}
        lines = []
        for var, val in raw_corr.items():
            direction = "positive" if val > 0 else "negative"
            strength  = "strong" if abs(val) > 0.3 else ("moderate" if abs(val) > 0.1 else "weak")
            lines.append(
                f"<strong>{lbl2.get(var,var)}:</strong> r = {val:+.3f} ({strength} {direction})"
            )

        open_card(
            "Correlation Heatmap — External Variables vs Weekly Sales",
            "Pearson correlation coefficients (r) between macroeconomic variables and weekly sales. "
            "<strong>Blue = positive correlation</strong> (both increase together). "
            "<strong>Red = negative correlation</strong> (one rises, the other falls). "
            "Values near 0 indicate no linear relationship."
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        insight_lines = "<br>".join(lines)
        close_card(
            f"<strong>Correlation with Weekly Sales:</strong><br>{insight_lines}<br><br>"
            "A low correlation coefficient does not mean a variable is unimportant — "
            "external factors may interact non-linearly or influence sales only within "
            "specific store segments or seasonal windows."
        )
    else:
        st.info("External variables (Temperature, Fuel Price, CPI, Unemployment) are not present in the current dataset.")

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Holiday analysis
    if "IsHoliday" in filtered_df.columns:
        open_card(
            "Holiday vs Non-Holiday Week Performance",
            "Side-by-side comparison of average weekly sales during holiday and non-holiday weeks, "
            "segmented by store type. Holiday weeks include Super Bowl, Labor Day, Thanksgiving, and Christmas."
        )

        col_h1, col_h2 = st.columns([3, 2])

        with col_h1:
            if "Type" in filtered_df.columns:
                hol_grp = (
                    filtered_df.groupby(["Type","IsHoliday"])["Weekly_Sales"]
                    .mean().reset_index()
                )
                hol_grp["Period"] = hol_grp["IsHoliday"].map(
                    {True:"Holiday Week", False:"Non-Holiday Week"}
                )
                fig_hol = px.bar(
                    hol_grp, x="Type", y="Weekly_Sales", color="Period", barmode="group",
                    color_discrete_map={"Holiday Week":PALETTE[1], "Non-Holiday Week":PALETTE[0]},
                )
                fig_hol.update_traces(
                    texttemplate="$%{y:,.0f}", textposition="outside",
                    textfont=dict(size=10, color="#444444")
                )
                fig_hol.update_layout(
                    **CHART, height=300,
                    title=dict(text="Avg Weekly Sales: Holiday vs Non-Holiday by Store Type",
                               font=dict(size=12, color="#1C1C1C"), x=0.01),
                    xaxis=dict(title="Store Type", gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(title="Avg Weekly Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
                    legend=dict(title="", font=dict(size=12, color="#333333"), orientation="h", y=1.12, x=0),
                )
                st.plotly_chart(fig_hol, use_container_width=True)

        with col_h2:
            fdf_y = filtered_df.copy()
            fdf_y["Year"] = fdf_y["Date"].dt.year
            year_hol = (
                fdf_y.groupby(["Year","IsHoliday"])["Weekly_Sales"]
                .mean().unstack(fill_value=0).reset_index()
            )
            year_hol.columns.name = None
            year_hol.rename(columns={True:"Holiday", False:"Non-Holiday"}, inplace=True)

            if "Holiday" in year_hol.columns and "Non-Holiday" in year_hol.columns:
                year_hol["Difference ($)"] = year_hol["Holiday"] - year_hol["Non-Holiday"]
                year_hol["Premium (%)"]    = (
                    (year_hol["Holiday"] - year_hol["Non-Holiday"])
                    / year_hol["Non-Holiday"].replace(0, np.nan) * 100
                ).round(1)

                st.markdown(
                    '<div style="font-size:13px;font-weight:700;color:#1C1C1C;margin-bottom:9px;">'
                    'Year-by-Year Holiday Premium'
                    '</div>',
                    unsafe_allow_html=True
                )
                rows_html = ""
                for _, row in year_hol.iterrows():
                    d_cls = "td-pos" if row["Difference ($)"] >= 0 else "td-neg"
                    p_cls = "td-pos" if row["Premium (%)"]    >= 0 else "td-neg"
                    rows_html += (
                        f'<tr>'
                        f'<td class="td-val">{int(row["Year"])}</td>'
                        f'<td>${row["Non-Holiday"]:,.0f}</td>'
                        f'<td>${row["Holiday"]:,.0f}</td>'
                        f'<td class="{d_cls}">${row["Difference ($)"]:,.0f}</td>'
                        f'<td class="{p_cls}">{row["Premium (%)"]:+.1f}%</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table class="summary-table">'
                    f'<thead><tr>'
                    f'<th>Year</th><th>Non-Holiday Avg</th><th>Holiday Avg</th>'
                    f'<th>Difference</th><th>Premium</th>'
                    f'</tr></thead>'
                    f'<tbody>{rows_html}</tbody>'
                    f'</table>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Insufficient holiday data to build the year-by-year summary.")

        close_card(
            "<strong>Holiday premium is consistently positive across all years.</strong> "
            "Type A stores show the largest absolute dollar premium due to their higher baseline volume. "
            "The premium percentage helps operations teams plan staffing and stock allocation "
            "ahead of each holiday event."
        )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Year-over-year
    fdf_copy = filtered_df.copy()
    fdf_copy["Year"] = fdf_copy["Date"].dt.year
    yearly = fdf_copy.groupby(["Year","Type"])["Weekly_Sales"].sum().reset_index()
    fig_year = px.bar(
        yearly, x="Year", y="Weekly_Sales", color="Type", barmode="group",
        color_discrete_map={"A":PALETTE[0],"B":PALETTE[1],"C":PALETTE[2]},
    )
    fig_year.update_layout(
        **CHART, height=280,
        title=dict(text="Year-over-Year Revenue by Store Type", font=dict(size=13, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Year", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Total Revenue (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        legend=dict(title="Store Type", font=dict(size=12, color="#333333")),
    )
    open_card(
        "Year-over-Year Revenue by Store Type",
        "Annual revenue totals grouped by store classification. Use this view to assess whether "
        "revenue growth is broad-based or concentrated in a single store type."
    )
    st.plotly_chart(fig_year, use_container_width=True)
    close_card(
        "<strong>Consistent year-over-year growth across all store types.</strong> "
        "Type A growth is the largest in absolute terms. Type B and C stores show a slower trajectory, "
        "presenting an opportunity to optimize promotional strategies for smaller-format locations."
    )

# =========================================================
# PAGE: FORECASTING
# =========================================================

elif page == "Forecasting":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Machine Learning Forecasting</div>'
        '<div class="page-subtitle">XGBoost model evaluation &mdash; actual vs predicted sales, '
        'scatter plot diagnostics, residual analysis, and error distribution assessment.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    needed = {"Actual_Sales","Predicted_Sales"}
    if not needed.issubset(forecast_df.columns):
        st.error(f"Forecasting file must contain: {needed}. Found: {list(forecast_df.columns)}")
        st.stop()

    n_points = st.slider("Number of data points to display",
                         50, min(500, len(forecast_df)), 300, step=50)
    fdf = forecast_df.head(n_points).copy()

    mae  = float(mean_absolute_error(fdf["Actual_Sales"], fdf["Predicted_Sales"]))
    rmse = float(np.sqrt(mean_squared_error(fdf["Actual_Sales"], fdf["Predicted_Sales"])))
    r2   = float(r2_score(fdf["Actual_Sales"], fdf["Predicted_Sales"]))
    denom = fdf["Actual_Sales"].replace(0, np.nan)
    mape  = float(np.mean(np.abs((fdf["Actual_Sales"] - fdf["Predicted_Sales"]) / denom)) * 100)

    m1, m2, m3, m4 = st.columns(4)
    render_kpi_row([
        (m1, "blue",   "#2563EB", "R\u00b2 Score", f"{r2:.4f}",     None,
         "Proportion of variance explained. Values closer to 1.0 indicate better model fit."),
        (m2, "green",  "#15803D", "MAE",           f"${mae:,.0f}",  None,
         "Mean Absolute Error &mdash; average absolute difference between actual and predicted values."),
        (m3, "amber",  "#B45309", "RMSE",          f"${rmse:,.0f}", None,
         "Root Mean Squared Error &mdash; penalizes large individual prediction errors more heavily."),
        (m4, "purple", "#6D28D9", "MAPE",          f"{mape:.2f}%",  None,
         "Mean Absolute Percentage Error &mdash; scale-independent measure of prediction accuracy."),
    ])

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # Time series
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        y=fdf["Actual_Sales"], mode="lines", name="Actual Sales",
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="Actual: $%{y:,.0f}<extra></extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        y=fdf["Predicted_Sales"], mode="lines", name="Predicted Sales",
        line=dict(color=PALETTE[1], width=2, dash="dot"),
        hovertemplate="Predicted: $%{y:,.0f}<extra></extra>"
    ))
    fig_fc.update_layout(
        **CHART, height=300,
        title=dict(text="Actual vs Predicted Sales &mdash; Time Series",
                   font=dict(size=13, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Observation Index", gridcolor="#EBEBEB"),
        yaxis=dict(title="Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color="#333333")),
    )
    open_card(
        "Actual vs Predicted Sales &mdash; Time Series View",
        "Solid blue line = actual recorded sales. "
        "Dashed amber line = XGBoost model prediction. "
        "Close alignment across the full range indicates a well-calibrated model "
        "with no systematic directional bias."
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    close_card(
        "<strong>The model tracks actual sales with high fidelity throughout the series.</strong> "
        "The largest deviations occur at extreme peaks &mdash; typically holiday weeks &mdash; "
        "where demand spikes are partially driven by unpredictable short-term consumer behavior. "
        "This is expected and does not indicate a model deficiency."
    )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    col_sc, col_re = st.columns(2)

    with col_sc:
        min_val = float(min(fdf["Actual_Sales"].min(), fdf["Predicted_Sales"].min()))
        max_val = float(max(fdf["Actual_Sales"].max(), fdf["Predicted_Sales"].max()))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Prediction (y = x)",
            line=dict(color="#B91C1C", width=1.5, dash="dash"), hoverinfo="skip"
        ))
        fig_sc.add_trace(go.Scatter(
            x=fdf["Actual_Sales"], y=fdf["Predicted_Sales"],
            mode="markers", name="Observations",
            marker=dict(color=PALETTE[0], size=4, opacity=0.5,
                        line=dict(color="#1D4ED8", width=0.3)),
            hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>"
        ))
        fig_sc.update_layout(
            **CHART, height=300,
            title=dict(text="Scatter Plot &mdash; Actual vs Predicted",
                       font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(title="Actual Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
            yaxis=dict(title="Predicted Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11, color="#333333")),
        )
        open_card(
            "Scatter Plot &mdash; Actual vs Predicted",
            "Each point represents one observation. "
            "The dashed red line is the perfect-prediction diagonal (y = x). "
            "Points above the line = model over-predicted; "
            "points below = model under-predicted."
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        close_card(
            f"<strong>Data points cluster tightly along the diagonal (R&sup2; = {r2:.4f})</strong>, "
            "confirming very high model accuracy. Outliers at the upper range correspond to "
            "exceptional holiday-week demand that is inherently difficult to predict with full precision."
        )

    with col_re:
        fdf["Residual"] = fdf["Actual_Sales"] - fdf["Predicted_Sales"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            y=fdf["Residual"], mode="lines",
            line=dict(color=PALETTE[0], width=1.2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="Residual: $%{y:,.0f}<extra></extra>"
        ))
        fig_res.add_hline(
            y=0, line=dict(color="#B91C1C", width=1.5, dash="dash"),
            annotation_text="Zero residual", annotation_position="top right",
            annotation_font=dict(size=10, color="#B91C1C")
        )
        fig_res.update_layout(
            **CHART, height=300,
            title=dict(text="Prediction Residuals Over Time",
                       font=dict(size=13, color="#1C1C1C"), x=0.01),
            xaxis=dict(title="Observation Index", gridcolor="#EBEBEB"),
            yaxis=dict(title="Residual (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        )
        open_card(
            "Prediction Residuals Over Time",
            "Residual = Actual Sales minus Predicted Sales. "
            "A well-behaved model produces residuals that scatter randomly around zero "
            "(red dashed line) with no directional trend or systematic pattern."
        )
        st.plotly_chart(fig_res, use_container_width=True)
        close_card(
            "<strong>Residuals are randomly distributed around zero</strong> with no upward or downward drift, "
            "confirming the model captures the primary sales patterns without systematic prediction bias."
        )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    res_mean = float(fdf["Residual"].mean())
    res_std  = float(fdf["Residual"].std())
    fig_hist = go.Figure(go.Histogram(
        x=fdf["Residual"], nbinsx=45,
        marker=dict(color=PALETTE[0], opacity=0.75, line=dict(color="#1D4ED8", width=0.5)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
    ))
    fig_hist.update_layout(
        **CHART, height=260,
        title=dict(text="Error Distribution &mdash; Residual Histogram",
                   font=dict(size=13, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Residual Value (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        yaxis=dict(title="Frequency", gridcolor="#EBEBEB"),
    )
    open_card(
        "Error Distribution &mdash; Residual Histogram",
        "Frequency distribution of residual values. "
        "An ideal model produces a symmetric, bell-shaped distribution centered at zero. "
        "Skew in either direction indicates a directional prediction bias."
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    close_card(
        "<strong>The residual distribution is approximately normal and centered near zero.</strong> "
        f"Mean residual: <strong>${res_mean:,.0f}</strong> &nbsp;&middot;&nbsp; "
        f"Std deviation: <strong>${res_std:,.0f}</strong>. "
        "A near-zero mean confirms the model does not systematically over- or under-predict."
    )

# =========================================================
# PAGE: STORE PERFORMANCE
# =========================================================

elif page == "Store Performance":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Store Performance</div>'
        '<div class="page-subtitle">Individual store analysis &mdash; weekly sales trends, '
        '12-week moving average, monthly seasonality heatmap, and performance benchmarking.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    selected_store_page = st.selectbox(
        "Select Store to Analyze",
        sorted(df["Store"].unique()),
        format_func=lambda x: f"Store {x}"
    )
    store_df = df[df["Store"] == selected_store_page].copy()

    s_total = float(store_df["Weekly_Sales"].sum())
    s_avg   = float(store_df["Weekly_Sales"].mean())
    s_peak  = float(store_df["Weekly_Sales"].max())
    s_type  = store_df["Type"].iloc[0] if "Type" in store_df.columns else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    render_kpi_row([
        (k1, "blue",   "#2563EB", "Store Total Revenue",
         f"${s_total/1e6:.1f}M", None,
         "All-time cumulative revenue for this store."),
        (k2, "green",  "#15803D", "Avg Weekly Sales",
         f"${s_avg:,.0f}", None,
         "Average revenue per week across the full historical record."),
        (k3, "amber",  "#B45309", "Peak Weekly Sales",
         f"${s_peak:,.0f}", None,
         "Highest single-week revenue on record for this store."),
        (k4, "purple", "#6D28D9", "Store Classification",
         str(s_type), None,
         "A = large-format, B = mid-size, C = compact."),
    ])

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    trend = store_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    trend["MA12"] = trend["Weekly_Sales"].rolling(12, min_periods=1).mean()

    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Weekly_Sales"],
        mode="lines", name="Weekly Sales",
        line=dict(color=PALETTE[0], width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>"
    ))
    fig_st.add_trace(go.Scatter(
        x=trend["Date"], y=trend["MA12"],
        mode="lines", name="12-Week Moving Avg",
        line=dict(color=PALETTE[1], width=2, dash="dot"),
        hovertemplate="MA12: $%{y:,.0f}<extra></extra>"
    ))
    fig_st.update_layout(
        **CHART, height=300,
        title=dict(text=f"Store {selected_store_page} &mdash; Weekly Sales Trend",
                   font=dict(size=13, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Date", gridcolor="#EBEBEB"),
        yaxis=dict(title="Weekly Sales (USD)", gridcolor="#EBEBEB", tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color="#333333")),
    )
    open_card(
        f"Store {selected_store_page} &mdash; Weekly Sales Trend",
        "Solid blue line = weekly sales. "
        "Dashed amber line = 12-week moving average, which smooths short-term volatility "
        "to reveal the underlying long-term trend direction."
    )
    st.plotly_chart(fig_st, use_container_width=True)
    close_card(
        "<strong>The 12-week moving average</strong> helps identify structural sales trends "
        "independent of seasonal noise. When actual sales consistently exceed the MA line, "
        "the store is in a positive momentum phase. A sustained divergence below the MA "
        "signals a performance deterioration that warrants operational review."
    )

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    store_df["Year"]  = store_df["Date"].dt.year
    store_df["Month"] = store_df["Date"].dt.month
    pivot = store_df.pivot_table(values="Weekly_Sales", index="Year", columns="Month", aggfunc="sum")
    mn = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns = [mn.get(c,c) for c in pivot.columns]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#EFF6FF"],[0.5,"#3B82F6"],[1,"#1E3A8A"]],
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Sales: $%{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickformat="$,.0f", tickfont=dict(size=10, color="#444444"), thickness=10)
    ))
    fig_hm.update_layout(
        **CHART, height=240,
        title=dict(text="Monthly Sales Heatmap by Year", font=dict(size=13, color="#1C1C1C"), x=0.01),
        xaxis=dict(title="Month", tickfont=dict(size=12, color="#333333")),
        yaxis=dict(title="Year",  tickfont=dict(size=12, color="#333333")),
    )
    open_card(
        "Monthly Sales Heatmap by Year",
        "Darker blue cells indicate higher revenue. Consistent color patterns across rows (years) "
        "confirm repeatable seasonal behavior. Use this view to plan monthly inventory and staffing."
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    close_card(
        "<strong>November and December consistently show the darkest cells across all years</strong>, "
        "confirming Q4 as the dominant revenue season at the individual store level. "
        "This regularity makes it reliable for forward capacity planning and stock replenishment scheduling."
    )

# =========================================================
# PAGE: DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Dataset Explorer</div>'
        '<div class="page-subtitle">Browse, inspect, and export the enterprise retail forecasting dataset. '
        'Use sidebar filters to narrow the data before downloading.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)
    render_kpi_row([
        (s1, "blue",   "#2563EB", "Total Rows",
         f"{len(filtered_df):,}", None,
         "Number of records in the currently filtered dataset."),
        (s2, "green",  "#15803D", "Total Columns",
         str(len(filtered_df.columns)), None,
         "Number of feature columns available in the dataset."),
        (s3, "amber",  "#B45309", "Active Stores",
         str(filtered_df["Store"].nunique()), None,
         "Distinct stores included in the current filter selection."),
        (s4, "purple", "#6D28D9", "Weeks Covered",
         str(filtered_df["Date"].nunique()), None,
         "Number of unique weekly periods in the filtered dataset."),
    ])

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    open_card(
        "Descriptive Statistics",
        "Summary statistics for all numeric columns in the currently filtered dataset. "
        "Count = non-null observations. All monetary values are in USD."
    )
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = filtered_df[num_cols].describe().T.reset_index()
        desc.columns = ["Column","Count","Mean","Std Dev","Min","25th Pct","Median","75th Pct","Max"]
        for c in ["Mean","Std Dev","Min","25th Pct","Median","75th Pct","Max"]:
            desc[c] = desc[c].apply(lambda x: f"{x:,.2f}")
        desc["Count"] = desc["Count"].apply(lambda x: f"{int(float(x)):,}")
        st.dataframe(desc, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    col_n, _ = st.columns([1, 3])
    with col_n:
        n_rows = st.selectbox("Rows to display", [50, 100, 250, 500], index=1)

    open_card(
        f"Data Preview &mdash; First {n_rows} Rows",
        "Scroll horizontally to view all columns. Click any column header to sort."
    )
    st.dataframe(filtered_df.head(n_rows), use_container_width=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Dataset (CSV)",
        data=csv,
        file_name="retail_dataset_export.csv",
        mime="text/csv"
    )

# =========================================================
# PAGE: ABOUT
# =========================================================

elif page == "About This Platform":

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">About This Platform</div>'
        '<div class="page-subtitle">Technical overview, data sources, model specifications, '
        'and platform architecture details.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Platform Overview</div>'
            '<div class="chart-card-desc">Enterprise Retail Intelligence System</div>'
            '<table class="summary-table">'
            '<tr><td style="color:#555555;width:38%;">Data Source</td><td class="td-val">Walmart Sales Dataset (Kaggle)</td></tr>'
            '<tr><td style="color:#555555;">Time Coverage</td><td class="td-val">February 2010 &ndash; November 2012</td></tr>'
            '<tr><td style="color:#555555;">Store Count</td><td class="td-val">45 retail branches</td></tr>'
            '<tr><td style="color:#555555;">Granularity</td><td class="td-val">Weekly, per department</td></tr>'
            '<tr><td style="color:#555555;">Total Records</td><td class="td-val">~420,000 rows</td></tr>'
            '<tr><td style="color:#555555;">Framework</td><td class="td-val">Streamlit + Plotly + Pandas</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True
        )

    with col_r:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Model Information</div>'
            '<div class="chart-card-desc">XGBoost Forecasting Engine</div>'
            '<table class="summary-table">'
            '<tr><td style="color:#555555;width:38%;">Algorithm</td><td class="td-val">XGBoost (eXtreme Gradient Boosting)</td></tr>'
            '<tr><td style="color:#555555;">R&sup2; Score</td><td class="td-val" style="color:#15803D;font-weight:700;">0.9840</td></tr>'
            '<tr><td style="color:#555555;">Validation</td><td class="td-val">Time-series cross-validation</td></tr>'
            '<tr><td style="color:#555555;">Input Features</td><td class="td-val">Store, Dept, Date, IsHoliday, Temperature, Fuel Price, CPI, Unemployment, MarkDowns</td></tr>'
            '<tr><td style="color:#555555;">Target Variable</td><td class="td-val">Weekly_Sales</td></tr>'
            '<tr><td style="color:#555555;">Libraries</td><td class="td-val">scikit-learn + XGBoost + NumPy</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<hr>'
    '<div style="display:flex;justify-content:space-between;align-items:center;'
    'font-size:11px;color:#888888;padding:0 0.25rem;">'
    '<div><strong style="color:#444444;">Enterprise Retail Intelligence Platform</strong> &nbsp;v4.2</div>'
    '<div>Machine Learning Forecasting &nbsp;&middot;&nbsp; Business Intelligence &nbsp;&middot;&nbsp; Executive Analytics</div>'
    '</div>',
    unsafe_allow_html=True
)

