# ============================================================
# WALMART RETAIL INTELLIGENCE PLATFORM
# SALES ANALYTICS & XGBOOST FORECASTING
# VERSION 6.0 — FIXED + REDESIGNED
# ============================================================

import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Walmart Retail Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_PATH = Path("data/processed")

DATA_FILE = BASE_PATH / "walmart_featured.csv"
PREDICTION_FILE = BASE_PATH / "test_predictions.csv"
EVALUATION_FILE = BASE_PATH / "model_evaluation.csv"
FEATURE_IMPORTANCE_FILE = BASE_PATH / "xgboost_feature_importance.csv"
FUTURE_FORECAST_FILE = BASE_PATH / "future_sales_forecast.csv"
FUTURE_TOTAL_FILE = BASE_PATH / "future_total_sales_forecast.csv"


# ============================================================
# CORE FIX: SAFE HTML RENDERER
# ------------------------------------------------------------
# Root cause of the "raw tags showing as text" bug: Streamlit's
# st.markdown() runs content through a Markdown parser BEFORE
# allowing HTML. Markdown treats any line indented 4+ spaces as
# a preformatted code block, so indented HTML strings (which is
# how Python naturally formats nested f-strings) get printed as
# literal text instead of being rendered.
#
# Fix: always dedent + strip before calling st.markdown(). Every
# HTML block in this file goes through render_html() / this
# same logic inside the render_* helpers below. Never call
# st.markdown() with a raw indented HTML string directly again.
# ============================================================

def render_html(content, target=None):
    """Safely render an HTML string via st.markdown, avoiding the
    Markdown-indentation-becomes-code-block bug."""
    cleaned = textwrap.dedent(content).strip()
    target = target or st
    target.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# GLOBAL STYLE
# ============================================================

render_html(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background: #F5F6F8;
        color: #172033;
    }

    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }

    /* ========================= SIDEBAR ========================= */

    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.25rem 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    .brand {
        padding: 0.25rem 0.25rem 1.1rem 0.25rem;
        border-bottom: 1px solid #263247;
        margin-bottom: 1.1rem;
    }

    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 13px;
        color: white;
        flex-shrink: 0;
    }

    .brand-title {
        font-size: 14px;
        font-weight: 800;
        color: white;
        line-height: 1.25;
    }

    .brand-subtitle {
        font-size: 10px;
        color: #94A3B8;
        margin-top: 3px;
    }

    .sidebar-section {
        font-size: 9px;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        margin-top: 1.1rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-summary {
        background: #172033;
        border: 1px solid #263247;
        border-radius: 10px;
        padding: 11px;
        margin-top: 0.6rem;
    }

    .sidebar-summary-label {
        font-size: 9px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .sidebar-summary-value {
        font-size: 15px;
        font-weight: 700;
        color: white;
        margin-top: 3px;
    }

    .sidebar-footer {
        margin-top: 1.6rem;
        padding-top: 1rem;
        border-top: 1px solid #263247;
        text-align: center;
        color: #64748B;
        font-size: 9px;
        line-height: 1.7;
    }

    /* ========================= RADIO NAV ========================= */

    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] > div {
        gap: 3px !important;
    }

    div[data-testid="stRadio"] > div > label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 10px !important;
        color: #CBD5E1 !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        transition: background 0.15s ease;
    }

    div[data-testid="stRadio"] > div > label:hover {
        background: #172033 !important;
        color: white !important;
    }

    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: #2563EB !important;
        border: 1px solid #2563EB !important;
        color: white !important;
        font-weight: 700 !important;
    }

    /* ========================= TOP FILTER BAR ========================= */

    .filterbar {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 16px 4px 16px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    .filterbar-label {
        font-size: 9px;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* ========================= TOPBAR / HEADER ========================= */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .topbar-title {
        font-size: 11px;
        color: #64748B;
        font-weight: 600;
    }

    .topbar-status {
        font-size: 10px;
        color: #15803D;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: 700;
        white-space: nowrap;
    }

    .page-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .page-subtitle {
        font-size: 12px;
        color: #64748B;
        line-height: 1.6;
        margin-bottom: 1.1rem;
        max-width: 900px;
    }

    /* ========================= KPI ========================= */

    .kpi-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 120px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    .kpi-card:after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 3px;
        background: #2563EB;
    }

    .kpi-green:after { background: #16A34A; }
    .kpi-orange:after { background: #EA580C; }
    .kpi-purple:after { background: #7C3AED; }
    .kpi-red:after { background: #DC2626; }

    .kpi-label {
        font-size: 9px;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.9px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 8px;
        letter-spacing: -0.5px;
    }

    .kpi-description {
        font-size: 10.5px;
        color: #64748B;
        line-height: 1.45;
        margin-top: 6px;
    }

    /* ========================= CARDS ========================= */

    .section-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    .section-title {
        font-size: 14px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 3px;
    }

    .section-description {
        font-size: 10.5px;
        color: #64748B;
        line-height: 1.55;
        margin-bottom: 10px;
    }

    .insight-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 9px;
    }

    .insight-title {
        font-size: 11px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 4px;
    }

    .insight-text {
        font-size: 11px;
        color: #475569;
        line-height: 1.6;
    }

    .status-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 13px;
    }

    .status-label {
        font-size: 9px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.8px;
    }

    .status-value {
        font-size: 16px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 3px;
    }

    .status-note {
        font-size: 9.5px;
        color: #64748B;
        margin-top: 3px;
    }

    /* ========================= STREAMLIT WIDGETS ========================= */

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 9px !important;
        overflow: hidden !important;
    }

    div[data-testid="stDownloadButton"] button {
        background: #0F172A !important;
        color: white !important;
        border: none !important;
        border-radius: 7px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background: #1E293B !important;
    }

    .stSelectbox label,
    .stDateInput label,
    .stSlider label {
        font-size: 10px !important;
        font-weight: 700 !important;
        color: #64748B !important;
    }

    hr { border-color: #E2E8F0 !important; }

    </style>
    """
)


# ============================================================
# CHART THEME
# ============================================================

CHART_CONFIG = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#FFFFFF",
    "font": {"family": "Inter, sans-serif", "size": 10, "color": "#475569"},
    "margin": {"l": 10, "r": 10, "t": 45, "b": 10},
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value, decimals=0):
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.{decimals}f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.{decimals}f}K"
    return f"${value:,.0f}"


def safe_numeric(df, columns):
    existing = [c for c in columns if c in df.columns]
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_boolean_column(series):
    """IsHoliday (and similar flags) can arrive as real booleans,
    strings ('True'/'False'), or 0/1. Coerce all of them into a
    proper boolean Series so comparisons don't silently fail."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
    )


def render_page_header(title, subtitle):
    render_html(
        f"""
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """
    )


def render_section_header(title, description=""):
    render_html(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-description">{description}</div>
        </div>
        """
    )


def render_kpi(column, css_class, label, value, description):
    render_html(
        f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-description">{description}</div>
        </div>
        """,
        target=column,
    )


def render_insight(title, text):
    render_html(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """
    )


def render_status(column, label, value, note=""):
    render_html(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-note">{note}</div>
        </div>
        """,
        target=column,
    )


def load_csv(path, required=False):
    if not path.exists():
        if required:
            st.error(
                f"Required data file not found: `{path}`. "
                "Make sure the file exists relative to where you run "
                "`streamlit run`, and that it was committed to the repo "
                "(check .gitignore isn't excluding the data folder)."
            )
            st.stop()
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read {path}: {exc}")
        st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all_data():

    df = load_csv(DATA_FILE, required=True)

    predictions = load_csv(PREDICTION_FILE)
    evaluation = load_csv(EVALUATION_FILE)
    feature_importance = load_csv(FEATURE_IMPORTANCE_FILE)
    future_forecast = load_csv(FUTURE_FORECAST_FILE)
    future_total = load_csv(FUTURE_TOTAL_FILE)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Date" in predictions.columns:
        predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = normalize_boolean_column(df["IsHoliday"])

    for data in [predictions, evaluation, feature_importance, future_forecast, future_total]:
        if not data.empty:
            safe_numeric(data, data.select_dtypes(include=["number"]).columns.tolist())

    safe_numeric(
        df,
        ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size", "Store", "Dept"],
    )

    return df, predictions, evaluation, feature_importance, future_forecast, future_total


(df, predictions, evaluation, feature_importance, future_forecast, future_total) = load_all_data()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = ["Date", "Store", "Weekly_Sales"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("The Walmart dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

df = df.dropna(subset=["Date", "Weekly_Sales"]).copy()
df = df.sort_values("Date").reset_index(drop=True)

if df.empty:
    st.error("The dataset has no valid rows after removing missing Date / Weekly_Sales values.")
    st.stop()


# ============================================================
# SIDEBAR — NAVIGATION ONLY (filters moved to a top bar so the
# layout mirrors the reference dashboard: nav on the left,
# filters as a horizontal bar under the page title)
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="brand">
            <div class="brand-box">
                <div class="brand-mark">RI</div>
                <div>
                    <div class="brand-title">Retail Intelligence</div>
                    <div class="brand-subtitle">Walmart Sales Analytics Platform</div>
                </div>
            </div>
        </div>
        """
    )

    render_html('<div class="sidebar-section">Navigation</div>')

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Sales Analytics",
            "Forecasting",
            "Store Performance",
            "Dataset Explorer",
            "About Platform",
        ],
        label_visibility="collapsed",
    )

    render_html('<div class="sidebar-section">Active View</div>')
    sidebar_summary_placeholder = st.container()

    render_html(
        """
        <div class="sidebar-footer">
            Walmart Retail Intelligence<br>
            Machine Learning &middot; Business Analytics<br>
            Forecasting &middot; Decision Support
        </div>
        """
    )


# ============================================================
# TOP FILTER BAR
# ============================================================

render_html('<div class="filterbar">')

f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])

with f1:
    render_html('<div class="filterbar-label">Store</div>')
    store_options = ["All Stores"] + (
        sorted(df["Store"].dropna().unique().tolist()) if "Store" in df.columns else []
    )
    selected_store = st.selectbox("Store", store_options, label_visibility="collapsed")

with f2:
    render_html('<div class="filterbar-label">Store Type</div>')
    type_options = ["All Types"] + (
        sorted(df["Type"].dropna().unique().tolist()) if "Type" in df.columns else []
    )
    selected_type = st.selectbox("Store Type", type_options, label_visibility="collapsed")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

with f3:
    render_html('<div class="filterbar-label">From</div>')
    date_from = st.date_input(
        "From", value=min_date, min_value=min_date, max_value=max_date,
        label_visibility="collapsed",
    )

with f4:
    render_html('<div class="filterbar-label">To</div>')
    date_to = st.date_input(
        "To", value=max_date, min_value=min_date, max_value=max_date,
        label_visibility="collapsed",
    )

render_html("</div>")

if date_from > date_to:
    st.error("The start date cannot be later than the end date.")
    st.stop()


# --------------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------------

filtered_df = df[
    (df["Date"] >= pd.to_datetime(date_from)) & (df["Date"] <= pd.to_datetime(date_to))
].copy()

if selected_store != "All Stores":
    filtered_df = filtered_df[filtered_df["Store"] == selected_store]

if selected_type != "All Types" and "Type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Type"] == selected_type]


# --------------------------------------------------------
# SIDEBAR SUMMARY (rendered after filters are known)
# --------------------------------------------------------

with sidebar_summary_placeholder:
    if not filtered_df.empty:
        sidebar_sales = filtered_df["Weekly_Sales"].sum()
        sidebar_stores = filtered_df["Store"].nunique()
        sidebar_weeks = filtered_df["Date"].nunique()

        render_html(
            f"""
            <div class="sidebar-summary">
                <div class="sidebar-summary-label">Stores</div>
                <div class="sidebar-summary-value">{sidebar_stores:,}</div>
            </div>
            <div class="sidebar-summary">
                <div class="sidebar-summary-label">Weekly Periods</div>
                <div class="sidebar-summary-value">{sidebar_weeks:,}</div>
            </div>
            <div class="sidebar-summary">
                <div class="sidebar-summary-label">Revenue</div>
                <div class="sidebar-summary-value">{money(sidebar_sales, 2)}</div>
            </div>
            """
        )
    else:
        render_html(
            """
            <div class="sidebar-summary">
                <div class="sidebar-summary-label">Status</div>
                <div class="sidebar-summary-value">No data</div>
            </div>
            """
        )


# ============================================================
# DATA GUARD
# ============================================================

if filtered_df.empty:
    st.warning("No records are available for the selected filters. Please adjust the store, type, or date range.")
    st.stop()


# ============================================================
# TOP BAR
# ============================================================

render_html(
    """
    <div class="topbar">
        <div class="topbar-title">Walmart Retail Intelligence Platform</div>
        <div class="topbar-status">Pipeline Completed</div>
    </div>
    """
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    render_page_header(
        "Executive Retail Overview",
        "Network-wide view of revenue, seasonality, store contribution, and forecast readiness.",
    )

    network_weekly = filtered_df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    total_revenue = network_weekly["Weekly_Sales"].sum()
    avg_weekly = network_weekly["Weekly_Sales"].mean()
    peak_weekly = network_weekly["Weekly_Sales"].max()
    peak_date = network_weekly.loc[network_weekly["Weekly_Sales"].idxmax(), "Date"]
    active_stores = filtered_df["Store"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    render_kpi(c1, "", "Total Revenue", money(total_revenue, 2), "Cumulative revenue for the selected filters.")
    render_kpi(c2, "kpi-green", "Avg Weekly Sales", money(avg_weekly, 2), "Average sales per week, network-wide.")
    render_kpi(c3, "kpi-orange", "Peak Weekly Sales", money(peak_weekly, 2), f"Peak week: {peak_date.strftime('%d %b %Y')}.")
    render_kpi(c4, "kpi-purple", "Active Stores", f"{active_stores:,}", "Stores included in this view.")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2.2, 1])

    with left:
        monthly = filtered_df.groupby(filtered_df["Date"].dt.to_period("M"))["Weekly_Sales"].sum().reset_index()
        monthly["Date"] = monthly["Date"].dt.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Weekly_Sales"], mode="lines",
            line=dict(color="#2563EB", width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            **CHART_CONFIG, height=340,
            title=dict(text="Monthly Revenue Trend", font=dict(size=14, color="#172033"), x=0.01),
            xaxis=dict(title="Month", gridcolor="#EEF2F7"),
            yaxis=dict(title="Revenue (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
            showlegend=False,
        )

        render_html(
            """
            <div class="section-card">
                <div class="section-title">Revenue Trend</div>
                <div class="section-description">How total network revenue moved month by month.</div>
            """
        )
        st.plotly_chart(fig, use_container_width=True)
        render_html("</div>")

    with right:
        if "IsHoliday" in filtered_df.columns:
            holiday = filtered_df.loc[filtered_df["IsHoliday"] == True, "Weekly_Sales"].mean()
            non_holiday = filtered_df.loc[filtered_df["IsHoliday"] == False, "Weekly_Sales"].mean()
            if pd.notna(holiday) and pd.notna(non_holiday) and non_holiday != 0:
                holiday_premium = (holiday - non_holiday) / non_holiday * 100
            else:
                holiday_premium = np.nan
        else:
            holiday_premium = np.nan

        store_revenue = filtered_df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
        top_store = store_revenue.index[0]
        top_store_value = store_revenue.iloc[0]
        top_store_share = top_store_value / store_revenue.sum() * 100

        render_insight(
            "Leading Store",
            f"Store {top_store} leads with {money(top_store_value, 2)} "
            f"({top_store_share:.1f}% of filtered revenue).",
        )

        if pd.notna(holiday_premium):
            render_insight(
                "Holiday Effect",
                f"Holiday weeks run {holiday_premium:+.1f}% vs. non-holiday weeks.",
            )
        else:
            render_insight("Holiday Effect", "Holiday flag not available for this data.")

        model_r2 = 0.9853
        model_wape = 8.07

        if not evaluation.empty:
            for col in evaluation.columns:
                lower = col.lower()
                if "r2" in lower:
                    try:
                        model_r2 = float(pd.to_numeric(evaluation[col], errors="coerce").dropna().iloc[0])
                    except Exception:
                        pass
                if "wape" in lower:
                    try:
                        model_wape = float(pd.to_numeric(evaluation[col], errors="coerce").dropna().iloc[0])
                    except Exception:
                        pass

        render_insight(
            "Forecast Readiness",
            f"XGBoost model: R² {model_r2:.4f}, WAPE {model_wape:.2f}% on held-out test data.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if "Type" in filtered_df.columns:
            type_revenue = filtered_df.groupby("Type")["Weekly_Sales"].sum().reset_index()
            fig_type = px.pie(type_revenue, names="Type", values="Weekly_Sales", hole=0.58)
            fig_type.update_traces(textposition="outside", texttemplate="%{label}<br>%{percent:.1%}")
            fig_type.update_layout(
                **CHART_CONFIG, height=300,
                title=dict(text="Revenue Mix by Store Type", font=dict(size=14, color="#172033"), x=0.01),
                showlegend=False,
            )
            render_html(
                """
                <div class="section-card">
                    <div class="section-title">Revenue Mix</div>
                    <div class="section-description">Share of revenue contributed by each store type.</div>
                """
            )
            st.plotly_chart(fig_type, use_container_width=True)
            render_html("</div>")
        else:
            render_html(
                """
                <div class="section-card">
                    <div class="section-title">Revenue Mix</div>
                    <div class="section-description">Store type column not available in this dataset.</div>
                </div>
                """
            )

    with col_b:
        dept_revenue = (
            filtered_df.groupby("Dept")["Weekly_Sales"].sum()
            .sort_values(ascending=False).head(10).sort_values().reset_index()
        )
        fig_dept = px.bar(dept_revenue, x="Weekly_Sales", y="Dept", orientation="h")
        fig_dept.update_traces(marker_color="#2563EB")
        fig_dept.update_layout(
            **CHART_CONFIG, height=300,
            title=dict(text="Top 10 Departments by Revenue", font=dict(size=14, color="#172033"), x=0.01),
            xaxis=dict(title="Revenue (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
            yaxis=dict(title="Department", gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        render_html(
            """
            <div class="section-card">
                <div class="section-title">Department Contribution</div>
                <div class="section-description">Highest-revenue departments in this view.</div>
            """
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        render_html("</div>")


# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    render_page_header(
        "Sales Analytics",
        "Store, department, seasonality, holiday, and external-variable breakdown.",
    )

    col1, col2 = st.columns(2)

    with col1:
        store_sales = (
            filtered_df.groupby("Store")["Weekly_Sales"].sum()
            .sort_values(ascending=False).head(10).sort_values().reset_index()
        )
        fig_store = px.bar(store_sales, x="Weekly_Sales", y="Store", orientation="h")
        fig_store.update_traces(marker_color="#2563EB")
        fig_store.update_layout(
            **CHART_CONFIG, height=350,
            title=dict(text="Top 10 Stores by Revenue", font=dict(size=14, color="#172033"), x=0.01),
            xaxis=dict(title="Revenue (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
            yaxis=dict(title="Store", gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig_store, use_container_width=True)

    with col2:
        if "Type" in filtered_df.columns:
            fig_box = px.box(filtered_df, x="Type", y="Weekly_Sales", color="Type")
            fig_box.update_layout(
                **CHART_CONFIG, height=350,
                title=dict(text="Weekly Sales Distribution by Store Type", font=dict(size=14, color="#172033"), x=0.01),
                xaxis=dict(title="Store Type"),
                yaxis=dict(title="Weekly Sales (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Store Type column not available in this dataset.")

    if "IsHoliday" in filtered_df.columns:
        st.markdown("<br>", unsafe_allow_html=True)

        holiday_group = filtered_df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
        holiday_group["Period"] = holiday_group["IsHoliday"].map({True: "Holiday Week", False: "Non-Holiday Week"})
        holiday_group = holiday_group.dropna(subset=["Period"])

        fig_holiday = px.bar(holiday_group, x="Period", y="Weekly_Sales", text="Weekly_Sales")
        fig_holiday.update_traces(
            texttemplate="$%{y:,.0f}", textposition="outside",
            marker_color=["#EA580C", "#2563EB"][: len(holiday_group)],
        )
        fig_holiday.update_layout(
            **CHART_CONFIG, height=300,
            title=dict(text="Average Weekly Sales: Holiday vs Non-Holiday", font=dict(size=14, color="#172033"), x=0.01),
            xaxis=dict(title="Period"),
            yaxis=dict(title="Average Weekly Sales (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
            showlegend=False,
        )

        render_html(
            """
            <div class="section-card">
                <div class="section-title">Holiday Sales Impact</div>
                <div class="section-description">How holiday weeks compare to regular weeks, historically.</div>
            """
        )
        st.plotly_chart(fig_holiday, use_container_width=True)
        render_html("</div>")

    corr_cols = [c for c in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Weekly_Sales"] if c in filtered_df.columns]

    if len(corr_cols) >= 2:
        corr = filtered_df[corr_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#B91C1C", "#F8FAFC", "#2563EB"],
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(
            **CHART_CONFIG, height=360,
            title=dict(text="Correlation Matrix", font=dict(size=14, color="#172033"), x=0.01),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    render_page_header(
        "Machine Learning Forecasting",
        "XGBoost diagnostics, held-out test performance, feature importance, and 12-week forward forecast.",
    )

    xgb_mae, xgb_rmse, xgb_r2, xgb_wape = 1280.50, 2667.55, 0.9853, 8.07
    naive_mae, naive_rmse = 1546.01, 3431.09
    mae_improvement, rmse_improvement = 17.17, 22.25

    if not evaluation.empty:
        for col in evaluation.columns:
            lower = col.lower()
            try:
                value = float(pd.to_numeric(evaluation[col], errors="coerce").dropna().iloc[0])
            except Exception:
                continue
            if "mae" in lower and "naive" not in lower:
                xgb_mae = value
            elif "rmse" in lower and "naive" not in lower:
                xgb_rmse = value
            elif "r2" in lower or "r\u00b2" in lower:
                xgb_r2 = value
            elif "wape" in lower:
                xgb_wape = value

    m1, m2, m3, m4 = st.columns(4)
    render_kpi(m1, "", "R\u00b2 Score", f"{xgb_r2:.4f}", "Variance explained on the held-out test period.")
    render_kpi(m2, "kpi-green", "WAPE", f"{xgb_wape:.2f}%", "Weighted absolute percentage error.")
    render_kpi(m3, "kpi-orange", "MAE", money(xgb_mae), "Average absolute prediction error.")
    render_kpi(m4, "kpi-purple", "RMSE", money(xgb_rmse), "Error metric that penalizes large misses more.")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        baseline_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE"],
            "Naive Baseline": [naive_mae, naive_rmse],
            "XGBoost": [xgb_mae, xgb_rmse],
        })
        fig_model = go.Figure()
        fig_model.add_trace(go.Bar(x=baseline_df["Metric"], y=baseline_df["Naive Baseline"], name="Naive Baseline", marker_color="#CBD5E1"))
        fig_model.add_trace(go.Bar(x=baseline_df["Metric"], y=baseline_df["XGBoost"], name="XGBoost", marker_color="#2563EB"))
        fig_model.update_layout(
            **CHART_CONFIG, height=330, barmode="group",
            title=dict(text="Model vs Naive Baseline", font=dict(size=14, color="#172033"), x=0.01),
            yaxis=dict(title="Error", tickformat=",.0f", gridcolor="#EEF2F7"),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig_model, use_container_width=True)

    with col_b:
        render_insight("MAE Improvement", f"XGBoost cuts MAE by {mae_improvement:.2f}% vs. the naive baseline.")
        render_insight("RMSE Improvement", f"XGBoost cuts RMSE by {rmse_improvement:.2f}% vs. the naive baseline.")
        render_insight("What this means", "Error is meaningfully lower than a simple last-value guess, so the model adds real value for planning.")
        render_insight("Caveat", "A high R\u00b2 is not certainty. Promotions, weather, and holiday spikes can still push actuals off-forecast.")

    if not predictions.empty and all(c in predictions.columns for c in ["Actual_Sales", "Predicted_Sales"]):

        n_available = len(predictions)

        if n_available < 20:
            st.info(f"Only {n_available} test observations available \u2014 not enough to chart meaningfully.")
        else:
            slider_min = min(20, n_available)
            slider_max = min(1000, n_available)
            slider_default = min(400, n_available)

            if slider_max <= slider_min:
                display_points = slider_max
            else:
                step = max(1, (slider_max - slider_min) // 10)
                display_points = st.slider(
                    "Test observations to display",
                    min_value=slider_min, max_value=slider_max,
                    value=slider_default, step=step,
                )

            plot_df = predictions.head(display_points).copy()
            plot_df["Observation"] = np.arange(len(plot_df))

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=plot_df["Observation"], y=plot_df["Actual_Sales"], mode="lines",
                name="Actual", line=dict(color="#2563EB", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=plot_df["Observation"], y=plot_df["Predicted_Sales"], mode="lines",
                name="Predicted", line=dict(color="#EA580C", width=2, dash="dot"),
            ))
            fig_pred.update_layout(
                **CHART_CONFIG, height=350,
                title=dict(text="Actual vs Predicted Sales", font=dict(size=14, color="#172033"), x=0.01),
                xaxis=dict(title="Test Observation", gridcolor="#EEF2F7"),
                yaxis=dict(title="Weekly Sales (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
                legend=dict(orientation="h", y=1.08, x=0),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            plot_df["Residual"] = plot_df["Actual_Sales"] - plot_df["Predicted_Sales"]

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                fig_res = px.histogram(plot_df, x="Residual", nbins=50)
                fig_res.update_traces(marker_color="#2563EB")
                fig_res.update_layout(
                    **CHART_CONFIG, height=300,
                    title=dict(text="Residual Distribution", font=dict(size=14, color="#172033"), x=0.01),
                    xaxis=dict(title="Residual (USD)", tickformat="$,.0f"),
                    yaxis=dict(title="Frequency"),
                )
                st.plotly_chart(fig_res, use_container_width=True)

            with col_r2:
                fig_scatter = px.scatter(plot_df, x="Actual_Sales", y="Predicted_Sales", opacity=0.45)
                min_value = min(plot_df["Actual_Sales"].min(), plot_df["Predicted_Sales"].min())
                max_value = max(plot_df["Actual_Sales"].max(), plot_df["Predicted_Sales"].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_value, max_value], y=[min_value, max_value], mode="lines",
                    name="Perfect Prediction", line=dict(color="#DC2626", dash="dash"),
                ))
                fig_scatter.update_layout(
                    **CHART_CONFIG, height=300,
                    title=dict(text="Actual vs Predicted Diagnostic", font=dict(size=14, color="#172033"), x=0.01),
                    xaxis=dict(title="Actual Sales", tickformat="$,.0f"),
                    yaxis=dict(title="Predicted Sales", tickformat="$,.0f"),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    elif not predictions.empty:
        st.info("Prediction file is missing 'Actual_Sales' / 'Predicted_Sales' columns \u2014 skipping actual-vs-predicted charts.")

    if not feature_importance.empty:
        feature_col = next((c for c in feature_importance.columns if "feature" in c.lower()), None)
        importance_col = next((c for c in feature_importance.columns if "importance" in c.lower()), None)

        if feature_col and importance_col:
            fi = feature_importance.copy()
            fi[importance_col] = pd.to_numeric(fi[importance_col], errors="coerce")
            fi = fi.dropna(subset=[importance_col]).sort_values(importance_col, ascending=False).head(10).sort_values(importance_col)

            fig_fi = px.bar(fi, x=importance_col, y=feature_col, orientation="h")
            fig_fi.update_traces(marker_color="#2563EB")
            fig_fi.update_layout(
                **CHART_CONFIG, height=360,
                title=dict(text="Top Predictive Features", font=dict(size=14, color="#172033"), x=0.01),
                xaxis=dict(title="Importance", gridcolor="#EEF2F7"),
                yaxis=dict(title="Feature", gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("12-Week Forward Forecast", "Forward projection from the latest XGBoost pipeline.")

    if not future_forecast.empty:
        forecast = future_forecast.copy()

        date_col = next((c for c in forecast.columns if "date" in c.lower() or "week" in c.lower()), None)
        if date_col is not None:
            forecast[date_col] = pd.to_datetime(forecast[date_col], errors="coerce")

        value_col = next(
            (c for c in ["Predicted_Sales", "Forecast_Sales", "Predicted_Total_Sales", "Forecast", "Sales_Forecast"] if c in forecast.columns),
            None,
        )
        if value_col is None:
            numeric_cols = forecast.select_dtypes(include="number").columns.tolist()
            value_col = numeric_cols[-1] if numeric_cols else None

        if date_col is not None and value_col is not None:
            forecast[value_col] = pd.to_numeric(forecast[value_col], errors="coerce")
            forecast = forecast.dropna(subset=[date_col, value_col])

        if date_col is not None and value_col is not None and not forecast.empty:
            total_forecast = forecast[value_col].sum()
            avg_forecast = forecast[value_col].mean()
            peak_idx = forecast[value_col].idxmax()
            lowest_idx = forecast[value_col].idxmin()
            peak_forecast_date = forecast.loc[peak_idx, date_col]
            lowest_forecast_date = forecast.loc[lowest_idx, date_col]

            f1, f2, f3, f4 = st.columns(4)
            render_kpi(f1, "", "12-Week Forecast", money(total_forecast, 2), "Total projected sales across the horizon.")
            render_kpi(f2, "kpi-green", "Avg Weekly Forecast", money(avg_forecast, 2), "Average projected weekly sales.")
            render_kpi(f3, "kpi-orange", "Peak Forecast Week", money(forecast[value_col].max(), 2), peak_forecast_date.strftime("%d %b %Y"))
            render_kpi(f4, "kpi-purple", "Lowest Forecast Week", money(forecast[value_col].min(), 2), lowest_forecast_date.strftime("%d %b %Y"))

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=forecast[date_col], y=forecast[value_col], mode="lines+markers", name="Forecast",
                line=dict(color="#2563EB", width=2.5), marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
            ))
            fig_future.update_layout(
                **CHART_CONFIG, height=360,
                title=dict(text="12-Week Future Sales Forecast", font=dict(size=14, color="#172033"), x=0.01),
                xaxis=dict(title="Forecast Week", gridcolor="#EEF2F7"),
                yaxis=dict(title="Forecast Sales (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
                showlegend=False,
            )
            st.plotly_chart(fig_future, use_container_width=True)

            render_html(
                """
                <div class="section-card">
                    <div class="section-title">Forecast Assumptions</div>
                    <div class="section-description">
                        This forecast extends past the historical data, so it leans on assumptions about
                        future calendar and business conditions.
                    </div>
                """
            )

            assumption_col1, assumption_col2 = st.columns(2)
            with assumption_col1:
                render_status(
                    st.container(), "Historical Data Cutoff",
                    df["Date"].max().strftime("%d %b %Y"),
                    "Last observed date in the dataset.",
                )
            with assumption_col2:
                future_start = forecast[date_col].min()
                future_end = forecast[date_col].max()
                render_status(
                    st.container(), "Forecast Horizon",
                    f"{future_start.strftime('%d %b %Y')} \u2014 {future_end.strftime('%d %b %Y')}",
                    f"{len(forecast)} future weekly periods.",
                )

            render_html(
                """
                <br>
                <div class="insight-card">
                    <div class="insight-title">Holiday Assumption</div>
                    <div class="insight-text">
                        Future holiday weeks are a scenario, not an observed fact \u2014 promotions and
                        weather aren't known in advance the way past holidays were.
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-title">How to use this</div>
                    <div class="insight-text">
                        Treat this as a planning input for inventory, staffing, and capacity \u2014
                        not a guaranteed sales number.
                    </div>
                </div>
                """
            )
            render_html("</div>")

            display_forecast = forecast[[date_col, value_col]].copy()
            display_forecast.columns = ["Forecast Week", "Predicted Sales"]
            display_forecast["Forecast Week"] = display_forecast["Forecast Week"].dt.strftime("%d %b %Y")
            display_forecast["Predicted Sales"] = display_forecast["Predicted Sales"].apply(lambda x: f"${x:,.2f}")

            st.dataframe(display_forecast, use_container_width=True, hide_index=True)

            download_forecast = forecast.to_csv(index=False).encode("utf-8")
            st.download_button("Download Forecast CSV", download_forecast, file_name="walmart_12_week_forecast.csv", mime="text/csv")
        else:
            st.warning("Could not identify a date column and a forecast value column in the future forecast file.")
    else:
        st.warning("Future forecast file was not found. Run the forecasting pipeline first.")


# ============================================================
# STORE PERFORMANCE
# ============================================================

elif page == "Store Performance":

    render_page_header(
        "Store Performance",
        "Store-level benchmarking, revenue trend, and seasonality diagnostics.",
    )

    selected_store_analysis = st.selectbox(
        "Select Store",
        sorted(df["Store"].dropna().unique().tolist()),
        format_func=lambda x: f"Store {x}",
    )

    store_df = df[df["Store"] == selected_store_analysis].copy()
    store_weekly = store_df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    store_total = store_weekly["Weekly_Sales"].sum()
    store_avg = store_weekly["Weekly_Sales"].mean()
    store_peak = store_weekly["Weekly_Sales"].max()
    peak_date = store_weekly.loc[store_weekly["Weekly_Sales"].idxmax(), "Date"]
    store_type = store_df["Type"].iloc[0] if "Type" in store_df.columns and not store_df.empty else "N/A"

    s1, s2, s3, s4 = st.columns(4)
    render_kpi(s1, "", "Total Revenue", money(store_total, 2), "Cumulative sales for this store.")
    render_kpi(s2, "kpi-green", "Avg Weekly Sales", money(store_avg, 2), "Average weekly sales.")
    render_kpi(s3, "kpi-orange", "Peak Weekly Sales", money(store_peak, 2), f"Peak: {peak_date.strftime('%d %b %Y')}.")
    render_kpi(s4, "kpi-purple", "Store Type", str(store_type), "Store classification.")

    st.markdown("<br>", unsafe_allow_html=True)

    store_weekly["MA12"] = store_weekly["Weekly_Sales"].rolling(12, min_periods=1).mean()

    fig_store = go.Figure()
    fig_store.add_trace(go.Scatter(x=store_weekly["Date"], y=store_weekly["Weekly_Sales"], mode="lines", name="Weekly Sales", line=dict(color="#2563EB", width=1.7)))
    fig_store.add_trace(go.Scatter(x=store_weekly["Date"], y=store_weekly["MA12"], mode="lines", name="12-Week Moving Avg", line=dict(color="#EA580C", width=2, dash="dot")))
    fig_store.update_layout(
        **CHART_CONFIG, height=350,
        title=dict(text=f"Store {selected_store_analysis} Sales Trend", font=dict(size=14, color="#172033"), x=0.01),
        xaxis=dict(title="Date", gridcolor="#EEF2F7"),
        yaxis=dict(title="Weekly Sales (USD)", tickformat="$,.0f", gridcolor="#EEF2F7"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_store, use_container_width=True)

    store_month = store_df.copy()
    store_month["Year"] = store_month["Date"].dt.year
    store_month["Month"] = store_month["Date"].dt.month

    month_pivot = store_month.pivot_table(values="Weekly_Sales", index="Year", columns="Month", aggfunc="sum")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_pivot.columns = [month_names.get(int(c), str(c)) for c in month_pivot.columns]

    fig_heatmap = px.imshow(month_pivot, aspect="auto", color_continuous_scale=["#EFF6FF", "#60A5FA", "#1D4ED8"])
    fig_heatmap.update_layout(
        **CHART_CONFIG, height=280,
        title=dict(text="Monthly Seasonality", font=dict(size=14, color="#172033"), x=0.01),
        xaxis=dict(title="Month"), yaxis=dict(title="Year"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    benchmark = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    store_rank = benchmark.rank(ascending=False, method="min")[selected_store_analysis]
    total_store_count = len(benchmark)

    render_insight(
        "Network Position",
        f"Store {selected_store_analysis} ranks {int(store_rank)} of {total_store_count} stores by cumulative revenue.",
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    render_page_header(
        "Dataset Explorer",
        "Inspect the analysis-ready dataset: structure, quality, and summary statistics.",
    )

    d1, d2, d3, d4 = st.columns(4)
    render_kpi(d1, "", "Rows", f"{len(filtered_df):,}", "Records in the active filter.")
    render_kpi(d2, "kpi-green", "Columns", f"{len(filtered_df.columns):,}", "Available variables.")
    render_kpi(d3, "kpi-orange", "Stores", f"{filtered_df['Store'].nunique():,}", "Distinct stores.")
    render_kpi(d4, "kpi-purple", "Weekly Periods", f"{filtered_df['Date'].nunique():,}", "Unique weekly observations.")

    st.markdown("<br>", unsafe_allow_html=True)

    quality = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": [str(filtered_df[c].dtype) for c in filtered_df.columns],
        "Missing Values": [int(filtered_df[c].isna().sum()) for c in filtered_df.columns],
        "Missing %": [round(filtered_df[c].isna().mean() * 100, 2) for c in filtered_df.columns],
    })

    render_section_header("Data Quality Overview", "Missing-value profile for the active dataset view.")
    st.dataframe(quality, use_container_width=True, hide_index=True)

    numeric_columns = filtered_df.select_dtypes(include=np.number).columns.tolist()

    if numeric_columns:
        descriptive = filtered_df[numeric_columns].describe().T.reset_index()
        descriptive.columns = ["Column", "Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]
        render_section_header("Descriptive Statistics", "Summary statistics for numerical variables.")
        st.dataframe(descriptive, use_container_width=True, hide_index=True)

    n_rows = st.selectbox("Rows to display", [50, 100, 250, 500], index=1)
    render_section_header("Data Preview", "Filtered records from the analysis-ready dataset.")
    st.dataframe(filtered_df.head(n_rows), use_container_width=True, height=430)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Dataset", csv_data, file_name="walmart_filtered_dataset.csv", mime="text/csv")


# ============================================================
# ABOUT PLATFORM
# ------------------------------------------------------------
# Previously this page built one giant hand-written HTML string
# per column, indented, which is exactly the pattern that
# triggered the Markdown-code-block bug. Rebuilt using the same
# render_status()/render_section_header() helpers used
# everywhere else, so it can't break independently again.
# ============================================================

elif page == "About Platform":

    render_page_header(
        "About the Platform",
        "Architecture, methodology, dataset coverage, and how to read the forecast.",
    )

    col1, col2 = st.columns(2)

    with col1:
        render_html(
            """
            <div class="section-card">
                <div class="section-title">Platform Overview</div>
                <div class="section-description">End-to-end retail analytics and forecasting platform.</div>
            </div>
            """
        )
        render_status(st.container(), "Dataset", "Walmart Retail Sales", "Historical weekly store-department sales.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(
            st.container(), "Historical Coverage",
            f"{df['Date'].min().strftime('%b %Y')} \u2014 {df['Date'].max().strftime('%b %Y')}",
            f"Last observed date: {df['Date'].max().strftime('%d %b %Y')}.",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(st.container(), "Network", f"{df['Store'].nunique()} Stores", "Multiple departments across store types.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(st.container(), "Data Granularity", "Store \u00d7 Department \u00d7 Week", "Weekly sales observation.")

    with col2:
        render_html(
            """
            <div class="section-card">
                <div class="section-title">Forecasting Engine</div>
                <div class="section-description">Machine learning pipeline used for weekly sales prediction.</div>
            </div>
            """
        )
        render_status(st.container(), "Algorithm", "XGBoost Regressor", "Gradient boosting for structured retail data.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(st.container(), "Validation Strategy", "Chronological Train / Test Split", "Historical data separated from the held-out test period.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(st.container(), "R\u00b2", "0.9853", "Held-out test performance.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_status(st.container(), "WAPE", "8.07%", "Weighted forecast error.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Model Feature Groups", "Variables used by the forecasting pipeline.")

    feature_groups = pd.DataFrame({
        "Feature Group": ["Store & Department", "Calendar", "Holiday", "Economic", "Time-Series Lag", "Rolling Statistics"],
        "Examples": [
            "Store, Dept, Type_Code, Size",
            "Year, Month, Week, Quarter, DayOfWeek",
            "IsHoliday",
            "Temperature, Fuel_Price, CPI, Unemployment",
            "Lag_1_Week_Sales, Lag_4_Week_Sales, Lag_12_Week_Sales",
            "Rolling_Mean_4, Rolling_Mean_12, Rolling_Std_4",
        ],
    })
    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Business Interpretation", "How to use this output in a real planning context.")

    render_insight("Planning", "Use the forecast for inventory allocation, staffing, and capacity planning.")
    render_insight("Performance Monitoring", "Use actual-vs-predicted results to flag stores or periods needing investigation.")
    render_insight("Holiday Planning", "Past holiday effects inform planning, but future impact stays uncertain \u2014 promotions and behavior aren't known in advance.")
    render_insight("Model Limitation", "This is decision support, not a guaranteed number. Promotions, competitor moves, and weather can still shift actuals.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Analytical Workflow", "End-to-end project pipeline.")

    workflow = pd.DataFrame({
        "Stage": ["1", "2", "3", "4", "5", "6", "7"],
        "Process": [
            "Raw Data", "Cleaning & Validation", "Time-Series Feature Engineering",
            "Chronological Train/Test Split", "XGBoost Training", "Model Evaluation", "12-Week Forecast",
        ],
    })
    st.dataframe(workflow, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <br>
    <hr>
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:9px; color:#94A3B8; padding-top:5px;">
        <div><strong style="color:#475569;">Walmart Retail Intelligence Platform</strong></div>
        <div>Sales Analytics &middot; Machine Learning &middot; Forecasting</div>
    </div>
    """
)