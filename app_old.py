# ============================================================
# WALMART SALES FORECASTING DASHBOARD
# PROFESSIONAL STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Walmart Forecast Dashboard",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # Main featured dataset
    df = pd.read_csv(
        "data/processed/walmart_featured.csv"
    )

    # Forecast results
    forecast_df = pd.read_csv(
        "data/processed/pro_xgboost_results.csv"
    )

    return df, forecast_df


df, forecast_df = load_data()

# ============================================================
# TITLE SECTION
# ============================================================

st.title("📊 Walmart Sales Forecast Dashboard")

st.markdown(
    "Professional Data Science Forecasting Application"
)

# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📌 Business KPI Overview")

col1, col2, col3, col4 = st.columns(4)

total_sales = df["Weekly_Sales"].sum()

average_sales = df["Weekly_Sales"].mean()

total_stores = df["Store"].nunique()

total_departments = df["Dept"].nunique()

with col1:

    st.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

with col2:

    st.metric(
        "Average Weekly Sales",
        f"${average_sales:,.0f}"
    )

with col3:

    st.metric(
        "Total Stores",
        total_stores
    )

with col4:

    st.metric(
        "Departments",
        total_departments
    )

# ============================================================
# DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

# ============================================================
# MONTHLY SALES TREND
# ============================================================

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    df.groupby(
        df["Date"].dt.to_period("M")
    )["Weekly_Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(14, 5))

monthly_sales.plot(ax=ax)

ax.set_title("Monthly Sales Trend")

ax.set_xlabel("Month")

ax.set_ylabel("Total Sales")

plt.xticks(rotation=45)

st.pyplot(fig)

# ============================================================
# TOP 10 STORES
# ============================================================

st.subheader("🏪 Top 10 Stores by Sales")

top_stores = (
    df.groupby("Store")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 5))

top_stores.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Top 10 Stores")

ax.set_xlabel("Store")

ax.set_ylabel("Total Sales")

st.pyplot(fig)

# ============================================================
# FORECAST RESULTS
# ============================================================

st.subheader("🤖 Forecast vs Actual Sales")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    forecast_df["Actual_Sales"][:200],
    label="Actual Sales"
)

ax.plot(
    forecast_df["Predicted_Sales"][:200],
    label="Predicted Sales"
)

ax.set_title("XGBoost Forecast Comparison")

ax.set_xlabel("Observations")

ax.set_ylabel("Sales")

ax.legend()

st.pyplot(fig)

# ============================================================
# STORE FILTER ANALYSIS
# ============================================================

st.subheader("🔍 Store Analysis")

store_option = st.selectbox(
    "Select Store",
    sorted(df["Store"].unique())
)

store_data = df[
    df["Store"] == store_option
]

store_sales = (
    store_data.groupby("Date")["Weekly_Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(14, 5))

store_sales.plot(ax=ax)

ax.set_title(
    f"Store {store_option} Sales Trend"
)

ax.set_xlabel("Date")

ax.set_ylabel("Sales")

st.pyplot(fig)

# ============================================================
# STORE TYPE PERFORMANCE
# ============================================================

st.subheader("🏬 Store Type Performance")

store_type_sales = (
    df.groupby("Type")["Weekly_Sales"]
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 5))

store_type_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Sales by Store Type")

ax.set_xlabel("Store Type")

ax.set_ylabel("Average Weekly Sales")

st.pyplot(fig)

# ============================================================
# HOLIDAY SALES ANALYSIS
# ============================================================

st.subheader("🎄 Holiday vs Non-Holiday Sales")

holiday_sales = (
    df.groupby("IsHoliday")["Weekly_Sales"]
    .mean()
)

fig, ax = plt.subplots(figsize=(7, 5))

holiday_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Holiday Sales Comparison")

ax.set_xlabel("Holiday")

ax.set_ylabel("Average Weekly Sales")

st.pyplot(fig)

# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("🗂 Dataset Preview")

st.dataframe(
    df.head(20)
)

# ============================================================
# FEATURE IMPORTANCE IMAGE
# ============================================================

st.subheader("⭐ Feature Importance")

st.image(
    "data/processed/pro_xgboost_feature_importance.png"
)

# ============================================================
# FUTURE FORECAST PREVIEW
# ============================================================

st.subheader("📅 Future Forecast Preview")

future_forecast = pd.read_csv(
    "data/processed/future_sales_forecast.csv"
)

st.dataframe(
    future_forecast.head(20)
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    ### 🚀 Professional Walmart Forecasting Project

    Built with:
    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - Streamlit

    Project Status: Portfolio Ready
    """
)
