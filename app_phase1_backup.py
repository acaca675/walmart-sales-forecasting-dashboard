import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0F172A;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

div[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

h1, h2, h3, h4 {
    color: white;
}

.stDataFrame {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/walmart_featured.csv"
    )

    forecast_df = pd.read_csv(
        "data/processed/pro_xgboost_results.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df, forecast_df

df, forecast_df = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Sales Analytics",
        "Forecasting",
        "Store Performance",
        "Dataset"
    ]
)

# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title("Executive Retail Overview")

    st.markdown(
        "Enterprise Retail Sales Forecasting and Business Intelligence Platform"
    )

    total_sales = df["Weekly_Sales"].sum()

    avg_sales = df["Weekly_Sales"].mean()

    total_stores = df["Store"].nunique()

    total_departments = df["Dept"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "Average Weekly Sales",
        f"${avg_sales:,.0f}"
    )

    col3.metric(
        "Stores",
        total_stores
    )

    col4.metric(
        "Departments",
        total_departments
    )

    st.markdown("---")

    monthly_sales = (
        df.groupby(
            df["Date"].dt.to_period("M")
        )["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Date"] = monthly_sales["Date"].astype(str)

    fig = px.line(
        monthly_sales,
        x="Date",
        y="Weekly_Sales",
        title="Monthly Sales Trend"
    )

    fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    st.title("Sales Analytics")

    top_stores = (
        df.groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_stores,
        x="Store",
        y="Weekly_Sales",
        title="Top 10 Stores by Sales"
    )

    fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    seasonality = (
        df.groupby(
            df["Date"].dt.month
        )["Weekly_Sales"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(
        seasonality,
        x="Date",
        y="Weekly_Sales",
        title="Seasonality Analysis"
    )

    fig2.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    st.title("Forecasting Performance")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=forecast_df["Actual_Sales"][:200],
            mode='lines',
            name='Actual Sales'
        )
    )

    fig.add_trace(
        go.Scatter(
            y=forecast_df["Predicted_Sales"][:200],
            mode='lines',
            name='Predicted Sales'
        )
    )

    fig.update_layout(
        title="Forecast vs Actual Sales",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R² Score",
        "0.9840"
    )

    col2.metric(
        "RMSE",
        "2,772.87"
    )

    col3.metric(
        "MAE",
        "1,312.98"
    )

# ============================================================
# STORE PERFORMANCE
# ============================================================

elif page == "Store Performance":

    st.title("Store Performance Analysis")

    store_option = st.selectbox(
        "Select Store",
        sorted(df["Store"].unique())
    )

    store_data = df[
        df["Store"] == store_option
    ]

    sales_trend = (
        store_data.groupby("Date")["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        sales_trend,
        x="Date",
        y="Weekly_Sales",
        title=f"Store {store_option} Sales Trend"
    )

    fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.title("Dataset Preview")

    st.dataframe(
        df.head(100)
    )
