# ============================================================
# WALMART SALES FORECASTING
# NOTEBOOK 02 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# LOAD CLEAN DATASET
# ============================================================

print("=" * 60)
print("WALMART SALES — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

DATA_PATH = "data/processed/walmart_clean.csv"

print("\nLoading clean dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

print(f"\nTotal Stores  : {df['Store'].nunique():,}")
print(f"Total Depts   : {df['Dept'].nunique():,}")

print("\nColumns:\n")
print(df.columns.tolist())

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

if len(missing_values) == 0:

    print("\nNo missing values found!")

else:

    print("\nMissing Values:\n")

    for column, count in missing_values.items():

        print(f"{column:<15}: {count:,}")

# ============================================================
# WEEKLY SALES STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("WEEKLY SALES STATISTICS")
print("=" * 60)

sales_stats = df["Weekly_Sales"].describe()

print("\nWeekly_Sales Summary:\n")

print(sales_stats)

# ============================================================
# TOP 10 STORES BY SALES
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 STORES BY TOTAL SALES")
print("=" * 60)

top_stores = (
    df.groupby("Store")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Stores:\n")

print(top_stores)

# ============================================================
# SALES DISTRIBUTION PLOT
# ============================================================

print("\nCreating sales distribution plot...")

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Weekly_Sales"],
    bins=50
)

plt.title("Weekly Sales Distribution")

plt.xlabel("Weekly Sales")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("data/processed/sales_distribution.png")

print("Sales distribution plot saved successfully!")

# ============================================================
# TOP STORES VISUALIZATION
# ============================================================

print("\nCreating top store visualization...")

plt.figure(figsize=(12, 6))

top_stores.plot(kind="bar")

plt.title("Top 10 Stores by Total Sales")

plt.xlabel("Store")

plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig("data/processed/top_10_stores.png")

print("Top stores chart saved successfully!")

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("- sales_distribution.png")
print("- top_10_stores.png")

# ============================================================
# ADVANCED EDA
# ============================================================

# Convert Date column into datetime
df["Date"] = pd.to_datetime(df["Date"])

# ============================================================
# MONTHLY SALES TREND
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY SALES TREND ANALYSIS")
print("=" * 60)

# Create Year-Month column
df["YearMonth"] = df["Date"].dt.to_period("M")

monthly_sales = (
    df.groupby("YearMonth")["Weekly_Sales"]
    .sum()
)

print("\nMonthly Sales Preview:\n")
print(monthly_sales.head())

# Plot monthly sales trend
plt.figure(figsize=(14, 6))

monthly_sales.plot()

plt.title("Monthly Sales Trend")

plt.xlabel("Month")

plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("data/processed/monthly_sales_trend.png")

print("\nMonthly sales trend chart saved!")

# ============================================================
# HOLIDAY IMPACT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("HOLIDAY IMPACT ANALYSIS")
print("=" * 60)

holiday_sales = (
    df.groupby("IsHoliday")["Weekly_Sales"]
    .mean()
)

print("\nAverage Weekly Sales:\n")
print(holiday_sales)

# Plot holiday comparison
plt.figure(figsize=(6, 5))

holiday_sales.plot(kind="bar")

plt.title("Holiday vs Non-Holiday Sales")

plt.xlabel("IsHoliday")

plt.ylabel("Average Weekly Sales")

plt.tight_layout()

plt.savefig("data/processed/holiday_sales_comparison.png")

print("\nHoliday comparison chart saved!")

# ============================================================
# CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION HEATMAP")
print("=" * 60)

numeric_columns = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Size",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5"
]

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig("data/processed/correlation_heatmap.png")

print("\nCorrelation heatmap saved!")

# ============================================================
# TOP 10 DEPARTMENTS
# ============================================================

print("\n" + "=" * 60)
print("TOP DEPARTMENTS ANALYSIS")
print("=" * 60)

top_departments = (
    df.groupby("Dept")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop Departments:\n")
print(top_departments)

# Plot top departments
plt.figure(figsize=(12, 6))

top_departments.plot(kind="bar")

plt.title("Top 10 Departments by Sales")

plt.xlabel("Department")

plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig("data/processed/top_departments.png")

print("\nTop departments chart saved!")

# ============================================================
# STORE TYPE PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("STORE TYPE PERFORMANCE")
print("=" * 60)

store_type_sales = (
    df.groupby("Type")["Weekly_Sales"]
    .mean()
)

print("\nAverage Sales by Store Type:\n")
print(store_type_sales)

# Plot store type performance
plt.figure(figsize=(7, 5))

store_type_sales.plot(kind="bar")

plt.title("Average Sales by Store Type")

plt.xlabel("Store Type")

plt.ylabel("Average Weekly Sales")

plt.tight_layout()

plt.savefig("data/processed/store_type_performance.png")

print("\nStore type performance chart saved!")

# ============================================================
# SEASONALITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SEASONALITY ANALYSIS")
print("=" * 60)

# Extract month
df["Month"] = df["Date"].dt.month

seasonality = (
    df.groupby("Month")["Weekly_Sales"]
    .mean()
)

print("\nAverage Sales by Month:\n")
print(seasonality)

# Plot seasonality
plt.figure(figsize=(12, 5))

seasonality.plot(marker="o")

plt.title("Average Monthly Seasonality")

plt.xlabel("Month")

plt.ylabel("Average Weekly Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig("data/processed/seasonality_analysis.png")

print("\nSeasonality chart saved!")

# ============================================================
# OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

plt.figure(figsize=(12, 5))

sns.boxplot(x=df["Weekly_Sales"])

plt.title("Weekly Sales Outlier Detection")

plt.tight_layout()

plt.savefig("data/processed/outlier_detection.png")

print("\nOutlier detection chart saved!")

# ============================================================
# FINAL ADVANCED EDA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ADVANCED EDA COMPLETED")
print("=" * 60)

print("\nGenerated Advanced EDA Files:\n")

advanced_files = [
    "monthly_sales_trend.png",
    "holiday_sales_comparison.png",
    "correlation_heatmap.png",
    "top_departments.png",
    "store_type_performance.png",
    "seasonality_analysis.png",
    "outlier_detection.png"
]

for file in advanced_files:

    print(f"- {file}")
