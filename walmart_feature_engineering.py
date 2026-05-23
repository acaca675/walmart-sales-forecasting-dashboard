# ============================================================
# WALMART SALES FORECASTING
# FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("WALMART SALES — FEATURE ENGINEERING")
print("=" * 60)

DATA_PATH = "data/processed/walmart_clean.csv"

print("\nLoading clean dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ============================================================
# CONVERT DATE COLUMN
# ============================================================

print("\nConverting Date column...")

df["Date"] = pd.to_datetime(df["Date"])

print("Date conversion completed!")

# ============================================================
# BASIC TIME FEATURES
# ============================================================

print("\nCreating basic time features...")

# Year
df["Year"] = df["Date"].dt.year

# Month
df["Month"] = df["Date"].dt.month

# Week
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Quarter
df["Quarter"] = df["Date"].dt.quarter

# Day of Week
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Weekend Feature
df["IsWeekend"] = np.where(
    df["DayOfWeek"] >= 5,
    1,
    0
)

print("Basic time features created successfully!")

# ============================================================
# SALES LAG FEATURES
# ============================================================

print("\nCreating lag features...")

# Sort dataset first
df = df.sort_values(
    by=["Store", "Dept", "Date"]
)

# Previous week sales
df["Lag_1_Week_Sales"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(1)
)

# Previous 4 weeks sales
df["Lag_4_Week_Sales"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(4)
)

print("Lag features created successfully!")

# ============================================================
# ROLLING FEATURES
# ============================================================

print("\nCreating rolling mean features...")

# Rolling 4-week average
df["Rolling_Mean_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
)

# Rolling 12-week average
df["Rolling_Mean_12"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(
        lambda x: x.shift(1).rolling(12).mean()
    )
)

print("Rolling features created successfully!")

# ============================================================
# HANDLE NEW MISSING VALUES
# ============================================================

print("\nHandling missing values from lag features...")

lag_columns = [
    "Lag_1_Week_Sales",
    "Lag_4_Week_Sales",
    "Rolling_Mean_4",
    "Rolling_Mean_12"
]

for column in lag_columns:

    df[column] = df[column].fillna(0)

print("Lag missing values cleaned!")

# ============================================================
# FINAL FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print(f"\nFinal Dataset Shape : {df.shape}")

print("\nNew Features Added:\n")

new_features = [
    "Year",
    "Month",
    "Week",
    "Quarter",
    "DayOfWeek",
    "IsWeekend",
    "Lag_1_Week_Sales",
    "Lag_4_Week_Sales",
    "Rolling_Mean_4",
    "Rolling_Mean_12"
]

for feature in new_features:

    print(f"- {feature}")

# ============================================================
# SAVE FEATURED DATASET
# ============================================================

print("\nSaving featured dataset...")

output_path = "data/processed/walmart_featured.csv"

df.to_csv(output_path, index=False)

print("Featured dataset saved successfully!")

print(f"\nSaved File:\n{output_path}")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)
