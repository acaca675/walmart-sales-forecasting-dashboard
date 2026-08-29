# ============================================================
# WALMART SALES FORECASTING
# XGBOOST FORECASTING & MODEL EVALUATION PIPELINE
# ============================================================
#
# Project:
#   Walmart Sales Forecasting
#
# Objective:
#   Build a leakage-aware machine learning forecasting pipeline
#   to predict weekly Walmart sales.
#
# Workflow:
#
#   Raw / Featured Data
#          ↓
#   Data Validation
#          ↓
#   Time-Series Feature Engineering
#          ↓
#   Chronological Train/Test Split
#          ↓
#   Naive Baseline
#          ↓
#   XGBoost Model
#          ↓
#   Model Evaluation
#          ↓
#   Error Analysis
#          ↓
#   Feature Importance
#          ↓
#   12-Week Recursive Forecast
#          ↓
#   Export Results
#
# Important:
#   - No random train/test split
#   - Lag features use historical observations only
#   - Rolling features use SHIFT(1) to prevent target leakage
#   - Train/test split is based on unique chronological dates
#   - Future forecast is generated recursively
#
# Author: Nabilla Salwa Salsabilla
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train / test ratio
TRAIN_RATIO = 0.80

# Future forecast horizon
FORECAST_HORIZON = 12

# Dataset
DATA_PATH = "data/processed/walmart_featured.csv"

# Output directories
OUTPUT_DIR = "data/processed"
MODEL_DIR = "data/processed/models"

# Create output directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. DISPLAY PROJECT HEADER
# ============================================================

print("=" * 70)
print("WALMART SALES FORECASTING")
print("XGBOOST FORECASTING & MODEL EVALUATION PIPELINE")
print("=" * 70)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n[1/12] Loading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Please make sure walmart_featured.csv exists "
        "inside data/processed/."
    )


df = pd.read_csv(DATA_PATH)


print("Dataset loaded successfully.")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 5. DATA VALIDATION
# ============================================================

print("\n[2/12] Validating dataset...")


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "Date",
    "Store",
    "Dept",
    "Weekly_Sales"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"- {col}"
            for col in missing_columns
        )
    )


# ------------------------------------------------------------
# Date conversion
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


if df["Date"].isna().any():

    invalid_dates = df["Date"].isna().sum()

    raise ValueError(
        f"{invalid_dates:,} invalid date values found."
    )


# ------------------------------------------------------------
# Remove exact duplicates
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()


if duplicate_count > 0:

    print(
        f"Removing {duplicate_count:,} duplicate rows..."
    )

    df = df.drop_duplicates()


# ------------------------------------------------------------
# Validate target
# ------------------------------------------------------------

if df["Weekly_Sales"].isna().any():

    missing_target = (
        df["Weekly_Sales"]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{missing_target:,} missing Weekly_Sales values found."
    )


# ------------------------------------------------------------
# Initial chronological sorting
#
# Required before creating lag features.
# ------------------------------------------------------------

df = (
    df
    .sort_values(
        ["Store", "Dept", "Date"]
    )
    .reset_index(drop=True)
)


print(
    f"Date range: "
    f"{df['Date'].min().date()} "
    f"to "
    f"{df['Date'].max().date()}"
)


print(
    f"Stores: "
    f"{df['Store'].nunique():,}"
)


print(
    f"Departments: "
    f"{df['Dept'].nunique():,}"
)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================
#
# Important:
#
# Lag features are calculated separately for each Store × Dept.
#
# Rolling features use SHIFT(1), meaning:
#
# Current Week
#      ↑
#      |
# Previous historical observations only
#
# This prevents current Weekly_Sales from leaking
# into the features used to predict that same week.
# ============================================================

print("\n[3/12] Creating time-series features...")


group_columns = [
    "Store",
    "Dept"
]


# ------------------------------------------------------------
# Calendar features
# ------------------------------------------------------------

df["Year"] = (
    df["Date"]
    .dt.year
)

df["Month"] = (
    df["Date"]
    .dt.month
)

df["Week"] = (
    df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["Quarter"] = (
    df["Date"]
    .dt.quarter
)

df["DayOfWeek"] = (
    df["Date"]
    .dt.dayofweek
)

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)


# ------------------------------------------------------------
# Holiday
# ------------------------------------------------------------

if "IsHoliday" not in df.columns:

    df["IsHoliday"] = 0

else:

    df["IsHoliday"] = (
        pd.to_numeric(
            df["IsHoliday"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------------------
# Lag 1
# ------------------------------------------------------------

df["Lag_1_Week_Sales"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .shift(1)
)


# ------------------------------------------------------------
# Lag 4
# ------------------------------------------------------------

df["Lag_4_Week_Sales"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .shift(4)
)


# ------------------------------------------------------------
# Lag 12
# ------------------------------------------------------------

df["Lag_12_Week_Sales"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .shift(12)
)


# ------------------------------------------------------------
# Rolling Mean 4
# ------------------------------------------------------------

df["Rolling_Mean_4"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=4,
            min_periods=2
        )
        .mean()
    )
)


# ------------------------------------------------------------
# Rolling Mean 12
# ------------------------------------------------------------

df["Rolling_Mean_12"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=12,
            min_periods=4
        )
        .mean()
    )
)


# ------------------------------------------------------------
# Rolling Standard Deviation
# ------------------------------------------------------------

df["Rolling_Std_4"] = (
    df
    .groupby(group_columns)["Weekly_Sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=4,
            min_periods=2
        )
        .std()
    )
)


# ------------------------------------------------------------
# Remove rows without sufficient historical information
# ------------------------------------------------------------

required_lag_features = [
    "Lag_1_Week_Sales",
    "Lag_4_Week_Sales",
    "Lag_12_Week_Sales",
    "Rolling_Mean_4",
    "Rolling_Mean_12"
]


df_model = (
    df
    .dropna(
        subset=required_lag_features
    )
    .copy()
)


# ------------------------------------------------------------
# IMPORTANT:
# After feature engineering, sort chronologically.
#
# This ordering is used for the train/test date split.
# ------------------------------------------------------------

df_model = (
    df_model
    .sort_values(
        ["Date", "Store", "Dept"]
    )
    .reset_index(drop=True)
)


print(
    f"Rows available for modeling: "
    f"{len(df_model):,}"
)


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

print("\n[4/12] Selecting model features...")


candidate_features = [

    "Store",

    "Dept",

    "IsHoliday",

    "Temperature",

    "Fuel_Price",

    "CPI",

    "Unemployment",

    "Size",

    "Type_Code",

    "Year",

    "Month",

    "Week",

    "Quarter",

    "DayOfWeek",

    "IsWeekend",

    "Lag_1_Week_Sales",

    "Lag_4_Week_Sales",

    "Lag_12_Week_Sales",

    "Rolling_Mean_4",

    "Rolling_Mean_12",

    "Rolling_Std_4"

]


# Keep only columns available in the dataset
feature_columns = [
    col
    for col in candidate_features
    if col in df_model.columns
]


target_column = "Weekly_Sales"


if len(feature_columns) == 0:

    raise ValueError(
        "No valid model features were found."
    )


print(
    f"Selected features: "
    f"{len(feature_columns)}"
)


print("\nFeatures:")

for feature in feature_columns:

    print(
        f"- {feature}"
    )


# ============================================================
# 8. PREPARE FEATURE MATRIX
# ============================================================

print("\n[5/12] Preparing feature matrix...")


X = (
    df_model[
        feature_columns
    ]
    .copy()
)


y = (
    df_model[
        target_column
    ]
    .copy()
)


# ------------------------------------------------------------
# Convert non-numeric features safely
# ------------------------------------------------------------

categorical_columns = (
    X
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


if categorical_columns:

    print(
        "\nEncoding categorical columns:"
    )

    for column in categorical_columns:

        print(
            f"- {column}"
        )

        # Convert categories to stable integer codes
        X[column] = (
            X[column]
            .astype("category")
            .cat.codes
        )


# ------------------------------------------------------------
# Convert all remaining columns to numeric where possible
# ------------------------------------------------------------

for column in X.columns:

    if not pd.api.types.is_numeric_dtype(
        X[column]
    ):

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# Replace infinite values
# ------------------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# Fill remaining missing values
# ------------------------------------------------------------

for column in X.columns:

    if X[column].isna().any():

        median_value = (
            X[column]
            .median()
        )

        if pd.isna(median_value):

            median_value = 0

        X[column] = (
            X[column]
            .fillna(median_value)
        )


# ============================================================
# 9. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
#
# We split using UNIQUE DATES rather than row count.
#
# This guarantees that the same date cannot appear
# simultaneously in training and testing data.
#
# Example:
#
# 2010 ───────────── 2022-04-13 | 2022-04-20 ───────── 2012
#             TRAIN             |          TEST
#
# ============================================================

print(
    "\n[6/12] Performing chronological train/test split..."
)


# ------------------------------------------------------------
# Get unique dates
# ------------------------------------------------------------

unique_dates = (
    df_model[
        "Date"
    ]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)


if len(unique_dates) < 2:

    raise ValueError(
        "Not enough unique dates for train/test split."
    )


# ------------------------------------------------------------
# Determine cutoff date
# ------------------------------------------------------------

split_date_index = int(
    len(unique_dates)
    * TRAIN_RATIO
)


# Protect against invalid index
split_date_index = max(
    1,
    min(
        split_date_index,
        len(unique_dates) - 1
    )
)


cutoff_date = (
    unique_dates[
        split_date_index
    ]
)


# ------------------------------------------------------------
# Create masks
# ------------------------------------------------------------

train_mask = (
    df_model["Date"]
    < cutoff_date
)


test_mask = (
    df_model["Date"]
    >= cutoff_date
)


# ------------------------------------------------------------
# Split X
# ------------------------------------------------------------

X_train = (
    X.loc[
        train_mask
    ]
    .copy()
)


X_test = (
    X.loc[
        test_mask
    ]
    .copy()
)


# ------------------------------------------------------------
# Split y
# ------------------------------------------------------------

y_train = (
    y.loc[
        train_mask
    ]
    .copy()
)


y_test = (
    y.loc[
        test_mask
    ]
    .copy()
)


# ------------------------------------------------------------
# Date information
# ------------------------------------------------------------

train_dates = (
    df_model.loc[
        train_mask,
        "Date"
    ]
)


test_dates = (
    df_model.loc[
        test_mask,
        "Date"
    ]
)


# ------------------------------------------------------------
# Validate split
# ------------------------------------------------------------

if len(X_train) == 0:

    raise ValueError(
        "Training set is empty."
    )


if len(X_test) == 0:

    raise ValueError(
        "Testing set is empty."
    )


if train_dates.max() >= test_dates.min():

    raise ValueError(
        "\nInvalid chronological split.\n"
        "Training and testing periods overlap."
    )


# ------------------------------------------------------------
# Display split
# ------------------------------------------------------------

print("\nTraining period:")

print(
    f"{train_dates.min().date()} "
    f"→ "
    f"{train_dates.max().date()}"
)


print("\nTesting period:")

print(
    f"{test_dates.min().date()} "
    f"→ "
    f"{test_dates.max().date()}"
)


print("\nTraining rows:")

print(
    f"{len(X_train):,}"
)


print("\nTesting rows:")

print(
    f"{len(X_test):,}"
)


print("\nUnique training dates:")

print(
    f"{train_dates.nunique():,}"
)


print("\nUnique testing dates:")

print(
    f"{test_dates.nunique():,}"
)


# ============================================================
# 10. NAIVE BASELINE
# ============================================================
#
# Baseline prediction:
#
#   Current sales = previous week's sales
#
# This establishes whether XGBoost actually provides
# additional predictive value.
# ============================================================

print(
    "\n[7/12] Evaluating naive baseline..."
)


naive_predictions = (
    df_model.loc[
        test_mask,
        "Lag_1_Week_Sales"
    ]
    .values
)


# ------------------------------------------------------------
# Naive MAE
# ------------------------------------------------------------

naive_mae = mean_absolute_error(
    y_test,
    naive_predictions
)


# ------------------------------------------------------------
# Naive RMSE
# ------------------------------------------------------------

naive_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        naive_predictions
    )
)


# ------------------------------------------------------------
# Naive R²
# ------------------------------------------------------------

naive_r2 = r2_score(
    y_test,
    naive_predictions
)


# ------------------------------------------------------------
# Naive WAPE
# ------------------------------------------------------------

naive_wape = (
    np.sum(
        np.abs(
            y_test.values
            - naive_predictions
        )
    )
    /
    np.sum(
        np.abs(
            y_test.values
        )
    )
)


print(
    "\nNaive Baseline Performance"
)


print(
    f"MAE  : {naive_mae:,.2f}"
)


print(
    f"RMSE : {naive_rmse:,.2f}"
)


print(
    f"R²   : {naive_r2:.4f}"
)


print(
    f"WAPE : {naive_wape:.2%}"
)


# ============================================================
# 11. XGBOOST MODEL
# ============================================================

print(
    "\n[8/12] Training XGBoost model..."
)


model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=8,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train,
    verbose=False
)


print(
    "XGBoost training completed."
)


# ============================================================
# 12. XGBOOST PREDICTIONS
# ============================================================

print(
    "\nGenerating XGBoost predictions..."
)


predictions = (
    model
    .predict(X_test)
)


# Prevent negative sales predictions
predictions = np.maximum(
    predictions,
    0
)


print(
    "Predictions completed."
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

print(
    "\n[9/12] Evaluating XGBoost model..."
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

r2 = r2_score(
    y_test,
    predictions
)


# ------------------------------------------------------------
# WAPE
# ------------------------------------------------------------

wape = (
    np.sum(
        np.abs(
            y_test.values
            - predictions
        )
    )
    /
    np.sum(
        np.abs(
            y_test.values
        )
    )
)


# ------------------------------------------------------------
# Improvement versus baseline
# ------------------------------------------------------------

mae_improvement = (
    1
    -
    (
        mae
        /
        naive_mae
    )
)


rmse_improvement = (
    1
    -
    (
        rmse
        /
        naive_rmse
    )
)


print(
    "\nXGBoost Performance"
)


print(
    f"MAE  : {mae:,.2f}"
)


print(
    f"RMSE : {rmse:,.2f}"
)


print(
    f"R²   : {r2:.4f}"
)


print(
    f"WAPE : {wape:.2%}"
)


print(
    "\nImprovement vs Naive Baseline"
)


print(
    f"MAE improvement  : "
    f"{mae_improvement:.2%}"
)


print(
    f"RMSE improvement : "
    f"{rmse_improvement:.2%}"
)


# ============================================================
# 14. MODEL EVALUATION TABLE
# ============================================================

evaluation_df = pd.DataFrame({

    "Model": [
        "Naive Baseline",
        "XGBoost"
    ],

    "MAE": [
        naive_mae,
        mae
    ],

    "RMSE": [
        naive_rmse,
        rmse
    ],

    "R2": [
        naive_r2,
        r2
    ],

    "WAPE": [
        naive_wape,
        wape
    ]

})


evaluation_path = os.path.join(
    OUTPUT_DIR,
    "model_evaluation.csv"
)


evaluation_df.to_csv(
    evaluation_path,
    index=False
)


print(
    f"\nEvaluation saved to: "
    f"{evaluation_path}"
)


# ============================================================
# 15. TEST PREDICTION DATASET
# ============================================================

print(
    "\nCreating prediction dataset..."
)


results_df = (
    df_model.loc[
        test_mask,
        [
            "Date",
            "Store",
            "Dept"
        ]
    ]
    .copy()
)


results_df["Actual_Sales"] = (
    y_test.values
)


results_df["Predicted_Sales"] = (
    predictions
)


results_df["Naive_Predicted_Sales"] = (
    naive_predictions
)


# ------------------------------------------------------------
# Error
# ------------------------------------------------------------

results_df["Error"] = (
    results_df["Actual_Sales"]
    -
    results_df["Predicted_Sales"]
)


results_df["Absolute_Error"] = (
    results_df["Error"]
    .abs()
)


# ------------------------------------------------------------
# Absolute percentage error
# ------------------------------------------------------------

results_df[
    "Absolute_Percentage_Error"
] = np.where(

    results_df["Actual_Sales"] != 0,

    results_df["Absolute_Error"]
    /
    results_df["Actual_Sales"].abs(),

    np.nan

)


# ------------------------------------------------------------
# Forecast direction
# ------------------------------------------------------------

results_df["Prediction_Direction"] = np.where(

    results_df["Error"] > 0,

    "Underprediction",

    np.where(

        results_df["Error"] < 0,

        "Overprediction",

        "Exact"

    )

)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)


results_df.to_csv(
    prediction_path,
    index=False
)


print(
    f"Prediction results saved to: "
    f"{prediction_path}"
)


# ============================================================
# 16. FORECAST ERROR ANALYSIS
# ============================================================

print(
    "\n[10/12] Performing forecast error analysis..."
)


# ------------------------------------------------------------
# Overall error
# ------------------------------------------------------------

mean_error = (
    results_df["Error"]
    .mean()
)


median_error = (
    results_df["Error"]
    .median()
)


mean_absolute_error_value = (
    results_df["Absolute_Error"]
    .mean()
)


print(
    "\nOverall Error Analysis"
)


print(
    f"Mean Error   : "
    f"{mean_error:,.2f}"
)


print(
    f"Median Error : "
    f"{median_error:,.2f}"
)


if mean_error > 0:

    model_tendency = (
        "Underprediction"
    )

elif mean_error < 0:

    model_tendency = (
        "Overprediction"
    )

else:

    model_tendency = (
        "Approximately unbiased"
    )


print(
    f"Model tendency: "
    f"{model_tendency}"
)


# ============================================================
# 17. STORE-LEVEL ERROR ANALYSIS
# ============================================================

store_error_df = (
    results_df
    .groupby("Store")
    .agg(

        Actual_Sales=(
            "Actual_Sales",
            "sum"
        ),

        Predicted_Sales=(
            "Predicted_Sales",
            "sum"
        ),

        MAE=(
            "Absolute_Error",
            "mean"
        )

    )
    .reset_index()
)


store_error_df["Error"] = (
    store_error_df["Actual_Sales"]
    -
    store_error_df["Predicted_Sales"]
)


store_error_df["Absolute_Error"] = (
    store_error_df["Error"]
    .abs()
)


store_error_df = (
    store_error_df
    .sort_values(
        "MAE",
        ascending=False
    )
)


store_error_path = os.path.join(
    OUTPUT_DIR,
    "store_forecast_error.csv"
)


store_error_df.to_csv(
    store_error_path,
    index=False
)


# ============================================================
# 18. DEPARTMENT-LEVEL ERROR ANALYSIS
# ============================================================

dept_error_df = (
    results_df
    .groupby("Dept")
    .agg(

        Actual_Sales=(
            "Actual_Sales",
            "sum"
        ),

        Predicted_Sales=(
            "Predicted_Sales",
            "sum"
        ),

        MAE=(
            "Absolute_Error",
            "mean"
        )

    )
    .reset_index()
)


dept_error_df["Error"] = (
    dept_error_df["Actual_Sales"]
    -
    dept_error_df["Predicted_Sales"]
)


dept_error_df["Absolute_Error"] = (
    dept_error_df["Error"]
    .abs()
)


dept_error_df = (
    dept_error_df
    .sort_values(
        "MAE",
        ascending=False
    )
)


dept_error_path = os.path.join(
    OUTPUT_DIR,
    "department_forecast_error.csv"
)


dept_error_df.to_csv(
    dept_error_path,
    index=False
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

print(
    "\n[11/12] Calculating feature importance..."
)


importance_df = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)


print(
    "\nTop 10 Predictive Features"
)


print(
    importance_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "xgboost_sales_forecasting.pkl"
)


joblib.dump(
    model,
    model_path
)


print(
    f"\nModel saved to: "
    f"{model_path}"
)


# ============================================================
# 21. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

print(
    "\nCreating forecast evaluation visualization..."
)


# ------------------------------------------------------------
# Aggregate test results by date
#
# This produces a cleaner business-level time series
# rather than plotting thousands of Store × Dept observations.
# ------------------------------------------------------------

daily_test_df = (
    results_df
    .groupby(
        "Date",
        as_index=False
    )
    .agg(

        Actual_Sales=(
            "Actual_Sales",
            "sum"
        ),

        Predicted_Sales=(
            "Predicted_Sales",
            "sum"
        )

    )
    .sort_values("Date")
)


plt.figure(
    figsize=(15, 6)
)


plt.plot(

    daily_test_df["Date"],

    daily_test_df["Actual_Sales"],

    label="Actual Sales"

)


plt.plot(

    daily_test_df["Date"],

    daily_test_df["Predicted_Sales"],

    label="XGBoost Prediction"

)


plt.title(
    "Walmart Weekly Sales: Actual vs XGBoost Prediction"
)


plt.xlabel(
    "Date"
)


plt.ylabel(
    "Weekly Sales"
)


plt.legend()


plt.xticks(
    rotation=45
)


plt.tight_layout()


forecast_plot_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_actual_vs_predicted.png"
)


plt.savefig(
    forecast_plot_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    f"Forecast plot saved to: "
    f"{forecast_plot_path}"
)


# ============================================================
# 22. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = (
    importance_df
    .head(10)
    .sort_values(
        "Importance"
    )
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)


plt.title(
    "Top 10 XGBoost Predictive Features"
)


plt.xlabel(
    "Importance Score"
)


plt.tight_layout()


importance_plot_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_feature_importance.png"
)


plt.savefig(
    importance_plot_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    f"Feature importance plot saved to: "
    f"{importance_plot_path}"
)


# ============================================================
# 23. FUTURE 12-WEEK RECURSIVE FORECAST
# ============================================================
#
# This section generates actual future predictions.
#
# Unlike the previous version:
#
#     future_forecast = predictions[:12]
#
# this implementation creates new future dates and recursively
# updates the lag and rolling features using model predictions.
#
# Future external variables:
#
#   Temperature
#   Fuel_Price
#   CPI
#   Unemployment
#
# are carried forward using the latest known values.
#
# This is a modeling assumption and should be disclosed
# as a limitation in the README.
# ============================================================

print(
    "\n[12/12] Generating 12-week future forecast..."
)


# ------------------------------------------------------------
# Latest historical date
# ------------------------------------------------------------

last_date = (
    df_model["Date"]
    .max()
)


future_dates = pd.date_range(

    start=
        last_date
        + pd.Timedelta(weeks=1),

    periods=
        FORECAST_HORIZON,

    freq="7D"

)


# ------------------------------------------------------------
# Latest observation for each Store × Dept
# ------------------------------------------------------------

latest_rows = (

    df_model

    .sort_values(
        [
            "Store",
            "Dept",
            "Date"
        ]
    )

    .groupby(
        [
            "Store",
            "Dept"
        ],
        as_index=False
    )

    .tail(1)

    .copy()

)


# ------------------------------------------------------------
# Historical sales dictionary
#
# Used to generate recursive lag and rolling features.
# ------------------------------------------------------------

history = {}


for (
    store,
    dept
), group in (

    df_model
    .groupby(
        [
            "Store",
            "Dept"
        ]
    )

):

    history[
        (
            store,
            dept
        )
    ] = (

        group
        .sort_values("Date")
        ["Weekly_Sales"]
        .tolist()

    )


# ------------------------------------------------------------
# Future Holiday Calendar
# ------------------------------------------------------------
#
# The original Walmart dataset flags specific weekly periods
# associated with major holidays.
#
# For future forecasting, historical lookup alone is not enough
# because future dates do not exist in the training dataset.
#
# Therefore, known holiday weeks within the forecast horizon
# are explicitly defined.
#
# Walmart holiday weeks used in the original dataset include:
#
#   Thanksgiving
#   Christmas
#   Super Bowl
#   Labor Day
#
# For the current forecast horizon (Nov 2012 - Jan 2013),
# the relevant holiday periods are Thanksgiving and Christmas.
#
# ------------------------------------------------------------

future_holiday_dates = {

    # Thanksgiving week
    pd.Timestamp("2012-11-23"): 1,

    # Christmas week
    pd.Timestamp("2012-12-28"): 1,

}


# ------------------------------------------------------------
# Historical holiday lookup
# ------------------------------------------------------------

holiday_lookup = (

    df.groupby("Date")["IsHoliday"]
    .max()
    .to_dict()

)


# ------------------------------------------------------------
# Function to determine future holiday status
# ------------------------------------------------------------

def get_holiday_flag(forecast_date):

    forecast_date = pd.Timestamp(
        forecast_date
    )

    return int(
        future_holiday_dates.get(
            forecast_date,
            holiday_lookup.get(
                forecast_date,
                0
            )
        )
    )

    # First check explicitly defined future holidays
    if forecast_date in future_holiday_dates:

        return 1

    # If date exists in historical dataset,
    # use the original Walmart holiday flag
    if forecast_date in holiday_lookup:

        return int(
            holiday_lookup[
                forecast_date
            ]
        )

    # Otherwise assume normal week
    return 0


# ------------------------------------------------------------
# Store future predictions
# ------------------------------------------------------------

future_forecasts = []


# ============================================================
# RECURSIVE FORECAST LOOP
# ============================================================

for future_date in future_dates:

    print(
        f"Forecasting week: "
        f"{future_date.date()}"
    )


    for _, latest in (
        latest_rows.iterrows()
    ):

        store = latest["Store"]

        dept = latest["Dept"]


        key = (
            store,
            dept
        )


        sales_history = (
            history[key]
        )


        # ----------------------------------------------------
        # Lag 1
        # ----------------------------------------------------

        if len(sales_history) >= 1:

            lag_1 = (
                sales_history[-1]
            )

        else:

            lag_1 = 0


        # ----------------------------------------------------
        # Lag 4
        # ----------------------------------------------------

        if len(sales_history) >= 4:

            lag_4 = (
                sales_history[-4]
            )

        else:

            lag_4 = lag_1


        # ----------------------------------------------------
        # Lag 12
        # ----------------------------------------------------

        if len(sales_history) >= 12:

            lag_12 = (
                sales_history[-12]
            )

        else:

            lag_12 = lag_4


        # ----------------------------------------------------
        # Recent 4 weeks
        # ----------------------------------------------------

        recent_4 = (
            sales_history[-4:]
        )


        if len(recent_4) == 0:

            rolling_mean_4 = 0

            rolling_std_4 = 0

        elif len(recent_4) == 1:

            rolling_mean_4 = (
                np.mean(recent_4)
            )

            rolling_std_4 = 0

        else:

            rolling_mean_4 = (
                np.mean(recent_4)
            )

            rolling_std_4 = (
                np.std(
                    recent_4,
                    ddof=1
                )
            )


        # ----------------------------------------------------
        # Recent 12 weeks
        # ----------------------------------------------------

        recent_12 = (
            sales_history[-12:]
        )


        if len(recent_12) > 0:

            rolling_mean_12 = (
                np.mean(recent_12)
            )

        else:

            rolling_mean_12 = 0


        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        future_year = (
            future_date.year
        )


        future_month = (
            future_date.month
        )


        future_week = (
            int(
                future_date
                .isocalendar()
                .week
            )
        )


        future_quarter = (
            future_date.quarter
        )


        future_day_of_week = (
            future_date.dayofweek
        )


        future_is_weekend = int(
            future_day_of_week >= 5
        )


        # ----------------------------------------------------
        # Holiday
        #
        # Known historical dates are used where available.
        # Unknown future dates default to 0.
        # ----------------------------------------------------

        future_is_holiday = get_holiday_flag(
    future_date
    )


        # ----------------------------------------------------
        # Create future feature row
        # ----------------------------------------------------

        future_row = {}


        for feature in feature_columns:


            if feature == "Store":

                future_row[
                    feature
                ] = store


            elif feature == "Dept":

                future_row[
                    feature
                ] = dept


            elif feature == "IsHoliday":

                future_row[
                    feature
                ] = future_is_holiday


            elif feature == "Year":

                future_row[
                    feature
                ] = future_year


            elif feature == "Month":

                future_row[
                    feature
                ] = future_month


            elif feature == "Week":

                future_row[
                    feature
                ] = future_week


            elif feature == "Quarter":

                future_row[
                    feature
                ] = future_quarter


            elif feature == "DayOfWeek":

                future_row[
                    feature
                ] = future_day_of_week


            elif feature == "IsWeekend":

                future_row[
                    feature
                ] = future_is_weekend


            elif feature == "Lag_1_Week_Sales":

                future_row[
                    feature
                ] = lag_1


            elif feature == "Lag_4_Week_Sales":

                future_row[
                    feature
                ] = lag_4


            elif feature == "Lag_12_Week_Sales":

                future_row[
                    feature
                ] = lag_12


            elif feature == "Rolling_Mean_4":

                future_row[
                    feature
                ] = rolling_mean_4


            elif feature == "Rolling_Mean_12":

                future_row[
                    feature
                ] = rolling_mean_12


            elif feature == "Rolling_Std_4":

                future_row[
                    feature
                ] = rolling_std_4


            else:

                # Future external variables are carried forward
                # from the latest available observation.

                if feature in latest.index:

                    future_value = (
                        latest[feature]
                    )

                else:

                    future_value = 0


                future_row[
                    feature
                ] = future_value


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        future_X = pd.DataFrame(
            [future_row]
        )


        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for column in future_X.columns:

            if not pd.api.types.is_numeric_dtype(
                future_X[column]
            ):

                future_X[column] = pd.to_numeric(
                    future_X[column],
                    errors="coerce"
                )


        # ----------------------------------------------------
        # Handle invalid values
        # ----------------------------------------------------

        future_X = future_X.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )


        for column in future_X.columns:

            if future_X[column].isna().any():

                if column in X_train.columns:

                    fill_value = (
                        X_train[column]
                        .median()
                    )

                else:

                    fill_value = 0


                if pd.isna(fill_value):

                    fill_value = 0


                future_X[column] = (
                    future_X[column]
                    .fillna(fill_value)
                )


        # ----------------------------------------------------
        # Ensure correct feature order
        # ----------------------------------------------------

        future_X = (
            future_X[
                feature_columns
            ]
        )


        # ----------------------------------------------------
        # Generate forecast
        # ----------------------------------------------------

        forecast_value = (
            model
            .predict(
                future_X
            )[0]
        )


        # ----------------------------------------------------
        # Prevent negative sales
        # ----------------------------------------------------

        forecast_value = max(
            0,
            float(
                forecast_value
            )
        )


        # ----------------------------------------------------
        # Save forecast
        # ----------------------------------------------------

        future_forecasts.append({

            "Forecast_Week":
                future_date,

            "Store":
                store,

            "Dept":
                dept,

            "Predicted_Sales":
                forecast_value

        })


        # ----------------------------------------------------
        # Recursive update
        #
        # The prediction becomes historical input
        # for the next forecast period.
        # ----------------------------------------------------

        history[key].append(
            forecast_value
        )


# ============================================================
# 24. SAVE FUTURE FORECAST
# ============================================================

future_forecast_df = (
    pd.DataFrame(
        future_forecasts
    )
)


future_forecast_path = os.path.join(
    OUTPUT_DIR,
    "future_sales_forecast.csv"
)


future_forecast_df.to_csv(
    future_forecast_path,
    index=False
)


print(
    f"\nFuture forecast saved to: "
    f"{future_forecast_path}"
)


# ============================================================
# 25. AGGREGATED FUTURE FORECAST
# ============================================================

future_total_df = (

    future_forecast_df

    .groupby(
        "Forecast_Week",
        as_index=False
    )

    ["Predicted_Sales"]

    .sum()

)


future_total_path = os.path.join(
    OUTPUT_DIR,
    "future_total_sales_forecast.csv"
)


future_total_df.to_csv(
    future_total_path,
    index=False
)


# ============================================================
# 26. FUTURE FORECAST VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(

    future_total_df[
        "Forecast_Week"
    ],

    future_total_df[
        "Predicted_Sales"
    ],

    marker="o"

)


plt.title(
    "Walmart 12-Week Future Sales Forecast"
)


plt.xlabel(
    "Forecast Week"
)


plt.ylabel(
    "Predicted Sales"
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


future_plot_path = os.path.join(
    OUTPUT_DIR,
    "future_sales_forecast.png"
)


plt.savefig(
    future_plot_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    f"Future forecast plot saved to: "
    f"{future_plot_path}"
)


# ============================================================
# 27. FUTURE FORECAST SUMMARY
# ============================================================

total_forecast_sales = (
    future_forecast_df[
        "Predicted_Sales"
    ]
    .sum()
)


average_weekly_forecast = (
    future_total_df[
        "Predicted_Sales"
    ]
    .mean()
)


peak_forecast_row = (
    future_total_df
    .loc[
        future_total_df[
            "Predicted_Sales"
        ].idxmax()
    ]
)


lowest_forecast_row = (
    future_total_df
    .loc[
        future_total_df[
            "Predicted_Sales"
        ].idxmin()
    ]
)


print(
    "\nFuture Forecast Summary"
)


print(
    f"Total 12-week forecast : "
    f"{total_forecast_sales:,.2f}"
)


print(
    f"Average weekly forecast: "
    f"{average_weekly_forecast:,.2f}"
)


print(
    f"Peak forecast week     : "
    f"{peak_forecast_row['Forecast_Week'].date()} "
    f"({peak_forecast_row['Predicted_Sales']:,.2f})"
)


print(
    f"Lowest forecast week   : "
    f"{lowest_forecast_row['Forecast_Week'].date()} "
    f"({lowest_forecast_row['Predicted_Sales']:,.2f})"
)


# ============================================================
# 28. FINAL PROJECT SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "FORECASTING PIPELINE COMPLETED"
)


print(
    "=" * 70
)


print(
    "\nModel Performance"
)


print(
    f"Naive MAE  : "
    f"{naive_mae:,.2f}"
)


print(
    f"XGB MAE    : "
    f"{mae:,.2f}"
)


print(
    f"Naive RMSE : "
    f"{naive_rmse:,.2f}"
)


print(
    f"XGB RMSE   : "
    f"{rmse:,.2f}"
)


print(
    f"XGB R²     : "
    f"{r2:.4f}"
)


print(
    f"XGB WAPE   : "
    f"{wape:.2%}"
)


print(
    f"RMSE improvement vs baseline: "
    f"{rmse_improvement:.2%}"
)


print(
    f"MAE improvement vs baseline : "
    f"{mae_improvement:.2%}"
)


print(
    "\nForecast Error"
)


print(
    f"Mean Error   : "
    f"{mean_error:,.2f}"
)


print(
    f"Median Error : "
    f"{median_error:,.2f}"
)


print(
    f"Model tendency: "
    f"{model_tendency}"
)


print(
    "\nFuture Forecast"
)


print(
    f"Forecast horizon: "
    f"{FORECAST_HORIZON} weeks"
)


print(
    f"Total forecast : "
    f"{total_forecast_sales:,.2f}"
)


print(
    "\nGenerated Files:"
)


generated_files = [

    evaluation_path,

    prediction_path,

    store_error_path,

    dept_error_path,

    importance_path,

    model_path,

    forecast_plot_path,

    importance_plot_path,

    future_forecast_path,

    future_total_path,

    future_plot_path

]


for file_path in generated_files:

    print(
        f"- {file_path}"
    )


print(
    "\nPipeline status: "
    "Completed successfully."
)


# ============================================================
# END OF PIPELINE
# ============================================================