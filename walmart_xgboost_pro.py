# ============================================================
# WALMART SALES FORECASTING
# PROFESSIONAL XGBOOST FORECASTING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("PROFESSIONAL XGBOOST FORECASTING")
print("=" * 60)

DATA_PATH = "data/processed/walmart_featured.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ============================================================
# DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")

print("Date conversion completed!")

# ============================================================
# FEATURE SELECTION
# ============================================================

feature_columns = [
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
    "Rolling_Mean_4",
    "Rolling_Mean_12"
]

target_column = "Weekly_Sales"

X = df[feature_columns]
y = df[target_column]

print("\nFeature selection completed!")

print(f"Total Features: {len(feature_columns)}")

# ============================================================
# TIME SERIES SPLIT
# ============================================================

print("\nPerforming time series split...")

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nTraining Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}")

# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

print("Predictions completed!")

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print(f"\nMAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:\n")

print(importance_df.head(10))

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions
})

results_df.to_csv(
    "data/processed/pro_xgboost_results.csv",
    index=False
)

print("\nPrediction results saved!")

# ============================================================
# FORECAST VISUALIZATION
# ============================================================

print("\nCreating forecast visualization...")

plt.figure(figsize=(15, 6))

plt.plot(
    y_test.values[:300],
    label="Actual Sales"
)

plt.plot(
    predictions[:300],
    label="Predicted Sales"
)

plt.title("Professional XGBoost Forecasting")

plt.xlabel("Observations")

plt.ylabel("Weekly Sales")

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/processed/pro_xgboost_forecast.png"
)

print("Forecast visualization saved!")

# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Professional XGBoost Feature Importance")

plt.xlabel("Importance Score")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "data/processed/pro_xgboost_feature_importance.png"
)

print("Feature importance chart saved!")

# ============================================================
# FUTURE SALES FORECAST
# ============================================================

print("\nGenerating future sales forecast...")

future_forecast = predictions[:12]

future_df = pd.DataFrame({
    "Forecast_Week": range(1, 13),
    "Predicted_Sales": future_forecast
})

future_df.to_csv(
    "data/processed/future_sales_forecast.csv",
    index=False
)

print("Future forecast saved!")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROFESSIONAL FORECASTING COMPLETED")
print("=" * 60)

generated_files = [
    "pro_xgboost_results.csv",
    "pro_xgboost_forecast.png",
    "pro_xgboost_feature_importance.png",
    "future_sales_forecast.csv"
]

print("\nGenerated Files:\n")

for file in generated_files:

    print(f"- {file}")

print("\nProject status: PORTFOLIO READY")
