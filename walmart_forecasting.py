# ============================================================
# WALMART SALES FORECASTING
# MACHINE LEARNING FORECASTING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# LOAD FEATURED DATASET
# ============================================================

print("=" * 60)
print("WALMART SALES — MACHINE LEARNING FORECASTING")
print("=" * 60)

DATA_PATH = "data/processed/walmart_featured.csv"

print("\nLoading featured dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ============================================================
# FEATURE SELECTION
# ============================================================

print("\nSelecting features...")

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

print("Feature selection completed!")

print(f"\nTotal Features : {len(feature_columns)}")

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\nSplitting train and test dataset...")

# Time-series style split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nTraining Data : {X_train.shape}")
print(f"Testing Data  : {X_test.shape}")

# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# PREDICTION
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

print("\nCalculating feature importance...")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Feature Importance:\n")

print(importance_df.head(10))

# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving prediction results...")

results_df = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions
})

results_df.to_csv(
    "data/processed/forecast_results.csv",
    index=False
)

print("Prediction results saved!")

# ============================================================
# FORECAST VISUALIZATION
# ============================================================

print("\nCreating forecast visualization...")

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.values[:200],
    label="Actual Sales"
)

plt.plot(
    predictions[:200],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Observations")

plt.ylabel("Weekly Sales")

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/processed/forecast_vs_actual.png"
)

print("Forecast visualization saved!")

# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================

print("\nCreating feature importance chart...")

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Feature Importance")

plt.xlabel("Importance Score")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png"
)

print("Feature importance chart saved!")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:\n")

generated_files = [
    "forecast_results.csv",
    "forecast_vs_actual.png",
    "feature_importance.png"
]

for file in generated_files:

    print(f"- {file}")
