import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 50)
print("WALMART SALES FORECASTING")
print("=" * 50)

RAW_DIR = "data/raw"

train_path = os.path.join(RAW_DIR, "train.csv")
features_path = os.path.join(RAW_DIR, "features.csv")
stores_path = os.path.join(RAW_DIR, "stores.csv")

print("\nChecking dataset files...")

print(f"Train Path    : {train_path}")
print(f"Features Path : {features_path}")
print(f"Stores Path   : {stores_path}")

train = pd.read_csv(train_path)
features = pd.read_csv(features_path)
stores = pd.read_csv(stores_path)

print("\nDatasets loaded successfully!\n")

print("Train Shape    :", train.shape)
print("Features Shape :", features.shape)
print("Stores Shape   :", stores.shape)

# ============================================================
# STEP 2 — DATA TYPE CONVERSION
# ============================================================

print("\n" + "=" * 50)
print("STEP 2 — DATA TYPE CONVERSION")
print("=" * 50)

# Convert Date columns into datetime
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

print("\nDate columns converted successfully!")

# Convert IsHoliday into integer (0/1)
for dataframe in [train, features]:

    if dataframe["IsHoliday"].dtype == object:

        dataframe["IsHoliday"] = dataframe["IsHoliday"].map({
            "TRUE": 1,
            "FALSE": 0,
            True: 1,
            False: 0
        })

    dataframe["IsHoliday"] = dataframe["IsHoliday"].astype(int)

print("IsHoliday converted into numeric format!")

# Convert Store Type into string
stores["Type"] = stores["Type"].astype(str)

print("Store Type converted into string!")

# Display data types
print("\nTrain Data Types:")
print(train.dtypes)

print("\nFeatures Data Types:")
print(features.dtypes)

print("\nStores Data Types:")
print(stores.dtypes)

# ============================================================
# STEP 3 — MERGE DATASETS
# ============================================================

print("\n" + "=" * 50)
print("STEP 3 — MERGE DATASETS")
print("=" * 50)

# Merge train + features
print("\nMerging train and features datasets...")

df = train.merge(
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

print("Train + Features merged successfully!")

print("\nCurrent Shape:")
print(df.shape)

# Merge with stores dataset
print("\nMerging with stores dataset...")

df = df.merge(
    stores,
    on="Store",
    how="left"
)

print("Stores merged successfully!")

# Final shape
print("\nFinal Dataset Shape:")
print(df.shape)

# Display column names
print("\nFinal Columns:\n")
print(df.columns.tolist())

# Preview dataset
print("\nDataset Preview:\n")
print(df.head())

# ============================================================
# STEP 4 — HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 50)
print("STEP 4 — HANDLE MISSING VALUES")
print("=" * 50)

# Check missing values before cleaning
print("\nMissing Values Before Cleaning:\n")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if len(missing_before) == 0:

    print("No missing values found!")

else:

    for column, count in missing_before.items():

        percentage = (count / len(df)) * 100

        print(
            f"{column:<15} : "
            f"{count:>8,} rows "
            f"({percentage:.2f}%)"
        )

# ============================================================
# HANDLE TEMPERATURE & FUEL_PRICE
# ============================================================

print("\nCleaning Temperature and Fuel_Price...")

for column in ["Temperature", "Fuel_Price"]:

    df[column] = df.groupby("Store")[column].transform(
        lambda x: x.ffill().bfill()
    )

print("Temperature and Fuel_Price cleaned!")

# ============================================================
# HANDLE MARKDOWN COLUMNS
# ============================================================

print("\nCleaning MarkDown columns...")

markdown_columns = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5"
]

for column in markdown_columns:

    df[column] = (
        pd.to_numeric(df[column], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

print("MarkDown columns cleaned!")

# ============================================================
# HANDLE CPI & UNEMPLOYMENT
# ============================================================

print("\nCleaning CPI and Unemployment...")

for column in ["CPI", "Unemployment"]:

    df[column] = df.groupby("Store")[column].transform(
        lambda x: x.fillna(x.median())
    )

    df[column] = df[column].fillna(
        df[column].median()
    )

print("CPI and Unemployment cleaned!")

# ============================================================
# CHECK NEGATIVE WEEKLY SALES
# ============================================================

negative_sales = (df["Weekly_Sales"] < 0).sum()

print("\nChecking negative Weekly_Sales...")

if negative_sales > 0:

    print(
        f"Found {negative_sales:,} rows "
        f"with negative Weekly_Sales"
    )

else:

    print("No negative Weekly_Sales found!")

# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

print("\nFinal Missing Value Check:\n")

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]

if len(missing_after) == 0:

    print("All missing values cleaned successfully!")

else:

    for column, count in missing_after.items():

        percentage = (count / len(df)) * 100

        print(
            f"{column:<15} : "
            f"{count:>8,} rows "
            f"({percentage:.2f}%)"
        )

# ============================================================
# STEP 5 — REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 50)
print("STEP 5 — REMOVE DUPLICATES")
print("=" * 50)

# Check duplicate rows
duplicate_count = df.duplicated(
    subset=["Store", "Dept", "Date"]
).sum()

print(f"\nDuplicate rows found: {duplicate_count:,}")

# Remove duplicates if they exist
if duplicate_count > 0:

    df = df.drop_duplicates(
        subset=["Store", "Dept", "Date"],
        keep="first"
    )

    print("Duplicates removed successfully!")

else:

    print("No duplicate rows found!")

# Final dataset shape
print(f"\nCurrent dataset shape: {df.shape}")

# ============================================================
# STEP 6 — FEATURE ENCODING
# ============================================================

print("\n" + "=" * 50)
print("STEP 6 — FEATURE ENCODING")
print("=" * 50)

# Encode Store Type
print("\nEncoding Store Type...")

type_mapping = {
    "A": 0,
    "B": 1,
    "C": 2
}

df["Type_Code"] = df["Type"].map(type_mapping)

print("Store Type encoded successfully!")

# Check encoded values
print("\nStore Type Distribution:\n")
print(df["Type"].value_counts())

# Check encoded result
print("\nEncoded Type_Code Distribution:\n")
print(df["Type_Code"].value_counts())

# Store size statistics
print("\nStore Size Statistics:\n")

store_stats = (
    df.groupby("Type")["Size"]
    .agg(["min", "max", "mean"])
    .round(0)
)

print(store_stats)

# ============================================================
# FINAL DATASET PREVIEW
# ============================================================

print("\nFinal Dataset Preview:\n")

print(df.head())

# Final shape confirmation
print("\nFinal Dataset Shape:")
print(df.shape)

# ============================================================
# STEP 7 — SORT & SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 50)
print("STEP 7 — SAVE CLEAN DATASET")
print("=" * 50)

# Sort dataset
print("\nSorting dataset...")

df = df.sort_values(
    by=["Store", "Dept", "Date"]
).reset_index(drop=True)

print("Dataset sorted successfully!")

# ============================================================
# CREATE PROCESSED DIRECTORY
# ============================================================

PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# SAVE CLEAN DATASET
# ============================================================

output_path = os.path.join(
    PROCESSED_DIR,
    "walmart_clean.csv"
)

print("\nSaving clean dataset...")

df.to_csv(output_path, index=False)

print("Dataset saved successfully!")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)

print(f"\nDataset Shape : {df.shape}")

print(f"\nTotal Stores  : {df['Store'].nunique():,}")
print(f"Total Depts   : {df['Dept'].nunique():,}")
print(f"Total Weeks   : {df['Date'].nunique():,}")

holiday_weeks = df[df["IsHoliday"] == 1]["Date"].nunique()

print(f"Holiday Weeks : {holiday_weeks:,}")

# Weekly Sales Statistics
sales_stats = df["Weekly_Sales"].describe()

print("\nWeekly_Sales Statistics:\n")

print(f"Minimum : ${sales_stats['min']:,.2f}")
print(f"Median  : ${sales_stats['50%']:,.2f}")
print(f"Mean    : ${sales_stats['mean']:,.2f}")
print(f"Maximum : ${sales_stats['max']:,.2f}")

# ============================================================
# FINAL COLUMN CHECK
# ============================================================

print("\nFinal Columns:\n")

for index, column in enumerate(df.columns, start=1):

    dtype = df[column].dtype
    missing = df[column].isnull().sum()

    print(
        f"{index:>2}. "
        f"{column:<15} "
        f"| Type: {str(dtype):<15} "
        f"| Missing: {missing}"
    )

print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 50)

print(f"\nClean dataset saved at:\n{output_path}")
