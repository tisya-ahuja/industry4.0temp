"""
ONLINE RETAIL: COMPLETE DATA CLEANING + EDA
===========================================

Input:
    Online Retail.xlsx

Expected raw data:
    541,909 rows x 8 columns

Expected cleaned transaction data:
    524,878 rows x 12 columns

Cleaning policy:
    1. Remove exact duplicate rows.
    2. Remove rows with missing Description.
    3. Remove cancellation invoices (InvoiceNo begins with C).
    4. Remove rows with Quantity <= 0 from the SALES dataset.
    5. Remove rows with UnitPrice <= 0.
    6. Keep missing CustomerID values; unknown-customer transactions can
       still be valid sales. Customer-level analysis excludes those rows.

Important:
    Negative quantities / cancellation invoices may be valuable for a
    separate returns analysis. This script saves the original data untouched
    and creates a sales-only analytical dataset.
"""

# ============================================================
# STEP 0: IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Make charts easier to read.
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

# ============================================================
# STEP 1: FILE PATHS AND OUTPUT FOLDER
# ============================================================

# Change this path if your Excel file is stored elsewhere.
FILE_PATH = "Online Retail.xlsx"

# All output files and charts will be saved in this folder.
OUTPUT_DIR = "tisya/online_retail_eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 2: LOAD THE DATA
# ============================================================

# Read the specified worksheet.
df = pd.read_excel(FILE_PATH, sheet_name="Online Retail")

# Keep an untouched copy for audit / comparison.
df_raw = df.copy()

print("\nSTEP 2: DATA LOADED")
print("=" * 60)
print("Raw shape:", df.shape)
print("Expected shape: (541909, 8)")
print(df.head())
print(df.tail())

# Stop early if the workbook is not the expected dataset.
expected_columns = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]
assert list(df.columns) == expected_columns, (
    "Unexpected columns. Check that the correct worksheet was loaded."
)
assert df.shape == (541909, 8), (
    "The input row count differs from the inspected workbook. "
    "Review the cleaning-stage counts before relying on expected values."
)

# ============================================================
# STEP 3: INITIAL DATA UNDERSTANDING
# ============================================================

print("\nSTEP 3: INITIAL DATA UNDERSTANDING")
print("=" * 60)

# Dimensions, column names, data types and memory usage.
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataFrame information:")
df.info()

print("\nFirst five rows:")
print(df.head())

print("\nRandom five rows:")
print(df.sample(5, random_state=42))

print("\nNumerical summary:")
print(df.describe())

print("\nCategorical summary:")
print(df.describe(include="object"))

# ============================================================
# STEP 4: MISSING-VALUE ANALYSIS
# ============================================================

print("\nSTEP 4: MISSING-VALUE ANALYSIS")
print("=" * 60)

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percent
}).sort_values("Missing_Count", ascending=False)

print(missing_report)

# Visualize missing values by column.
plt.figure(figsize=(10, 5))
sns.barplot(
    x=missing_report.index,
    y=missing_report["Missing_Count"]
)
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_missing_values.png"),
    bbox_inches="tight"
)
plt.show()

# Interpretation:
# Description is required to identify the product, so rows without it
# are removed in the cleaning pipeline.
#
# CustomerID is NOT required to calculate transaction revenue. We retain
# missing IDs in the main cleaned sales dataset and exclude them only from
# customer-level analyses.

# ============================================================
# STEP 5: DUPLICATE ANALYSIS
# ============================================================

print("\nSTEP 5: DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()
print("Exact duplicate rows:", duplicate_count)
print("Expected duplicate rows: 5268")

# Display duplicate examples before removing them.
print("\nDuplicate examples:")
print(df[df.duplicated(keep=False)].head(10))

# ============================================================
# STEP 6: OUTLIER / BUSINESS-LOGIC CHECKS
# ============================================================

print("\nSTEP 6: BUSINESS-LOGIC CHECKS")
print("=" * 60)

# Negative quantity can mean a return or reversal.
print("Negative Quantity rows:", (df["Quantity"] < 0).sum())
print("Zero Quantity rows:", (df["Quantity"] == 0).sum())
print("Positive Quantity rows:", (df["Quantity"] > 0).sum())

# Zero / negative prices should not be treated as ordinary paid sales.
print("Zero UnitPrice rows:", (df["UnitPrice"] == 0).sum())
print("Negative UnitPrice rows:", (df["UnitPrice"] < 0).sum())

# Cancellation invoices have an invoice number beginning with C.
is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
print("Cancellation invoice rows:", is_cancellation.sum())

# Check the date range and cardinalities.
print("Minimum invoice date:", df["InvoiceDate"].min())
print("Maximum invoice date:", df["InvoiceDate"].max())
print("Unique invoices:", df["InvoiceNo"].nunique())
print("Unique customers (excluding missing):", df["CustomerID"].nunique())
print("Unique countries:", df["Country"].nunique())

# Inspect the distributions before deciding how to handle extreme values.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(y=df["Quantity"], ax=axes[0])
axes[0].set_title("Quantity: Boxplot (Raw Data)")
sns.boxplot(y=df["UnitPrice"], ax=axes[1])
axes[1].set_title("UnitPrice: Boxplot (Raw Data)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_raw_outlier_boxplots.png"),
    bbox_inches="tight"
)
plt.show()

# Do not automatically remove high positive quantities or prices as
# statistical outliers. They may be genuine bulk orders or expensive items.
# We remove only records that fail the stated sales-data business rules.

# ============================================================
# STEP 7: CLEANING PIPELINE
# ============================================================

print("\nSTEP 7: CLEANING PIPELINE")
print("=" * 60)

clean = df.copy()
cleaning_log = []

def record_step(step_name, before_rows, after_df):
    """Save row/column counts and the number of rows removed at a step."""
    after_rows = len(after_df)
    cleaning_log.append({
        "Step": step_name,
        "Rows_Before": before_rows,
        "Rows_After": after_rows,
        "Rows_Removed": before_rows - after_rows,
        "Columns_After": after_df.shape[1]
    })
    print(
        f"{step_name}: {after_rows:,} rows x {after_df.shape[1]} columns "
        f"({before_rows - after_rows:,} rows removed)"
    )

# 7A. Remove exact duplicate rows.
before = len(clean)
clean = clean.drop_duplicates().copy()
record_step("Remove exact duplicates", before, clean)
assert clean.shape == (536641, 8)

# 7B. Remove rows with missing product descriptions.
before = len(clean)
clean = clean.dropna(subset=["Description"]).copy()
record_step("Remove missing Description", before, clean)
assert clean.shape == (535187, 8)

# 7C. Remove cancellation invoice records.
# Cancellation invoice numbers start with C.
before = len(clean)
cancel_mask = clean["InvoiceNo"].astype(str).str.startswith("C")
clean = clean.loc[~cancel_mask].copy()
record_step("Remove cancellation invoices", before, clean)
assert clean.shape == (525936, 8)

# 7D. Remove zero or negative quantities from the SALES dataset.
# Returns are excluded here because this dataset is intended to represent
# completed positive-quantity sales. Keep the raw data for returns analysis.
before = len(clean)
clean = clean.loc[clean["Quantity"] > 0].copy()
record_step("Keep positive quantities", before, clean)
assert clean.shape == (525462, 8)

# 7E. Remove zero or negative unit prices.
before = len(clean)
clean = clean.loc[clean["UnitPrice"] > 0].copy()
record_step("Keep positive unit prices", before, clean)
assert clean.shape == (524878, 8)

# Show the complete audit table.
cleaning_report = pd.DataFrame(cleaning_log)
print("\nCLEANING SUMMARY:")
print(cleaning_report.to_string(index=False))

# ============================================================
# STEP 8: DATA TYPE AND TEXT CLEANUP
# ============================================================

print("\nSTEP 8: DATA TYPE AND TEXT CLEANUP")
print("=" * 60)

# Invoice numbers and stock codes are identifiers, not numeric quantities.
# String types preserve leading zeros and alphanumeric codes.
clean["InvoiceNo"] = clean["InvoiceNo"].astype("string")
clean["StockCode"] = clean["StockCode"].astype("string")

# Clean whitespace from text fields.
clean["Description"] = clean["Description"].astype("string").str.strip()
clean["Country"] = clean["Country"].astype("string").str.strip()

# CustomerID is an identifier. Nullable Int64 allows missing values without
# converting the entire column to float.
clean["CustomerID"] = clean["CustomerID"].astype("Int64")

# Ensure date and numerical fields have the correct types.
clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], errors="coerce")
clean["Quantity"] = pd.to_numeric(clean["Quantity"], errors="coerce")
clean["UnitPrice"] = pd.to_numeric(clean["UnitPrice"], errors="coerce")

print(clean.dtypes)
print("Shape after type cleanup:", clean.shape)

# These conversions should not create missing values in the inspected file
# except CustomerID, which was already missing in the source.
assert clean["InvoiceDate"].notna().all()
assert clean["Quantity"].notna().all()
assert clean["UnitPrice"].notna().all()

# ============================================================
# STEP 9: FEATURE ENGINEERING
# ============================================================

print("\nSTEP 9: FEATURE ENGINEERING")
print("=" * 60)

# Revenue for each line item = number of units x unit price.
clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]

# Month is useful for monthly sales trends.
clean["InvoiceMonth"] = clean["InvoiceDate"].dt.to_period("M").astype(str)

# Year and hour allow time-based analysis.
clean["InvoiceYear"] = clean["InvoiceDate"].dt.year
clean["InvoiceHour"] = clean["InvoiceDate"].dt.hour

print("Shape after feature engineering:", clean.shape)
print("Expected final shape: (524878, 12)")
print("\nFinal columns:")
print(clean.columns.tolist())

assert clean.shape == (524878, 12)

# ============================================================
# STEP 10: FINAL DATA QUALITY VALIDATION
# ============================================================

print("\nSTEP 10: FINAL DATA QUALITY VALIDATION")
print("=" * 60)

print("Final shape:", clean.shape)
print("\nMissing values:")
print(clean.isna().sum())

print("\nDuplicate rows:", clean.duplicated().sum())
print("Negative quantities:", (clean["Quantity"] < 0).sum())
print("Zero quantities:", (clean["Quantity"] == 0).sum())
print("Non-positive prices:", (clean["UnitPrice"] <= 0).sum())
print("Cancellation invoices:",
      clean["InvoiceNo"].str.startswith("C").sum())

print("\nDate range:")
print(clean["InvoiceDate"].min(), "to", clean["InvoiceDate"].max())

# Expected: only CustomerID remains missing.
assert clean.duplicated().sum() == 0
assert (clean["Quantity"] > 0).all()
assert (clean["UnitPrice"] > 0).all()
assert clean["Description"].notna().all()
assert clean["CustomerID"].isna().sum() == 132186

# ============================================================
# STEP 11: UNIVARIATE EDA - NUMERICAL VARIABLES
# ============================================================

print("\nSTEP 11: NUMERICAL EDA")
print("=" * 60)

numerical_columns = ["Quantity", "UnitPrice", "Revenue"]

print(clean[numerical_columns].describe())

# Histograms show the shape and skewness of the distributions.
for column in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(clean[column], bins=60, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"03_distribution_{column}.png"),
        bbox_inches="tight"
    )
    plt.show()

# Boxplots are useful for seeing spread and extreme values.
for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=clean[column])
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"04_boxplot_{column}.png"),
        bbox_inches="tight"
    )
    plt.show()

# ============================================================
# STEP 12: UNIVARIATE EDA - CATEGORICAL VARIABLES
# ============================================================

print("\nSTEP 12: CATEGORICAL EDA")
print("=" * 60)

print("Number of unique products:", clean["StockCode"].nunique())
print("Number of unique descriptions:", clean["Description"].nunique())
print("Number of unique invoices:", clean["InvoiceNo"].nunique())
print("Number of countries:", clean["Country"].nunique())
print("Number of identified customers:", clean["CustomerID"].nunique())

# Top 15 countries by number of transaction lines.
country_frequency = clean["Country"].value_counts().head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=country_frequency.values, y=country_frequency.index)
plt.title("Top 15 Countries by Transaction-Line Count")
plt.xlabel("Number of transaction lines")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "05_country_frequency.png"),
    bbox_inches="tight"
)
plt.show()

# ============================================================
# STEP 13: REVENUE AND SALES KPIs
# ============================================================

print("\nSTEP 13: SALES KPIs")
print("=" * 60)

total_revenue = clean["Revenue"].sum()
total_units_sold = clean["Quantity"].sum()
total_orders = clean["InvoiceNo"].nunique()
identified_customers = clean["CustomerID"].nunique()
average_order_value = total_revenue / total_orders

print(f"Total revenue: {total_revenue:,.2f}")
print(f"Total units sold: {total_units_sold:,.0f}")
print(f"Unique invoices/orders: {total_orders:,}")
print(f"Identified customers: {identified_customers:,}")
print(f"Average revenue per invoice: {average_order_value:,.2f}")

# Important: this is average revenue per distinct invoice, not per customer.
# It uses all cleaned invoices, including invoices without CustomerID.

# ============================================================
# STEP 14: MONTHLY SALES ANALYSIS
# ============================================================

print("\nSTEP 14: MONTHLY SALES ANALYSIS")
print("=" * 60)

monthly_sales = (
    clean.groupby("InvoiceMonth")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units_Sold=("Quantity", "sum")
    )
    .sort_index()
)

print(monthly_sales)

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=monthly_sales,
    x=monthly_sales.index,
    y="Revenue",
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "06_monthly_revenue.png"),
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(
    x=monthly_sales.index,
    y=monthly_sales["Orders"]
)
plt.title("Number of Unique Invoices by Month")
plt.xlabel("Month")
plt.ylabel("Number of invoices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "07_monthly_orders.png"),
    bbox_inches="tight"
)
plt.show()

# Note: December 2011 ends on 9 December in this dataset, so it is an
# incomplete month and should not be compared directly with full months.

# ============================================================
# STEP 15: COUNTRY-WISE SALES ANALYSIS
# ============================================================

print("\nSTEP 15: COUNTRY-WISE SALES ANALYSIS")
print("=" * 60)

country_sales = (
    clean.groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units_Sold=("Quantity", "sum")
    )
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 countries by revenue:")
print(country_sales.head(10))

top10_countries = country_sales.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=top10_countries["Revenue"],
    y=top10_countries.index
)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "08_top_countries_revenue.png"),
    bbox_inches="tight"
)
plt.show()

# ============================================================
# STEP 16: PRODUCT-WISE SALES ANALYSIS
# ============================================================

print("\nSTEP 16: PRODUCT-WISE SALES ANALYSIS")
print("=" * 60)

product_sales = (
    clean.groupby(["StockCode", "Description"])
    .agg(
        Units_Sold=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Invoice_Count=("InvoiceNo", "nunique")
    )
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 products by revenue:")
print(product_sales.head(10))

print("\nTop 10 products by quantity sold:")
print(
    product_sales.sort_values("Units_Sold", ascending=False).head(10)
)

top_products_revenue = product_sales.head(10)
plt.figure(figsize=(12, 7))
sns.barplot(
    x=top_products_revenue["Revenue"],
    y=top_products_revenue.index.get_level_values("Description")
)
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "09_top_products_revenue.png"),
    bbox_inches="tight"
)
plt.show()

# ============================================================
# STEP 17: CUSTOMER ANALYSIS
# ============================================================

print("\nSTEP 17: CUSTOMER ANALYSIS")
print("=" * 60)

# CustomerID is missing for some valid transactions. Exclude those rows
# only for analyses that require an identified customer.
customer_data = clean.dropna(subset=["CustomerID"]).copy()

print("Rows available for customer analysis:", len(customer_data))
print("Unique identified customers:", customer_data["CustomerID"].nunique())

customer_sales = (
    customer_data.groupby("CustomerID")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units_Sold=("Quantity", "sum"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max")
    )
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 customers by revenue:")
print(customer_sales.head(10))

# Average revenue per identified customer.
print(
    "Average revenue per identified customer:",
    customer_sales["Revenue"].mean()
)

top_customers = customer_sales.head(10)
plt.figure(figsize=(12, 6))
sns.barplot(
    x=top_customers["Revenue"],
    y=top_customers.index.astype(str)
)
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "10_top_customers.png"),
    bbox_inches="tight"
)
plt.show()

# ============================================================
# STEP 18: HOURLY SALES ANALYSIS
# ============================================================

print("\nSTEP 18: HOURLY SALES ANALYSIS")
print("=" * 60)

hourly_sales = (
    clean.groupby("InvoiceHour")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units_Sold=("Quantity", "sum")
    )
    .sort_index()
)

print(hourly_sales)

plt.figure(figsize=(12, 5))
sns.barplot(
    x=hourly_sales.index,
    y=hourly_sales["Orders"]
)
plt.title("Orders by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Unique invoices")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "11_hourly_orders.png"),
    bbox_inches="tight"
)
plt.show()

# ============================================================
# STEP 19: CORRELATION ANALYSIS
# ============================================================

print("\nSTEP 19: CORRELATION ANALYSIS")
print("=" * 60)

correlation = clean[
    ["Quantity", "UnitPrice", "Revenue"]
].corr()

print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "12_correlation_heatmap.png"),
    bbox_inches="tight"
)
plt.show()

# Correlation is descriptive, not proof of causation.
# Revenue is mathematically derived from Quantity * UnitPrice, so its
# correlations with those variables are partly mechanical.

# ============================================================
# STEP 20: OPTIONAL RETURNS / CANCELLATIONS DATASET
# ============================================================

print("\nSTEP 20: OPTIONAL RETURNS / CANCELLATIONS AUDIT")
print("=" * 60)

# Preserve potentially useful reversal records separately rather than
# mixing them into the positive-sales dataset.
# This is an audit extract, not the final sales dataset.
returns_and_cancellations = df_raw[
    (df_raw["Quantity"] < 0)
    | (df_raw["InvoiceNo"].astype(str).str.startswith("C"))
].copy()

print(
    "Raw rows flagged as negative quantity or cancellation:",
    len(returns_and_cancellations)
)
print(
    "Negative-quantity rows:",
    (returns_and_cancellations["Quantity"] < 0).sum()
)
print(
    "Cancellation-invoice rows:",
    returns_and_cancellations["InvoiceNo"]
    .astype(str).str.startswith("C").sum()
)

# ============================================================
# STEP 21: EXPORT FINAL FILES
# ============================================================

print("\nSTEP 21: EXPORTING FILES")
print("=" * 60)

# Export the main cleaned sales data.
clean.to_csv(
    os.path.join(OUTPUT_DIR, "online_retail_cleaned.csv"),
    index=False
)

# Export useful EDA tables.
cleaning_report.to_csv(
    os.path.join(OUTPUT_DIR, "cleaning_summary.csv"),
    index=False
)
missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values_report.csv")
)
monthly_sales.to_csv(
    os.path.join(OUTPUT_DIR, "monthly_sales.csv")
)
country_sales.to_csv(
    os.path.join(OUTPUT_DIR, "country_sales.csv")
)
product_sales.to_csv(
    os.path.join(OUTPUT_DIR, "product_sales.csv")
)
customer_sales.to_csv(
    os.path.join(OUTPUT_DIR, "customer_sales.csv")
)
returns_and_cancellations.to_csv(
    os.path.join(OUTPUT_DIR, "returns_cancellations_audit.csv"),
    index=False
)

# Save the final cleaned data in Excel format too.
clean.to_excel(
    os.path.join(OUTPUT_DIR, "online_retail_cleaned.xlsx"),
    index=False
)

print("All output files saved in:", OUTPUT_DIR)

# ============================================================
# STEP 22: FINAL SUMMARY
# ============================================================

print("\nFINAL SUMMARY")
print("=" * 60)
print("Original shape:", df_raw.shape)
print("Final cleaned sales shape:", clean.shape)
print("Rows removed:", len(df_raw) - len(clean))
print("Columns added:", clean.shape[1] - df_raw.shape[1])
print("Final revenue:", round(clean["Revenue"].sum(), 2))
print("Final units sold:", int(clean["Quantity"].sum()))
print("Final unique invoices:", clean["InvoiceNo"].nunique())
print("Final identified customers:", clean["CustomerID"].nunique())
print("Final countries:", clean["Country"].nunique())
print("Remaining missing CustomerIDs:", clean["CustomerID"].isna().sum())

assert clean.shape == (524878, 12)
print("\nSUCCESS: Cleaning and EDA completed.")
