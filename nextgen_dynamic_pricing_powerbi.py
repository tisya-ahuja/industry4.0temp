#!/usr/bin/env python
"""
NextGen | Industry 4.0
Dynamic Pricing Strategy for Revenue Optimization in Retail Sales

One-run analysis pipeline for Power BI preparation.

INPUT
-----
The original UCI Online Retail Excel file, normally:
    Online Retail(1).xlsx

OUTPUT
------
A folder (default: nextgen_powerbi_data/) containing:
  01_FactSales.csv
  02_ProductMonth.csv
  03_Product.csv
  04_Calendar.csv
  05_Month.csv
  06_ElasticityResults.csv
  07_MonthlySummary.csv
  08_ProductSummary_ABC.csv
  09_PeakOffPeakElasticity.csv
  10_PricingScenarios.csv
  11_ProductRecommendations.csv
  12_DataQualityAudit.csv
  PowerBI_DAX_Measures.txt
  README_PowerBI_Model.txt
  NextGen_Analysis_Results.xlsx

INSTALL
-------
pip install pandas numpy openpyxl

RUN
---
python nextgen_dynamic_pricing_powerbi.py
python nextgen_dynamic_pricing_powerbi.py --input "Online Retail(1).xlsx"
python nextgen_dynamic_pricing_powerbi.py --input "Online Retail(1).xlsx" --output-dir "PowerBI_Ready"

IMPORTANT INTERPRETATION
------------------------
* The regression is an observational log-log association, not a causal
  estimate of what a price change will do.
* No product cost data are available. Outputs concern revenue, not profit.
* Product-month regression has a short time series (at most about 13 months).
  Treat estimates as exploratory and inspect reliability fields.
* December 2011 is incomplete in the source dataset (data ends 9 Dec 2011).
* Scenario outputs are illustrative what-if calculations, not forecasts.
* The conservative scenario screen is configurable below.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURATION — change these only if your approved project methodology
# requires different criteria. Record any changes in your report.
# ---------------------------------------------------------------------------
COUNTRY_FILTER = "United Kingdom"
MIN_DISTINCT_TRANSACTION_PRICES = 3
MIN_TRANSACTION_LINES = 50
MIN_MONTHLY_PANEL_POINTS = 5
MIN_R_SQUARED_FOR_SCENARIO = 0.30
MAX_ABS_ELASTICITY_FOR_SCENARIO = 10.0
PEAK_MONTHS = {11, 12}
PRICE_CHANGE_SCENARIOS = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]

# Codes used for postage, fees, adjustments or other non-merchandise entries.
NON_PRODUCT_CODES = {
    "POST", "C2", "M", "BANK", "PADS", "DOT",
    "CRUK", "D", "S", "AMAZONFEE"
}

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]


def log(message: str) -> None:
    print(message, flush=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load source workbook or CSV and standardise column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Place the original Online Retail.xlsx in this folder or pass --input."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, encoding="ISO-8859-1")
    else:
        raise ValueError("Input must be .xlsx, .xls or .csv")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from input: {missing}")

    return df


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the team's current proposal/script cleaning rules:
      - parse InvoiceDate
      - remove cancellation invoices (InvoiceNo begins with C)
      - retain positive Quantity and UnitPrice
      - remove missing CustomerID
      - restrict to UK
      - remove specified non-product StockCodes
    """
    df = raw.copy()
    raw_rows = len(df)

    # Preserve a step-by-step row audit; each stage count is cumulative.
    audit = []
    audit.append(("Raw input rows", len(df)))

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    bad_dates = int(df["InvoiceDate"].isna().sum())
    df = df.dropna(subset=["InvoiceDate"])
    audit.append(("After removing invalid/missing dates", len(df)))

    invoice_text = df["InvoiceNo"].astype(str).str.strip()
    df = df[~invoice_text.str.upper().str.startswith("C")]
    audit.append(("After removing cancellation invoices", len(df)))

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    audit.append(("After retaining positive quantity and price", len(df)))

    df = df.dropna(subset=["CustomerID"])
    audit.append(("After removing missing CustomerID", len(df)))

    df["Country"] = df["Country"].astype(str).str.strip()
    df = df[df["Country"].str.casefold() == COUNTRY_FILTER.casefold()]
    audit.append((f"After restricting to {COUNTRY_FILTER}", len(df)))

    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df = df[~df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)]
    audit.append(("After removing non-product StockCodes", len(df)))

    # Remove rows where key numeric fields became invalid.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Quantity", "UnitPrice", "InvoiceDate", "StockCode"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)

    audit.append(("Final cleaned rows", len(df)))
    audit_df = pd.DataFrame(audit, columns=["CleaningStep", "RowsRemaining"])
    audit_df["RowsRemovedAtStep"] = audit_df["RowsRemaining"].shift(1) - audit_df["RowsRemaining"]
    audit_df.loc[0, "RowsRemovedAtStep"] = 0
    audit_df["RowsRemovedAtStep"] = audit_df["RowsRemovedAtStep"].fillna(0).astype(int)
    audit_df["RowsRemaining"] = audit_df["RowsRemaining"].astype(int)

    if df.empty:
        raise ValueError("No rows remain after cleaning. Check the input and cleaning rules.")

    log(f"Raw rows: {raw_rows:,}")
    log(f"Cleaned UK rows: {len(df):,}")
    log(f"Rows removed: {raw_rows - len(df):,}")
    return df, audit_df


def build_dimensions_and_facts(df: pd.DataFrame):
    """Build Power BI-ready transaction, product, calendar and monthly tables."""
    sales = df.copy()
    sales["Date"] = sales["InvoiceDate"].dt.normalize()
    sales["Year"] = sales["InvoiceDate"].dt.year
    sales["MonthNumber"] = sales["InvoiceDate"].dt.month
    sales["MonthStart"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    sales["YearMonth"] = sales["InvoiceDate"].dt.strftime("%Y-%m")
    sales["MonthName"] = sales["InvoiceDate"].dt.strftime("%b")
    sales["Quarter"] = "Q" + sales["InvoiceDate"].dt.quarter.astype(str)
    sales["DayOfWeek"] = sales["InvoiceDate"].dt.day_name()

    # Product dimension: one row per StockCode.
    product = (
        sales.sort_values("InvoiceDate")
        .groupby("StockCode", as_index=False)
        .agg(
            Description=("Description", "first"),
            FirstSaleDate=("InvoiceDate", "min"),
            LastSaleDate=("InvoiceDate", "max"),
            TransactionLines=("InvoiceNo", "count"),
            TotalQuantity=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
            DistinctTransactionPrices=("UnitPrice", "nunique"),
        )
    )

    # Product-month panel. Average price is revenue / quantity (weighted).
    panel = (
        sales.groupby(["StockCode", "MonthStart"], as_index=False)
        .agg(
            Description=("Description", "first"),
            TotalQuantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            TransactionLines=("InvoiceNo", "count"),
            InvoiceCount=("InvoiceNo", "nunique"),
            CustomerCount=("CustomerID", "nunique"),
            MinTransactionPrice=("UnitPrice", "min"),
            MaxTransactionPrice=("UnitPrice", "max"),
        )
    )
    panel["AverageSellingPrice"] = panel["Revenue"] / panel["TotalQuantity"]
    panel["Year"] = panel["MonthStart"].dt.year
    panel["MonthNumber"] = panel["MonthStart"].dt.month
    panel["YearMonth"] = panel["MonthStart"].dt.strftime("%Y-%m")
    panel["MonthName"] = panel["MonthStart"].dt.strftime("%b")
    panel["IsPeakMonth"] = panel["MonthNumber"].isin(PEAK_MONTHS)

    # Calendar dimension covering every date from min to max transaction date.
    dates = pd.date_range(sales["Date"].min(), sales["Date"].max(), freq="D")
    calendar = pd.DataFrame({"Date": dates})
    calendar["Year"] = calendar["Date"].dt.year
    calendar["Quarter"] = "Q" + calendar["Date"].dt.quarter.astype(str)
    calendar["MonthNumber"] = calendar["Date"].dt.month
    calendar["MonthName"] = calendar["Date"].dt.strftime("%b")
    calendar["YearMonth"] = calendar["Date"].dt.strftime("%Y-%m")
    calendar["MonthStart"] = calendar["Date"].dt.to_period("M").dt.to_timestamp()
    calendar["Day"] = calendar["Date"].dt.day
    calendar["DayOfWeek"] = calendar["Date"].dt.day_name()
    calendar["IsWeekend"] = calendar["Date"].dt.dayofweek >= 5
    calendar["IsPeakMonth"] = calendar["MonthNumber"].isin(PEAK_MONTHS)

    # Monthly summary from cleaned transaction data.
    monthly = (
        sales.groupby(["MonthStart", "YearMonth", "Year", "MonthNumber"], as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            InvoiceCount=("InvoiceNo", "nunique"),
            CustomerCount=("CustomerID", "nunique"),
            ProductCount=("StockCode", "nunique"),
            TransactionLines=("InvoiceNo", "count"),
        )
        .sort_values("MonthStart")
    )
    monthly["AverageSellingPrice"] = monthly["Revenue"] / monthly["Quantity"]
    monthly["MonthName"] = monthly["MonthStart"].dt.strftime("%b")
    monthly["Quarter"] = "Q" + monthly["MonthStart"].dt.quarter.astype(str)
    monthly["IsPeakMonth"] = monthly["MonthNumber"].isin(PEAK_MONTHS)
    monthly["IsPartialMonth"] = (
        (monthly["MonthStart"].dt.to_period("M") == sales["InvoiceDate"].max().to_period("M"))
        & (sales["InvoiceDate"].max().day < monthly["MonthStart"].dt.days_in_month)
    )

    month_dimension = monthly[[
        "MonthStart", "Year", "MonthNumber", "MonthName", "YearMonth",
        "Quarter", "IsPeakMonth", "IsPartialMonth"
    ]].drop_duplicates("MonthStart").sort_values("MonthStart")

    return sales, product, panel, calendar, month_dimension, monthly


def abc_analysis(product: pd.DataFrame) -> pd.DataFrame:
    """ABC/Pareto classification based on cumulative product revenue."""
    out = product.copy().sort_values("TotalRevenue", ascending=False)
    total = out["TotalRevenue"].sum()
    out["RevenueShare"] = out["TotalRevenue"] / total if total else 0
    out["CumulativeRevenueShare"] = out["RevenueShare"].cumsum()
    # Assign category based on cumulative share, with first product included in A.
    out["ABCCategory"] = np.select(
        [
            out["CumulativeRevenueShare"] <= 0.80,
            out["CumulativeRevenueShare"] <= 0.95,
        ],
        ["A", "B"],
        default="C",
    )
    # Ensure the product that crosses the 80% threshold remains in A.
    crossed_80 = out.index[out["CumulativeRevenueShare"] > 0.80]
    if len(crossed_80):
        first_cross = crossed_80[0]
        previous = out.index.get_loc(first_cross) - 1
        if previous >= 0:
            out.iloc[previous, out.columns.get_loc("ABCCategory")] = "A"
    return out.sort_values("StockCode")


def fit_log_log(panel_part: pd.DataFrame, min_points: int):
    """
    OLS fit: ln(quantity) = intercept + elasticity * ln(avg price).
    Returns coefficient and transparent diagnostics. No p-value is reported.
    """
    p = panel_part.dropna(subset=["AverageSellingPrice", "TotalQuantity"]).copy()
    p = p[(p["AverageSellingPrice"] > 0) & (p["TotalQuantity"] > 0)]

    if len(p) < min_points or p["AverageSellingPrice"].nunique() < 2:
        return None

    x = np.log(p["AverageSellingPrice"].to_numpy(dtype=float))
    y = np.log(p["TotalQuantity"].to_numpy(dtype=float))

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.var(x) == 0 or np.var(y) == 0:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_log = float(np.sqrt(np.mean(residuals ** 2)))
    residual_sd = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else np.nan

    # OLS slope standard error (descriptive uncertainty diagnostic).
    sxx = float(np.sum((x - x.mean()) ** 2))
    if len(x) > 2 and sxx > 0:
        slope_se = float(np.sqrt((ss_res / (len(x) - 2)) / sxx))
    else:
        slope_se = np.nan

    residual_fit_corr = (
        float(np.corrcoef(residuals, fitted)[0, 1])
        if np.std(residuals) > 0 and np.std(fitted) > 0
        else np.nan
    )

    return {
        "Elasticity": float(slope),
        "Intercept": float(intercept),
        "RSquared": float(r2),
        "MonthlyObservations": int(len(p)),
        "SlopeStdError": slope_se,
        "LogResidualRMSE": rmse_log,
        "ResidualFitCorrelation": residual_fit_corr,
        "MinMonthlyPrice": float(p["AverageSellingPrice"].min()),
        "MaxMonthlyPrice": float(p["AverageSellingPrice"].max()),
        "DistinctMonthlyAveragePrices": int(p["AverageSellingPrice"].nunique()),
    }


def _vectorized_loglog_by_group(frame: pd.DataFrame, group_columns: list[str],
                               min_points: int) -> pd.DataFrame:
    """Fast grouped OLS for ln(quantity) on ln(weighted average price)."""
    d = frame.loc[
        (frame["AverageSellingPrice"] > 0) & (frame["TotalQuantity"] > 0),
        group_columns + ["AverageSellingPrice", "TotalQuantity"]
    ].copy()
    d["x"] = np.log(d["AverageSellingPrice"].astype(float))
    d["y"] = np.log(d["TotalQuantity"].astype(float))
    d["xx"] = d["x"] * d["x"]
    d["yy"] = d["y"] * d["y"]
    d["xy"] = d["x"] * d["y"]

    g = d.groupby(group_columns, as_index=False, sort=False).agg(
        n=("x", "size"),
        nx=("x", "nunique"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxx=("xx", "sum"),
        syy=("yy", "sum"),
        sxy=("xy", "sum"),
        min_price=("AverageSellingPrice", "min"),
        max_price=("AverageSellingPrice", "max"),
        distinct_prices=("AverageSellingPrice", "nunique"),
    )
    g["Sxx"] = g["sxx"] - (g["sx"] ** 2 / g["n"])
    g["Syy"] = g["syy"] - (g["sy"] ** 2 / g["n"])
    g["Sxy"] = g["sxy"] - (g["sx"] * g["sy"] / g["n"])
    g["Elasticity"] = g["Sxy"] / g["Sxx"].replace(0, np.nan)
    g["Intercept"] = (g["sy"] - g["Elasticity"] * g["sx"]) / g["n"]
    g["SSE"] = (g["Syy"] - g["Elasticity"] * g["Sxy"]).clip(lower=0)
    g["RSquared"] = 1 - g["SSE"] / g["Syy"].replace(0, np.nan)
    g["LogResidualRMSE"] = np.sqrt(g["SSE"] / g["n"])
    slope_variance = (
        (g["SSE"] / (g["n"] - 2).replace(0, np.nan)) /
        g["Sxx"].where(g["Sxx"] > 0)
    )
    g["SlopeStdError"] = np.sqrt(slope_variance.clip(lower=0))
    g = g[
        (g["n"] >= min_points) & (g["nx"] >= 2) &
        (g["Sxx"] > 0) & (g["Syy"] > 0)
    ].copy()
    g = g.rename(columns={
        "n": "MonthlyObservations",
        "min_price": "MinMonthlyPrice",
        "max_price": "MaxMonthlyPrice",
        "distinct_prices": "DistinctMonthlyAveragePrices",
    })
    g["ResidualFitCorrelation"] = np.nan
    keep = group_columns + [
        "Elasticity", "Intercept", "RSquared", "MonthlyObservations",
        "SlopeStdError", "LogResidualRMSE", "ResidualFitCorrelation",
        "MinMonthlyPrice", "MaxMonthlyPrice", "DistinctMonthlyAveragePrices"
    ]
    return g[keep]


def estimate_elasticities(sales: pd.DataFrame, panel: pd.DataFrame, product: pd.DataFrame):
    """Estimate product-level log-log elasticities using proposal screening rules."""
    transaction_stats = (
        sales.groupby("StockCode", as_index=False, sort=False)
        .agg(
            DistinctTransactionPrices=("UnitPrice", "nunique"),
            TransactionLines=("InvoiceNo", "count"),
            Revenue=("Revenue", "sum"),
        )
    )
    candidates = transaction_stats[
        (transaction_stats["DistinctTransactionPrices"] >= MIN_DISTINCT_TRANSACTION_PRICES)
        & (transaction_stats["TransactionLines"] >= MIN_TRANSACTION_LINES)
    ].copy()

    # Vectorised grouped OLS is substantially faster than fitting thousands
    # of tiny regressions one-by-one.
    estimates = _vectorized_loglog_by_group(
        panel[panel["StockCode"].isin(set(candidates["StockCode"]))],
        ["StockCode"],
        MIN_MONTHLY_PANEL_POINTS
    )
    elast = estimates.merge(
        product[["StockCode", "Description"]], on="StockCode", how="left"
    ).merge(transaction_stats, on="StockCode", how="left")

    elast["ElasticityClass"] = np.select(
        [
            elast["Elasticity"] < -1,
            (elast["Elasticity"] >= -1) & (elast["Elasticity"] < 0),
            elast["Elasticity"] == 0,
        ],
        ["Elastic", "Inelastic", "Unitary"],
        default="Positive estimate",
    )
    elast["ExtremeElasticityFlag"] = elast["Elasticity"].abs() > MAX_ABS_ELASTICITY_FOR_SCENARIO
    elast["LowFitFlag"] = elast["RSquared"] < MIN_R_SQUARED_FOR_SCENARIO
    elast["ReliableForScenario"] = (
        (elast["Elasticity"] < 0)
        & (elast["RSquared"] >= MIN_R_SQUARED_FOR_SCENARIO)
        & (elast["Elasticity"].abs() <= MAX_ABS_ELASTICITY_FOR_SCENARIO)
        & (elast["MonthlyObservations"] >= MIN_MONTHLY_PANEL_POINTS)
    )
    elast["ReliabilityStatus"] = np.select(
        [
            elast["Elasticity"] >= 0,
            elast["ExtremeElasticityFlag"],
            elast["LowFitFlag"],
        ],
        [
            "Positive estimate - investigate",
            "Extreme estimate - investigate",
            "Low model fit - investigate",
        ],
        default="Passes exploratory screen",
    )
    elast = elast.sort_values("Revenue", ascending=False)
    return elast, candidates


def peak_offpeak_analysis(panel: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
    """Estimate separate peak/off-peak relationships where enough periods exist."""
    eligible = panel[panel["StockCode"].isin(set(elasticity["StockCode"]))].copy()
    eligible["PeriodType"] = np.where(
        eligible["MonthNumber"].isin(PEAK_MONTHS), "Peak", "OffPeak"
    )
    estimates = _vectorized_loglog_by_group(
        eligible, ["StockCode", "PeriodType"],
        min_points=3
    )
    if estimates.empty:
        return pd.DataFrame(columns=[
            "StockCode", "PeakElasticity", "PeakRSquared", "PeakObservations",
            "OffPeakElasticity", "OffPeakRSquared", "OffPeakObservations",
            "PeakMinusOffPeakElasticity"
        ])

    wide = estimates.pivot(index="StockCode", columns="PeriodType")
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()

    def col(name):
        return wide[name] if name in wide.columns else np.nan

    result = pd.DataFrame({
        "StockCode": wide["StockCode"],
        "PeakElasticity": col("Elasticity_Peak"),
        "PeakRSquared": col("RSquared_Peak"),
        "PeakObservations": col("MonthlyObservations_Peak"),
        "OffPeakElasticity": col("Elasticity_OffPeak"),
        "OffPeakRSquared": col("RSquared_OffPeak"),
        "OffPeakObservations": col("MonthlyObservations_OffPeak"),
    })
    result["PeakMinusOffPeakElasticity"] = (
        result["PeakElasticity"] - result["OffPeakElasticity"]
    )
    return result


def build_scenarios(product_abc: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
    """
    Product-level what-if scenarios using:
      R_new = R_base * (1 + d) ** (1 + elasticity)

    Only products passing the transparent exploratory screen are adjusted.
    Other products are held at baseline (simulated change = zero).
    """
    base = product_abc[[
        "StockCode", "Description", "TotalRevenue", "TotalQuantity",
        "ABCCategory", "RevenueShare"
    ]].merge(
        elasticity[[
            "StockCode", "Elasticity", "RSquared", "MonthlyObservations",
            "ElasticityClass", "ReliableForScenario", "ReliabilityStatus"
        ]],
        on="StockCode", how="left"
    )

    scenario_frame = pd.DataFrame({"PriceChangePct": PRICE_CHANGE_SCENARIOS})
    out = base.merge(scenario_frame, how="cross")
    out["ReliableForScenario"] = out["ReliableForScenario"].fillna(False).astype(bool)
    out["ScenarioName"] = out["PriceChangePct"].map(lambda x: f"{x:+.0%} price change")

    applied = (
        out["ReliableForScenario"]
        & out["Elasticity"].notna()
        & (out["Elasticity"] < 0)
        & ((1 + out["PriceChangePct"]) > 0)
    )
    out["ScenarioApplied"] = applied
    price_factor = np.where(applied, 1 + out["PriceChangePct"], 1.0)
    elasticity_values = out["Elasticity"].fillna(0).to_numpy(dtype=float)

    out["SimulatedPriceIndex"] = price_factor
    out["SimulatedRevenue"] = out["TotalRevenue"] * np.where(
        applied, np.power(price_factor, 1 + elasticity_values), 1.0
    )
    out["SimulatedQuantity"] = out["TotalQuantity"] * np.where(
        applied, np.power(price_factor, elasticity_values), 1.0
    )
    out["BaselineRevenue"] = out["TotalRevenue"]
    out["BaselineQuantity"] = out["TotalQuantity"]
    out["RevenueDifference"] = out["SimulatedRevenue"] - out["BaselineRevenue"]
    out["RevenueChangePct"] = np.where(
        out["BaselineRevenue"] != 0,
        out["RevenueDifference"] / out["BaselineRevenue"] * 100,
        np.nan
    )
    keep = [
        "StockCode", "Description", "ABCCategory", "BaselineRevenue",
        "BaselineQuantity", "Elasticity", "RSquared", "MonthlyObservations",
        "ReliableForScenario", "ReliabilityStatus", "PriceChangePct",
        "ScenarioName", "SimulatedPriceIndex", "SimulatedRevenue",
        "SimulatedQuantity", "RevenueDifference", "RevenueChangePct",
        "ScenarioApplied"
    ]
    return out[keep]


def build_recommendations(product_abc: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
    """Create transparent, rules-based product action labels."""
    out = product_abc.merge(
        elasticity, on=["StockCode", "Description"], how="left",
        suffixes=("", "_Elasticity")
    )
    e = out["Elasticity"]
    r2 = out["RSquared"]
    no_estimate = e.isna()
    positive = e >= 0
    extreme = e.abs() > MAX_ABS_ELASTICITY_FOR_SCENARIO
    low_fit = r2.isna() | (r2 < MIN_R_SQUARED_FOR_SCENARIO)
    out["PricingAction"] = np.select(
        [
            no_estimate,
            positive,
            extreme,
            low_fit,
            e < -1,
            e > -1,
        ],
        [
            "Further investigation - insufficient data",
            "Investigate positive price-demand estimate",
            "Further investigation - extreme estimate",
            "Further investigation - low model fit",
            "Test targeted discount",
            "Test controlled price increase",
        ],
        default="Evaluate unit-elastic pricing",
    )
    out["SuggestedTestPrice_5pct"] = np.select(
        [
            out["PricingAction"] == "Test controlled price increase",
            out["PricingAction"] == "Test targeted discount",
        ],
        [1.05, 0.95],
        default=1.00,
    )
    out["RecommendationNote"] = (
        "Exploratory recommendation only; validate through a controlled price test. "
        "Revenue simulation is not profit optimisation."
    )
    return out


def build_data_quality_summary(raw: pd.DataFrame, sales: pd.DataFrame, panel: pd.DataFrame,
                               elasticity: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Compact data quality and modelling summary for the dashboard."""
    records = [
        ("Raw transaction rows", len(raw), "Rows in source file"),
        ("Cleaned transaction rows", len(sales), "Rows after documented cleaning"),
        ("Rows removed", len(raw) - len(sales), "Raw minus cleaned rows"),
        ("Unique invoices", sales["InvoiceNo"].nunique(), "Distinct invoice numbers"),
        ("Unique customers", sales["CustomerID"].nunique(), "Distinct non-missing customers"),
        ("Unique products", sales["StockCode"].nunique(), "Distinct StockCodes"),
        ("Total revenue GBP", sales["Revenue"].sum(), "Quantity multiplied by UnitPrice"),
        ("Total quantity", sales["Quantity"].sum(), "Sum of positive quantities"),
        ("First transaction date", sales["InvoiceDate"].min().date().isoformat(), "Date range start"),
        ("Last transaction date", sales["InvoiceDate"].max().date().isoformat(), "Date range end"),
        ("Product-month rows", len(panel), "Rows in product-month panel"),
        ("Products with elasticity", len(elasticity), "Products with estimable log-log regression"),
        ("Products passing scenario screen", int(elasticity["ReliableForScenario"].sum()) if len(elasticity) else 0,
         "Negative elasticity, R2 threshold, observation threshold and non-extreme estimate"),
        ("R-squared screen", MIN_R_SQUARED_FOR_SCENARIO, "Exploratory project threshold, not universal rule"),
        ("Maximum absolute elasticity screen", MAX_ABS_ELASTICITY_FOR_SCENARIO, "Extreme estimates held out of scenario adjustments"),
        ("Peak months", "November, December", "Calendar months defined as peak in project script"),
    ]
    return pd.DataFrame(records, columns=["Metric", "Value", "Definition"])


def write_powerbi_instructions(output_dir: Path) -> None:
    dax = r"""
POWER BI STARTER DAX MEASURES
=============================

Assumed table names:
  FactSales
  ProductMonth
  Product
  Calendar
  Month
  ElasticityResults
  PricingScenarios
  ProductRecommendations

Create relationships:
  Product[StockCode] 1 -> * FactSales[StockCode]
  Product[StockCode] 1 -> * ProductMonth[StockCode]
  Product[StockCode] 1 -> * ElasticityResults[StockCode]
  Product[StockCode] 1 -> * PricingScenarios[StockCode]
  Product[StockCode] 1 -> * ProductRecommendations[StockCode]
  Calendar[Date] 1 -> * FactSales[Date]
  Month[MonthStart] 1 -> * ProductMonth[MonthStart]

Use single-direction filtering from dimensions to facts. Mark Calendar as
the date table using Calendar[Date]. Sort MonthName by MonthNumber.

BASIC MEASURES
--------------

Total Revenue = SUM(FactSales[Revenue])

Total Quantity = SUM(FactSales[Quantity])

Total Invoices = DISTINCTCOUNT(FactSales[InvoiceNo])

Unique Customers = DISTINCTCOUNT(FactSales[CustomerID])

Unique Products = DISTINCTCOUNT(FactSales[StockCode])

Average Selling Price = DIVIDE([Total Revenue], [Total Quantity])

Products with Elasticity =
DISTINCTCOUNT(ElasticityResults[StockCode])

Elastic Products =
CALCULATE(
    DISTINCTCOUNT(ElasticityResults[StockCode]),
    ElasticityResults[Elasticity] < -1
)

Inelastic Products =
CALCULATE(
    DISTINCTCOUNT(ElasticityResults[StockCode]),
    ElasticityResults[Elasticity] >= -1,
    ElasticityResults[Elasticity] < 0
)

Positive Elasticity Estimates =
CALCULATE(
    DISTINCTCOUNT(ElasticityResults[StockCode]),
    ElasticityResults[Elasticity] >= 0
)

Median Elasticity = MEDIAN(ElasticityResults[Elasticity])

Scenario measures can be built from the exported PricingScenarios table:
Scenario Revenue = SUM(PricingScenarios[SimulatedRevenue])
Baseline Scenario Revenue = SUM(PricingScenarios[BaselineRevenue])
Scenario Revenue Difference =
    [Scenario Revenue] - [Baseline Scenario Revenue]

IMPORTANT:
PricingScenarios already contains the seven fixed scenario changes
(-15%, -10%, -5%, 0%, +5%, +10%, +15%). Add PriceChangePct or ScenarioName
as a slicer. Do not sum all seven scenarios together; filter to exactly
one scenario when displaying a scenario total.

The exported scenario table holds products that fail the reliability
screen at baseline revenue/quantity. ScenarioApplied identifies products
for which the illustrative price adjustment was applied.

For a fully interactive custom slider rather than the fixed seven scenarios,
create a disconnected What-if parameter and use the elasticity formula in a
measure. Validate its totals against the scenario export before presenting.
"""
    (output_dir / "PowerBI_DAX_Measures.txt").write_text(dax.strip() + "\n", encoding="utf-8")

    readme = """
NEXTGEN POWER BI DATA MODEL
===========================

Import the CSV files as separate tables. Suggested model:

Dimensions:
  Product
  Calendar (daily)
  Month (one row per month)

Facts / analytical tables:
  FactSales
  ProductMonth
  ElasticityResults
  PricingScenarios
  ProductRecommendations
  MonthlySummary
  PeakOffPeakElasticity
  ProductSummary_ABC
  DataQualityAudit

Recommended relationships:
  Product[StockCode] -> FactSales[StockCode]
  Product[StockCode] -> ProductMonth[StockCode]
  Product[StockCode] -> ElasticityResults[StockCode]
  Product[StockCode] -> PricingScenarios[StockCode]
  Product[StockCode] -> ProductRecommendations[StockCode]
  Product[StockCode] -> ProductSummary_ABC[StockCode]
  Product[StockCode] -> PeakOffPeakElasticity[StockCode]
  Calendar[Date] -> FactSales[Date]
  Month[MonthStart] -> ProductMonth[MonthStart]
  Month[MonthStart] -> MonthlySummary[MonthStart]

Use one-to-many, single-direction relationships from dimensions to facts.
The ProductMonth and ElasticityResults tables are product-level analytical
tables; do not create a many-to-many relationship between them.

Suggested report pages:
  1. Executive Overview
  2. Product & Sales Intelligence
  3. Price Elasticity
  4. Seasonality
  5. Pricing Scenario Simulator
  6. Product Recommendations
  7. Methodology & Data Quality

Important fields:
  FactSales[Revenue] = Quantity * UnitPrice
  ProductMonth[AverageSellingPrice] = monthly Revenue / monthly Quantity
  ElasticityResults[Elasticity] = log-log regression slope
  ElasticityResults[RSquared] = in-sample coefficient of determination
  PricingScenarios[PriceChangePct] = scenario change as decimal
  PricingScenarios[ScenarioApplied] = whether the price change was applied

Cautions:
  - The source is observational transaction data, not a randomized experiment.
  - Elasticity is an estimated association, not a causal response.
  - The data has no cost field; this is revenue, not profit, analysis.
  - December 2011 is partial (source ends 9 December 2011).
  - One year of data is insufficient to establish recurring annual seasonality.
  - R-squared threshold 0.30 and extreme elasticity threshold 10 are
    exploratory screening choices. Explain and sensitivity-test them.
  - Scenario results assume the estimated historical relationship remains
    stable and other factors do not change.
"""
    (output_dir / "README_PowerBI_Model.txt").write_text(readme.strip() + "\n", encoding="utf-8")


def export_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "FactSales": "01_FactSales.csv",
        "ProductMonth": "02_ProductMonth.csv",
        "Product": "03_Product.csv",
        "Calendar": "04_Calendar.csv",
        "Month": "05_Month.csv",
        "ElasticityResults": "06_ElasticityResults.csv",
        "MonthlySummary": "07_MonthlySummary.csv",
        "ProductSummary_ABC": "08_ProductSummary_ABC.csv",
        "PeakOffPeakElasticity": "09_PeakOffPeakElasticity.csv",
        "PricingScenarios": "10_PricingScenarios.csv",
        "ProductRecommendations": "11_ProductRecommendations.csv",
        "DataQualityAudit": "12_DataQualityAudit.csv",
    }

    for key, filename in filenames.items():
        frame = tables[key].copy()
        # Power BI handles UTF-8 BOM CSV reliably.
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    # CSV is the recommended Power BI import format. Avoid creating one giant
    # Excel workbook: the transaction fact table is large and CSV imports faster.
    write_powerbi_instructions(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run the NextGen retail pricing analysis and prepare Power BI tables."
    )
    parser.add_argument(
        "--input",
        default="Online Retail.xlsx",
        help="Path to original Online Retail Excel/CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default="nextgen_powerbi_data",
        help="Folder for all Power BI-ready outputs."
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    log("=" * 72)
    log("NEXTGEN | DYNAMIC PRICING ANALYSIS")
    log("=" * 72)

    raw = load_data(input_path)
    log(f"Loaded source: {input_path} ({len(raw):,} rows)")

    sales, audit = clean_data(raw)
    fact_sales, product, panel, calendar, month_dimension, monthly = build_dimensions_and_facts(sales)

    log("Building ABC/Pareto product analysis...")
    product_abc = abc_analysis(product)

    log("Estimating product-level log-log elasticity...")
    elasticity, candidates = estimate_elasticities(fact_sales, panel, product)
    log(f"Products meeting transaction eligibility: {len(candidates):,}")
    log(f"Products with estimable elasticity: {len(elasticity):,}")

    log("Building peak/off-peak comparisons...")
    peak_offpeak = peak_offpeak_analysis(panel, elasticity)

    log("Building price-change what-if scenarios...")
    scenarios = build_scenarios(product_abc, elasticity)

    log("Building product recommendations...")
    recommendations = build_recommendations(product_abc, elasticity)

    quality = build_data_quality_summary(raw, fact_sales, panel, elasticity, audit)

    tables = {
        "FactSales": fact_sales,
        "ProductMonth": panel,
        "Product": product,
        "Calendar": calendar,
        "Month": month_dimension,
        "ElasticityResults": elasticity,
        "MonthlySummary": monthly,
        "ProductSummary_ABC": product_abc,
        "PeakOffPeakElasticity": peak_offpeak,
        "PricingScenarios": scenarios,
        "ProductRecommendations": recommendations,
        "DataQualityAudit": pd.concat(
            [
                audit.assign(Metric="Cleaning step", Value=audit["RowsRemaining"].astype(str),
                             Definition=audit["CleaningStep"])[["Metric", "Value", "Definition"]],
                quality
            ],
            ignore_index=True
        ),
    }

    export_outputs(output_dir, tables)

    # Concise run summary.
    total_revenue = fact_sales["Revenue"].sum()
    reliable_count = int(elasticity["ReliableForScenario"].sum()) if len(elasticity) else 0
    log("")
    log("=" * 72)
    log("RUN COMPLETE")
    log("=" * 72)
    log(f"Cleaned rows:                 {len(fact_sales):,}")
    log(f"Unique invoices:              {fact_sales['InvoiceNo'].nunique():,}")
    log(f"Unique customers:             {fact_sales['CustomerID'].nunique():,}")
    log(f"Unique products:              {fact_sales['StockCode'].nunique():,}")
    log(f"Revenue (GBP):                £{total_revenue:,.2f}")
    log(f"Product-month rows:           {len(panel):,}")
    log(f"Elasticity estimates:         {len(elasticity):,}")
    log(f"Products passing scenario screen: {reliable_count:,}")
    log(f"Scenario rows:                {len(scenarios):,}")
    log(f"Output folder:                {output_dir.resolve()}")
    log("")
    log("Next: import the CSV tables into Power BI and follow README_PowerBI_Model.txt")
    log("Read PowerBI_DAX_Measures.txt for starter measures.")
    log("Remember: estimates are observational associations, not causal effects.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
