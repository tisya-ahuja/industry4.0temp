#!/usr/bin/env python
"""
NextGen | Dynamic Pricing Project
Generate all project figures from the CSV files created by
nextgen_dynamic_pricing_powerbi.py.

INSTALL
-------
pip install pandas matplotlib numpy

RUN (from the project folder)
-----------------------------
python generate_project_figures.py

Optional:
python generate_project_figures.py --input-dir nextgen_powerbi_data
python generate_project_figures.py --input-dir nextgen_powerbi_data --output-dir NextGen_Figures

OUTPUT
------
Creates:
  <output-dir>/PNG/          Individual high-resolution PNG charts
  <output-dir>/NextGen_Figures_Report.pdf   All charts in one PDF

This script reads the analysis outputs; it does not rerun the analysis.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Consistent readable defaults; matplotlib's default colour cycle is retained.
plt.rcParams.update({
    "figure.figsize": (11, 6.5),
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


FILES = {
    "monthly": "07_MonthlySummary.csv",
    "product": "03_Product.csv",
    "product_abc": "08_ProductSummary_ABC.csv",
    "elasticity": "06_ElasticityResults.csv",
    "peak_offpeak": "09_PeakOffPeakElasticity.csv",
    "scenarios": "10_PricingScenarios.csv",
    "recommendations": "11_ProductRecommendations.csv",
    "quality": "12_DataQualityAudit.csv",
}


def read_csv(folder: Path, filename: str, required: bool = True):
    path = folder / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing {path}\nRun nextgen_dynamic_pricing_powerbi.py first."
            )
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def save_figure(fig, name: str, png_dir: Path, pdf: PdfPages):
    fig.tight_layout()
    fig.savefig(png_dir / f"{name}.png", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {name}.png")


def money_axis(ax, axis="y"):
    import matplotlib.ticker as mtick
    formatter = mtick.FuncFormatter(lambda x, pos: f"£{x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def make_monthly_revenue(monthly, png_dir, pdf):
    d = monthly.sort_values("MonthStart").copy()
    d["MonthStart"] = pd.to_datetime(d["MonthStart"])
    fig, ax = plt.subplots()
    ax.plot(d["MonthStart"], d["Revenue"], marker="o", linewidth=2)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (GBP)")
    money_axis(ax)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    save_figure(fig, "01_monthly_revenue", png_dir, pdf)


def make_monthly_quantity(monthly, png_dir, pdf):
    d = monthly.sort_values("MonthStart").copy()
    d["MonthStart"] = pd.to_datetime(d["MonthStart"])
    fig, ax = plt.subplots()
    ax.plot(d["MonthStart"], d["Quantity"], marker="o", linewidth=2)
    ax.set_title("Monthly Quantity Sold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units sold")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    save_figure(fig, "02_monthly_quantity", png_dir, pdf)


def make_monthly_orders(monthly, png_dir, pdf):
    d = monthly.sort_values("MonthStart").copy()
    d["MonthStart"] = pd.to_datetime(d["MonthStart"])
    fig, ax = plt.subplots()
    ax.bar(d["MonthStart"].dt.strftime("%b %Y"), d["InvoiceCount"])
    ax.set_title("Monthly Invoice Count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unique invoices")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "03_monthly_invoices", png_dir, pdf)


def make_monthly_average_price(monthly, png_dir, pdf):
    d = monthly.sort_values("MonthStart").copy()
    d["MonthStart"] = pd.to_datetime(d["MonthStart"])
    fig, ax = plt.subplots()
    ax.plot(d["MonthStart"], d["AverageSellingPrice"], marker="o", linewidth=2)
    ax.set_title("Monthly Revenue-Weighted Average Selling Price")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average selling price (GBP)")
    money_axis(ax)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    save_figure(fig, "04_monthly_average_price", png_dir, pdf)


def make_top_products(product_abc, png_dir, pdf, metric, title, filename, xlabel, formatter=None):
    d = product_abc.nlargest(15, metric).sort_values(metric)
    labels = d["StockCode"].astype(str) + " | " + d["Description"].fillna("").astype(str).str.slice(0, 32)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, d[metric])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if formatter:
        formatter(ax)
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, filename, png_dir, pdf)


def make_pareto(product_abc, png_dir, pdf):
    d = product_abc.sort_values("TotalRevenue", ascending=False).reset_index(drop=True).copy()
    d["ProductRank"] = np.arange(1, len(d) + 1)
    d["CumulativeRevenueShare"] = d["TotalRevenue"].cumsum() / d["TotalRevenue"].sum() * 100
    # Plot top 50 product contribution to keep the figure legible.
    top = d.head(50)
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    ax1.bar(top["ProductRank"], top["TotalRevenue"])
    ax1.set_xlabel("Product rank (top 50)")
    ax1.set_ylabel("Revenue (GBP)")
    money_axis(ax1)
    ax2 = ax1.twinx()
    ax2.plot(top["ProductRank"], top["CumulativeRevenueShare"], marker=".", linewidth=2)
    ax2.set_ylabel("Cumulative revenue share (%)")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, linestyle="--", linewidth=1)
    ax1.set_title("Product Revenue Pareto Analysis — Top 50 Products")
    ax1.grid(True, axis="y", alpha=0.2)
    save_figure(fig, "07_product_pareto", png_dir, pdf)


def make_abc_mix(product_abc, png_dir, pdf):
    d = product_abc.groupby("ABCCategory", as_index=False).agg(
        ProductCount=("StockCode", "nunique"),
        Revenue=("TotalRevenue", "sum")
    )
    order = ["A", "B", "C"]
    d["ABCCategory"] = pd.Categorical(d["ABCCategory"], categories=order, ordered=True)
    d = d.sort_values("ABCCategory")
    fig, ax = plt.subplots()
    bars = ax.bar(d["ABCCategory"].astype(str), d["Revenue"])
    ax.set_title("Revenue Contribution by ABC Product Class")
    ax.set_xlabel("ABC class")
    ax.set_ylabel("Revenue (GBP)")
    money_axis(ax)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, count in zip(bars, d["ProductCount"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{count:,} products", ha="center", va="bottom", fontsize=9)
    save_figure(fig, "08_abc_revenue_contribution", png_dir, pdf)


def make_elasticity_histogram(elasticity, png_dir, pdf):
    d = elasticity["Elasticity"].replace([np.inf, -np.inf], np.nan).dropna()
    # Clip only the display range; retain full data in CSV and report notes.
    lo, hi = d.quantile([0.02, 0.98])
    shown = d[(d >= lo) & (d <= hi)]
    fig, ax = plt.subplots()
    ax.hist(shown, bins=35, edgecolor="black", alpha=0.8)
    ax.axvline(-1, linestyle="--", linewidth=1.5, label="Elasticity = -1")
    ax.axvline(0, linestyle=":", linewidth=1.5, label="Elasticity = 0")
    ax.set_title("Distribution of Estimated Price Elasticities")
    ax.set_xlabel("Estimated elasticity (central 96% shown)")
    ax.set_ylabel("Number of products")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(0.99, 0.97, f"All estimates: {len(d):,}\nMedian: {d.median():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round", "alpha": 0.12})
    save_figure(fig, "09_elasticity_distribution", png_dir, pdf)


def make_elasticity_classes(elasticity, png_dir, pdf):
    d = elasticity["ElasticityClass"].value_counts().reindex(
        ["Elastic", "Inelastic", "Unitary", "Positive estimate"], fill_value=0
    )
    fig, ax = plt.subplots()
    bars = ax.bar(d.index, d.values)
    ax.set_title("Products by Estimated Elasticity Class")
    ax.set_xlabel("Elasticity classification")
    ax.set_ylabel("Number of products")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, d.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:,}", ha="center", va="bottom")
    save_figure(fig, "10_elasticity_classification", png_dir, pdf)


def make_elasticity_vs_revenue(elasticity, png_dir, pdf):
    d = elasticity.replace([np.inf, -np.inf], np.nan).dropna(subset=["Elasticity", "Revenue", "RSquared"])
    fig, ax = plt.subplots()
    sc = ax.scatter(d["Elasticity"], d["Revenue"], c=d["RSquared"], alpha=0.65)
    ax.axvline(-1, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.set_title("Product Elasticity vs Revenue")
    ax.set_xlabel("Estimated price elasticity")
    ax.set_ylabel("Product revenue (GBP)")
    money_axis(ax)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("R-squared")
    ax.grid(True, alpha=0.2)
    save_figure(fig, "11_elasticity_vs_revenue", png_dir, pdf)


def make_r2_vs_elasticity(elasticity, png_dir, pdf):
    d = elasticity.replace([np.inf, -np.inf], np.nan).dropna(subset=["Elasticity", "RSquared"])
    fig, ax = plt.subplots()
    ax.scatter(d["Elasticity"], d["RSquared"], alpha=0.65)
    ax.axvline(-1, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axhline(0.30, linestyle="--", linewidth=1.2, label="Exploratory R² screen = 0.30")
    ax.set_title("Regression Fit vs Estimated Elasticity")
    ax.set_xlabel("Estimated price elasticity")
    ax.set_ylabel("R-squared")
    ax.set_ylim(min(-0.05, d["RSquared"].min() - 0.05), 1.05)
    ax.legend()
    ax.grid(True, alpha=0.2)
    save_figure(fig, "12_r_squared_vs_elasticity", png_dir, pdf)


def make_reliability_counts(elasticity, png_dir, pdf):
    d = elasticity["ReliabilityStatus"].fillna("No status").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(d.index, d.values)
    ax.set_title("Elasticity Estimate Reliability Screening")
    ax.set_xlabel("Number of products")
    ax.grid(True, axis="x", alpha=0.25)
    for y, value in enumerate(d.values):
        ax.text(value, y, f" {value:,}", va="center")
    save_figure(fig, "13_elasticity_reliability_screen", png_dir, pdf)


def make_peak_offpeak(peak, png_dir, pdf):
    if peak.empty:
        return
    d = peak.dropna(subset=["PeakElasticity", "OffPeakElasticity"])
    if d.empty:
        return
    # Use median product elasticity for a robust descriptive comparison.
    values = [d["OffPeakElasticity"].median(), d["PeakElasticity"].median()]
    fig, ax = plt.subplots()
    bars = ax.bar(["Off-peak", "Peak (Nov–Dec)"], values)
    ax.axhline(-1, linestyle="--", linewidth=1, label="Elasticity = -1")
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_title("Median Product Elasticity: Peak vs Off-Peak")
    ax.set_ylabel("Median estimated elasticity")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value,
                f"{value:.2f}", ha="center",
                va="bottom" if value >= 0 else "top")
    save_figure(fig, "14_peak_vs_offpeak_elasticity", png_dir, pdf)


def make_peak_offpeak_scatter(peak, png_dir, pdf):
    if peak.empty:
        return
    d = peak.dropna(subset=["PeakElasticity", "OffPeakElasticity"])
    if d.empty:
        return
    fig, ax = plt.subplots()
    ax.scatter(d["OffPeakElasticity"], d["PeakElasticity"], alpha=0.65)
    low = min(d["OffPeakElasticity"].min(), d["PeakElasticity"].min())
    high = max(d["OffPeakElasticity"].max(), d["PeakElasticity"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    ax.set_title("Product Elasticity: Peak vs Off-Peak")
    ax.set_xlabel("Off-peak elasticity")
    ax.set_ylabel("Peak elasticity")
    ax.grid(True, alpha=0.2)
    save_figure(fig, "15_product_peak_offpeak_scatter", png_dir, pdf)


def make_scenario_portfolio(scenarios, png_dir, pdf):
    d = scenarios.groupby(["PriceChangePct", "ScenarioName"], as_index=False).agg(
        BaselineRevenue=("BaselineRevenue", "sum"),
        SimulatedRevenue=("SimulatedRevenue", "sum"),
        ProductsAdjusted=("ScenarioApplied", "sum"),
    ).sort_values("PriceChangePct")
    d["RevenueDifference"] = d["SimulatedRevenue"] - d["BaselineRevenue"]
    fig, ax = plt.subplots()
    ax.bar(d["ScenarioName"], d["SimulatedRevenue"])
    ax.set_title("Portfolio Revenue Under Alternative Pricing Scenarios")
    ax.set_xlabel("Hypothetical price change")
    ax.set_ylabel("Simulated revenue (GBP)")
    money_axis(ax)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "16_scenario_portfolio_revenue", png_dir, pdf)


def make_scenario_delta(scenarios, png_dir, pdf):
    d = scenarios.groupby(["PriceChangePct", "ScenarioName"], as_index=False).agg(
        RevenueDifference=("RevenueDifference", "sum")
    ).sort_values("PriceChangePct")
    fig, ax = plt.subplots()
    bars = ax.bar(d["ScenarioName"], d["RevenueDifference"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Simulated Revenue Difference vs Baseline")
    ax.set_xlabel("Hypothetical price change")
    ax.set_ylabel("Revenue difference (GBP)")
    money_axis(ax)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "17_scenario_revenue_difference", png_dir, pdf)


def make_scenario_products(scenarios, png_dir, pdf):
    # Product opportunities at +5% for price increase / -5% for discount.
    d = scenarios[scenarios["PriceChangePct"].isin([-0.05, 0.05])].copy()
    d = d[d["ScenarioApplied"] == True]
    if d.empty:
        return
    d = d.sort_values("RevenueDifference", ascending=False).head(15)
    labels = d["StockCode"].astype(str) + " | " + d["Description"].fillna("").astype(str).str.slice(0, 28)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, d["RevenueDifference"])
    ax.set_title("Top Product-Level Simulated Revenue Differences (+5% / -5% Scenarios)")
    ax.set_xlabel("Revenue difference vs baseline (GBP)")
    money_axis(ax, "x")
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, "18_top_product_scenario_differences", png_dir, pdf)


def make_recommendation_actions(recommendations, png_dir, pdf):
    d = recommendations["PricingAction"].fillna("Unclassified").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(d.index, d.values)
    ax.set_title("Products by Pricing Recommendation")
    ax.set_xlabel("Number of products")
    ax.grid(True, axis="x", alpha=0.25)
    for y, value in enumerate(d.values):
        ax.text(value, y, f" {value:,}", va="center")
    save_figure(fig, "19_pricing_recommendation_counts", png_dir, pdf)


def make_recommendation_revenue(recommendations, png_dir, pdf):
    if "PricingAction" not in recommendations or "TotalRevenue" not in recommendations:
        return
    d = recommendations.groupby("PricingAction", as_index=False)["TotalRevenue"].sum()
    d = d.sort_values("TotalRevenue")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(d["PricingAction"], d["TotalRevenue"])
    ax.set_title("Historical Revenue by Pricing Recommendation Group")
    ax.set_xlabel("Historical revenue (GBP)")
    money_axis(ax, "x")
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, "20_revenue_by_recommendation_group", png_dir, pdf)


def make_price_quantity_scatter(product_abc, png_dir, pdf):
    d = product_abc.dropna(subset=["TotalQuantity", "TotalRevenue"]).copy()
    d = d[(d["TotalQuantity"] > 0) & (d["TotalRevenue"] > 0)]
    fig, ax = plt.subplots()
    sc = ax.scatter(d["TotalQuantity"], d["TotalRevenue"], alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Product Revenue vs Quantity Sold")
    ax.set_xlabel("Total quantity sold (log scale)")
    ax.set_ylabel("Total revenue (GBP, log scale)")
    ax.grid(True, alpha=0.2)
    save_figure(fig, "21_product_revenue_vs_quantity", png_dir, pdf)


def main():
    parser = argparse.ArgumentParser(description="Generate all NextGen project charts.")
    parser.add_argument("--input-dir", default="nextgen_powerbi_data",
                        help="Folder containing CSV outputs from the main analysis script.")
    parser.add_argument("--output-dir", default="NextGen_Figures",
                        help="Folder where PNGs and the PDF report will be saved.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    png_dir = output_dir / "PNG"
    png_dir.mkdir(parents=True, exist_ok=True)

    monthly = read_csv(input_dir, FILES["monthly"])
    product = read_csv(input_dir, FILES["product"])
    product_abc = read_csv(input_dir, FILES["product_abc"])
    elasticity = read_csv(input_dir, FILES["elasticity"])
    peak = read_csv(input_dir, FILES["peak_offpeak"], required=False)
    scenarios = read_csv(input_dir, FILES["scenarios"])
    recommendations = read_csv(input_dir, FILES["recommendations"])

    print(f"Reading analysis outputs from: {input_dir.resolve()}")
    print(f"Saving figures to: {output_dir.resolve()}")

    pdf_path = output_dir / "NextGen_Figures_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        make_monthly_revenue(monthly, png_dir, pdf)
        make_monthly_quantity(monthly, png_dir, pdf)
        make_monthly_orders(monthly, png_dir, pdf)
        make_monthly_average_price(monthly, png_dir, pdf)
        make_top_products(
            product_abc, png_dir, pdf, "TotalRevenue",
            "Top 15 Products by Revenue", "05_top_products_revenue",
            "Revenue (GBP)", lambda ax: money_axis(ax)
        )
        make_top_products(
            product_abc, png_dir, pdf, "TotalQuantity",
            "Top 15 Products by Quantity Sold", "06_top_products_quantity",
            "Quantity sold"
        )
        make_pareto(product_abc, png_dir, pdf)
        make_abc_mix(product_abc, png_dir, pdf)
        make_elasticity_histogram(elasticity, png_dir, pdf)
        make_elasticity_classes(elasticity, png_dir, pdf)
        make_elasticity_vs_revenue(elasticity, png_dir, pdf)
        make_r2_vs_elasticity(elasticity, png_dir, pdf)
        make_reliability_counts(elasticity, png_dir, pdf)
        make_peak_offpeak(peak, png_dir, pdf)
        make_peak_offpeak_scatter(peak, png_dir, pdf)
        make_scenario_portfolio(scenarios, png_dir, pdf)
        make_scenario_delta(scenarios, png_dir, pdf)
        make_scenario_products(scenarios, png_dir, pdf)
        make_recommendation_actions(recommendations, png_dir, pdf)
        make_recommendation_revenue(recommendations, png_dir, pdf)
        make_price_quantity_scatter(product_abc, png_dir, pdf)

    print("")
    print("=" * 68)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 68)
    print(f"Individual PNG charts: {png_dir.resolve()}")
    print(f"Combined PDF report:   {pdf_path.resolve()}")
    print("Use the PNG files in your report/presentation and the CSVs in Power BI.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
