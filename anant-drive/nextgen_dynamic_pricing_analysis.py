"""
NextGen - Industry 4.0 (Sep 2026) team project
Dynamic Pricing Strategy for Revenue Optimization in Retail Sales

Dataset: Online Retail, UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/352/online+retail

What this script does (follows Section 6 of the proposal):
  1. Data cleaning: removes cancellations/returns, non-positive
     quantities/prices, rows without a CustomerID; restricts to UK sales.
  2. Structuring: builds a product-month panel (quantity, avg price, revenue).
  3. EDA: revenue and volume trends, top products, Pareto (ABC) analysis.
  4. Elasticity estimation: log-log regression of quantity on price per
     product (slope = elasticity), with R-squared and a residual check.
  5. Seasonality: monthly demand pattern; elasticity in peak (Nov-Dec) vs
     off-peak months.
  6. Pricing recommendations: classify products elastic/inelastic and
     simulate revenue at alternative price points.

Uses only NumPy and Pandas.

NOTE (from the proposal): transactions are observational, not a price
experiment. The elasticities below are associations between price and
demand, not proven causal effects. The dataset has no cost data, so all
results are about revenue, not profit.

Run:
  pip install pandas numpy openpyxl
  python nextgen_dynamic_pricing_analysis.py
"""

import numpy as np
import pandas as pd

# Path to the raw dataset file downloaded from UCI (xlsx or csv).
FILE_PATH = "data/Online Retail.xlsx"  # change to your file name if different

# Products enter the elasticity step only if they have at least this many
# distinct selling prices and this many transaction lines (proposal Sec. 5).
MIN_DISTINCT_PRICES = 3
MIN_TRANSACTION_LINES = 50

# A product needs at least this many months in the panel for its log-log
# regression to be meaningful.
MIN_PANEL_POINTS = 5

# Months treated as the festive peak for the peak vs off-peak comparison.
PEAK_MONTHS = {11, 12}  # November, December

# Stock codes that are not real products (fees, postage, manual entries).
# Removed in cleaning as data-entry outliers (proposal Sec. 6, step 1).
NON_PRODUCT_CODES = {"POST", "C2", "M", "BANK", "PADS", "DOT",
                     "CRUK", "D", "S", "AMAZONFEE"}


def load_data(path):
    """Load the raw Online Retail file (Excel or CSV)."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    """Proposal step 1: cleaning. Returns the cleaned UK retail data."""
    n_raw = len(df)
    df = df.copy()

    # Parse dates and drop cancellations (invoice numbers start with 'C').
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Keep only real sales: positive quantity and positive price.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Drop rows without a customer ID and keep UK sales only
    # (UK is ~85% of revenue; proposal Sec. 5).
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Country"] == "United Kingdom"]

    # Drop non-product codes (postage, carriage, manual adjustments...).
    df = df[~df["StockCode"].astype(str).isin(NON_PRODUCT_CODES)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print("=== 1. Cleaning ===")
    print(f"Raw rows:              {n_raw:,}")
    print(f"Cleaned UK rows:       {len(df):,}")
    print(f"Invoices:              {df['InvoiceNo'].nunique():,}")
    print(f"Customers:             {df['CustomerID'].nunique():,}")
    print(f"Products:              {df['StockCode'].nunique():,}")
    print(f"Total revenue (GBP):   {df['Revenue'].sum():,.0f}")
    print(f"Period: {df['InvoiceDate'].min():%d %b %Y} to "
          f"{df['InvoiceDate'].max():%d %b %Y}")

    # Save the cleaned dataset so the team has one shared clean file
    # (no cleaned file was kept from the earlier proposal EDA).
    df.to_csv("anant-drive/cleaned_online_retail_uk.csv", index=False)
    print("Cleaned dataset saved to cleaned_online_retail_uk.csv\n")
    return df


def build_panel(df):
    """Proposal step 2: product-month panel.

    Average price is revenue-weighted (revenue / quantity), not a plain
    mean of line prices, so heavy discount lines count more.
    """
    df = df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    # Group by StockCode only: a product can have several spellings of its
    # description, and grouping by both would split one product's data.
    panel = (df.groupby(["StockCode", "Month"])
               .agg(Description=("Description", "first"),
                    total_qty=("Quantity", "sum"),
                    revenue=("Revenue", "sum"),
                    n_lines=("InvoiceNo", "count"))
               .reset_index())
    panel["avg_price"] = panel["revenue"] / panel["total_qty"]
    return panel


def run_eda(df, panel):
    """Proposal step 3: EDA."""
    print("=== 3. EDA ===")

    # Monthly revenue and volume trend.
    monthly = (panel.groupby("Month")
                    .agg(revenue=("revenue", "sum"),
                         qty=("total_qty", "sum"))
                    .reset_index()
                    .sort_values("Month"))
    print("Monthly revenue and volume:")
    for _, r in monthly.iterrows():
        print(f"  {r['Month']}   GBP {r['revenue']:>10,.0f}   "
              f"{int(r['qty']):>7,} units")

    # Top 10 products by revenue.
    top = (panel.groupby(["StockCode", "Description"])["revenue"]
                .sum()
                .sort_values(ascending=False)
                .head(10))
    print("\nTop 10 products by revenue:")
    for (code, desc), rev in top.items():
        print(f"  {code:<10} {str(desc)[:40]:<42} GBP {rev:>9,.0f}")

    # Pareto (ABC) analysis: how many products make 80% of revenue?
    rev_per_product = (panel.groupby("StockCode")["revenue"]
                            .sum()
                            .sort_values(ascending=False))
    cum_share = rev_per_product.cumsum() / rev_per_product.sum()
    n_for_80 = int((cum_share < 0.8).sum()) + 1
    print(f"\nPareto: {n_for_80} of {len(rev_per_product)} products "
          f"({n_for_80 / len(rev_per_product):.0%}) make 80% of revenue.\n")


def log_log_elasticity(product_panel, min_points=MIN_PANEL_POINTS):
    """Log-log regression of quantity on price for one product's panel.

    log(qty) = a + b * log(price); the slope b is the price elasticity.
    numpy.polyfit does the least-squares fit; R-squared and the residual
    pattern are computed by hand. Returns (elasticity, r_squared, n,
    residual-fit correlation) or None if there is too little price
    variation to estimate anything. min_points can be lowered for the
    peak/off-peak split, where only a few months exist per side.
    """
    p = product_panel
    if len(p) < min_points or p["avg_price"].nunique() < 2:
        return None

    x = np.log(p["avg_price"].to_numpy())
    y = np.log(p["total_qty"].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)

    y_hat = slope * x + intercept
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Residual pattern check: correlation between residuals and fitted
    # values. Near 0 means no obvious leftover trend in the errors.
    resid_corr = float(np.corrcoef(residuals, y_hat)[0, 1])

    return slope, r_squared, len(p), resid_corr


def estimate_elasticities(df, panel):
    """Proposal step 4: elasticity per product.

    Candidate products need real within-product price variation
    (>= 3 distinct prices) and enough volume (>= 50 transaction lines),
    matching the feasibility check in proposal Sec. 5.
    """
    stats = (df.groupby("StockCode")
               .agg(Description=("Description", "first"),
                    distinct_prices=("UnitPrice", "nunique"),
                    n_lines=("InvoiceNo", "count"),
                    revenue=("Revenue", "sum"))
               .reset_index())
    candidates = stats[(stats["distinct_prices"] >= MIN_DISTINCT_PRICES)
                       & (stats["n_lines"] >= MIN_TRANSACTION_LINES)]
    print("=== 4. Elasticity estimation (log-log regression) ===")
    print(f"Candidate products: {len(candidates):,} of "
          f"{len(stats):,} (>= {MIN_DISTINCT_PRICES} distinct prices, "
          f">= {MIN_TRANSACTION_LINES} lines)")

    rows = []
    for _, c in candidates.iterrows():
        code = c["StockCode"]
        result = log_log_elasticity(panel[panel["StockCode"] == code])
        if result is None:
            continue
        slope, r2, n, resid_corr = result
        rows.append({
            "StockCode": code,
            "Description": c["Description"],
            "elasticity": round(slope, 3),
            "r_squared": round(r2, 3),
            "months": n,
            "resid_corr": round(resid_corr, 3),
            "revenue": round(c["revenue"], 2),
        })

    elast = pd.DataFrame(rows)
    if elast.empty:
        print("No products had enough panel points for regression.\n")
        return elast, candidates

    # Descriptive patterns relevant to H1/H2. These are simple shares
    # and ranges of fitted associations, not statistical tests, so they
    # cannot prove or disprove the hypotheses on their own.
    share_negative = (elast["elasticity"] < 0).mean()
    print(f"Products with a usable estimate: {len(elast):,}")
    print(f"Share with negative fitted elasticity: {share_negative:.0%} "
          "(descriptive, relevant to H1 - not a test)")
    print(f"Fitted elasticity range: {elast['elasticity'].min():.2f} to "
          f"{elast['elasticity'].max():.2f} "
          f"(median {elast['elasticity'].median():.2f}; "
          "spread relevant to H2)\n")

    elast = elast.sort_values("elasticity")
    elast.to_csv("anant-drive/elasticity_results.csv", index=False)
    print("Full table saved to elasticity_results.csv\n")
    return elast, candidates


def seasonality(df, panel, candidates):
    """Proposal step 5: monthly seasonality and peak vs off-peak
    elasticity for the highest-revenue candidate products."""
    print("=== 5. Seasonality ===")
    df = df.copy()
    df["MonthNum"] = df["InvoiceDate"].dt.month
    monthly_rev = df.groupby("MonthNum")["Revenue"].sum().sort_index()
    peak_share = monthly_rev.loc[monthly_rev.index.isin(PEAK_MONTHS)].sum() \
        / monthly_rev.sum()
    print(f"Peak months (Nov-Dec) share of UK revenue: {peak_share:.0%}")

    # Peak vs off-peak elasticity for the top 10 revenue candidates.
    top_codes = candidates.nlargest(10, "revenue")
    print("\nPeak vs off-peak elasticity (top 10 revenue candidates;")
    print("only ~3 peak months exist, so the threshold is lowered to 3)")
    print(f"  {'StockCode':<10} {'off-peak':>9} {'peak':>9}")
    for _, c in top_codes.iterrows():
        code = c["StockCode"]
        sub = panel[panel["StockCode"] == code].copy()
        sub["MonthNum"] = sub["Month"].dt.month
        off = log_log_elasticity(sub[~sub["MonthNum"].isin(PEAK_MONTHS)],
                                 min_points=3)
        peak = log_log_elasticity(sub[sub["MonthNum"].isin(PEAK_MONTHS)],
                                  min_points=3)
        off_e = f"{off[0]:.2f}" if off else "n/a"
        peak_e = f"{peak[0]:.2f}" if peak else "n/a"
        print(f"  {code:<10} {off_e:>9} {peak_e:>9}")
    print()


def pricing_recommendations(elast):
    """Proposal step 6: classify products and simulate revenue at
    alternative price points using each product's own fitted elasticity.

    Simulation assumption: with fitted elasticity e, a price change of
    d% scales quantity by (1 + d)^e, so simulated revenue =
    new_price * new_qty. This assumes the fitted historical association
    stays stable and nothing else changes (season, stock, competitors,
    customer mix). It is a what-if illustration, not a forecast.
    """
    print("=== 6. Pricing recommendations ===")
    negative = elast[elast["elasticity"] < 0].copy()
    if negative.empty:
        print("No products with a negative elasticity estimate.")
        return

    negative["class"] = np.where(negative["elasticity"] < -1,
                                 "elastic", "inelastic")
    inelastic = negative[negative["class"] == "inelastic"].nlargest(
        10, "revenue")
    # Discount candidates need a decent fit (R2 >= 0.3); simulating a
    # discount from a weak regression would not be defensible.
    elastic = negative[(negative["class"] == "elastic")
                       & (negative["r_squared"] >= 0.3)].nlargest(
        10, "revenue")

    print("Inelastic products (|e| < 1): candidates for price increases")
    print("NOTE: every inelastic estimate here has a low R-squared, so")
    print("price insensitivity is weak evidence - validate before acting.")
    print(f"  {'StockCode':<10} {'e':>6} {'R2':>5} {'rev GBP':>10}  "
          "rev at +5% price (sim)")
    for _, r in inelastic.iterrows():
        sim = (1.05 ** (1 + r["elasticity"])) * r["revenue"]
        print(f"  {r['StockCode']:<10} {r['elasticity']:>6.2f} "
              f"{r['r_squared']:>5.2f} {r['revenue']:>10,.0f}  "
              f"{sim:>10,.0f}")

    print("\nElastic products (e < -1, R2 >= 0.3): candidates for "
          "discounts/promos")
    print(f"  {'StockCode':<10} {'e':>6} {'R2':>5} {'rev GBP':>10}  "
          "rev at -5% price (sim)")
    for _, r in elastic.iterrows():
        sim = (0.95 ** (1 + r["elasticity"])) * r["revenue"]
        print(f"  {r['StockCode']:<10} {r['elasticity']:>6.2f} "
              f"{r['r_squared']:>5.2f} {r['revenue']:>10,.0f}  "
              f"{sim:>10,.0f}")

    print("\nReminder: these are what-if revenue simulations built on")
    print("fitted associations from observational data, not causal")
    print("estimates or forecasts of what a price change will do.")


def main():
    df = load_data(FILE_PATH)
    df = clean_data(df)
    panel = build_panel(df)
    run_eda(df, panel)
    elast, candidates = estimate_elasticities(df, panel)
    if not elast.empty:
        seasonality(df, panel, candidates)
        pricing_recommendations(elast)


if __name__ == "__main__":
    main()
