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
