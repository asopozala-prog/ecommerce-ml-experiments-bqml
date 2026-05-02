-- Project 3: Daily Demand / Revenue Forecasting with ARIMA_PLUS
-- Dataset: bigquery-public-data.thelook_ecommerce
-- Target dataset: project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01
--
-- Goal:
--   1) Aggregate orders to a daily time series (orders + revenue)
--   2) Explore basic patterns (trend, seasonality)
--   3) Train an ARIMA_PLUS model on daily_revenue
--   4) Forecast the next 90 days
--   5) Evaluate forecast accuracy on a held-out period

-- ============================================================
-- Step 1: Build daily sales table (orders + revenue)
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales` AS

WITH order_totals AS (
  SELECT
    o.order_id,
    DATE(o.created_at) AS order_date,
    SUM(oi.sale_price) AS order_total
  FROM
    `bigquery-public-data.thelook_ecommerce.orders` AS o
  JOIN
    `bigquery-public-data.thelook_ecommerce.order_items` AS oi
  ON
    o.order_id = oi.order_id
  GROUP BY
    o.order_id,
    DATE(o.created_at)
)

SELECT
  order_date,
  COUNT(*) AS daily_num_orders,
  SUM(order_total) AS daily_revenue,
  AVG(order_total) AS daily_avg_order_value
FROM
  order_totals
GROUP BY
  order_date
ORDER BY
  order_date;


-- ============================================================
-- Step 2: Basic EDA helpers (run these SELECTs manually as needed)
-- ============================================================

-- 2a) Date range and number of days
-- SELECT
--   MIN(order_date) AS min_date,
--   MAX(order_date) AS max_date,
--   COUNT(*) AS n_days
-- FROM
--   `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales`;


-- 2b) Weekly pattern (day-of-week averages)
-- SELECT
--   FORMAT_DATE('%A', order_date) AS weekday,
--   AVG(daily_num_orders) AS avg_daily_orders,
--   AVG(daily_revenue)    AS avg_daily_revenue
-- FROM
--   `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales`
-- GROUP BY
--   weekday
-- ORDER BY
--   weekday;


-- 2c) Monthly trend (total orders and revenue)
-- SELECT
--   FORMAT_DATE('%Y-%m', order_date) AS year_month,
--   SUM(daily_num_orders) AS monthly_orders,
--   SUM(daily_revenue)    AS monthly_revenue
-- FROM
--   `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales`
-- GROUP BY
--   year_month
-- ORDER BY
--   year_month;


-- ============================================================
-- Step 3: Train ARIMA_PLUS model on daily_revenue
-- ============================================================
-- Train on data up to (but not including) 2026-02-01
-- Forecast horizon: 90 days (2026-02-01 to 2026-05-01)

CREATE OR REPLACE MODEL
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_revenue_arima`
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'order_date',
  time_series_data_col      = 'daily_revenue',
  horizon                   = 90,        -- forecast next 90 days
  holiday_region            = 'US'       -- optional: include US holidays
) AS
SELECT
  order_date,
  daily_revenue
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales`
WHERE
  order_date < DATE '2026-02-01';        -- training window


-- ============================================================
-- Step 4: Forecast the next 90 days
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_revenue_forecast_90d` AS
SELECT
  forecast_timestamp AS order_date,
  forecast_value     AS predicted_daily_revenue,
  prediction_interval_lower_bound AS predicted_lower,
  prediction_interval_upper_bound AS predicted_upper
FROM
  ML.FORECAST(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_revenue_arima`,
    STRUCT(90 AS horizon, 0.8 AS confidence_level)
  );


-- ============================================================
-- Step 5: Inspect forecast vs actuals (per day)
-- ============================================================

-- This SELECT lets you see the daily comparison on the holdout period
-- (2026-02-01 to 2026-05-01). Run it manually in the console.

-- WITH joined AS (
--   SELECT
--     a.order_date,
--     a.daily_revenue,
--     f.predicted_daily_revenue,
--     f.predicted_lower,
--     f.predicted_upper,
--     (f.predicted_daily_revenue - a.daily_revenue) AS error
--   FROM
--     `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales` AS a
--   LEFT JOIN
--     `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_revenue_forecast_90d` AS f
--   ON
--     a.order_date = DATE(f.order_date)   -- cast TIMESTAMP to DATE for join
--   WHERE
--     a.order_date >= DATE '2026-02-01'
--     AND a.order_date <= DATE '2026-05-01'
-- )
--
-- SELECT
--   *
-- FROM
--   joined
-- ORDER BY
--   order_date;


-- ============================================================
-- Step 6: Summary error metrics (MAE, RMSE, MAPE)
-- ============================================================

WITH joined AS (
  SELECT
    a.order_date,
    a.daily_revenue,
    f.predicted_daily_revenue,
    (f.predicted_daily_revenue - a.daily_revenue) AS error
  FROM
    `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_sales` AS a
  LEFT JOIN
    `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.daily_revenue_forecast_90d` AS f
  ON
    a.order_date = DATE(f.order_date)
  WHERE
    a.order_date >= DATE '2026-02-01'
    AND a.order_date <= DATE '2026-05-01'
)

SELECT
  AVG(ABS(error)) AS MAE,                               -- Mean Absolute Error
  SQRT(AVG(POW(error, 2))) AS RMSE,                     -- Root Mean Squared Error
  AVG(ABS(error) / NULLIF(daily_revenue, 0)) AS MAPE    -- Mean Absolute Percentage Error
FROM
  joined;
