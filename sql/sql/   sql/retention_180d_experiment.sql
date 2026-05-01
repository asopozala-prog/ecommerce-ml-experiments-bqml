-- Retention / Churn Experiment (180-Day Window)
-- Dataset: bigquery-public-data.thelook_ecommerce
-- Target dataset: project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01
--
-- Goal:
--   1) Engineer user-level past behavior features before a cutoff date
--   2) Label users as retained if they place an order in the next 180 days
--   3) Train a logistic regression model in BigQuery ML
--   4) Show that, with this dataset and features, the model has almost no signal
--
-- Cutoff date: 2022-01-01
-- Retention window: [2022-01-01, 2022-07-01)

-- ============================================================
-- Step 1: Build user-level past behavior features (RFM-style)
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_past_features_20220101` AS

WITH past_orders AS (
  SELECT
    o.user_id,
    DATE(o.created_at) AS order_date,
    SUM(oi.sale_price) AS order_total
  FROM
    `bigquery-public-data.thelook_ecommerce.orders` AS o
  JOIN
    `bigquery-public-data.thelook_ecommerce.order_items` AS oi
  ON
    o.order_id = oi.order_id
  WHERE
    -- Only history BEFORE the cutoff date
    o.created_at < TIMESTAMP '2022-01-01'
  GROUP BY
    o.user_id,
    DATE(o.created_at)
)

SELECT
  user_id,
  COUNT(*) AS past_num_orders,
  AVG(order_total) AS past_avg_order_value,
  SUM(order_total) AS past_total_spend,
  -- Recency: days since last order before cutoff
  DATE_DIFF(
    DATE '2022-01-01',     -- cutoff date
    MAX(order_date),       -- last order date before cutoff
    DAY
  ) AS days_since_last_order
FROM
  past_orders
GROUP BY
  user_id;


-- ============================================================
-- Step 2: Build 180-day retention labels
-- ============================================================

-- Users who place at least one order in the 180 days AFTER the cutoff
CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_labels_180d_20220101` AS

WITH future_orders_180d AS (
  SELECT DISTINCT
    user_id
  FROM
    `bigquery-public-data.thelook_ecommerce.orders`
  WHERE
    created_at >= TIMESTAMP '2022-01-01'
    AND created_at < TIMESTAMP_ADD(TIMESTAMP '2022-01-01', INTERVAL 180 DAY)
)

SELECT
  u.id AS user_id,
  IF(f.user_id IS NOT NULL, 1, 0) AS is_retained_180d
FROM
  `bigquery-public-data.thelook_ecommerce.users` AS u
LEFT JOIN
  future_orders_180d AS f
ON
  u.id = f.user_id;


-- ============================================================
-- Step 3: Inspect label distribution (class imbalance)
-- ============================================================
-- Run this SELECT in the BigQuery console after creating the labels.
-- In our exploration, we observed ~7.8% retained, ~92.2% not retained.

SELECT
  is_retained_180d,
  COUNT(*) AS n,
  COUNT(*) / SUM(COUNT(*)) OVER () AS fraction
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_labels_180d_20220101`
GROUP BY
  is_retained_180d;


-- ============================================================
-- Step 4: Combine past features + labels + static attributes
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_features_180d_20220101` AS

SELECT
  upf.user_id,
  upf.past_num_orders,
  upf.past_avg_order_value,
  upf.past_total_spend,
  upf.days_since_last_order,
  u.age,
  u.country,
  u.gender,
  ul.is_retained_180d
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_past_features_20220101` AS upf
JOIN
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_labels_180d_20220101` AS ul
ON
  upf.user_id = ul.user_id
JOIN
  `bigquery-public-data.thelook_ecommerce.users` AS u
ON
  upf.user_id = u.id;


-- Optional sanity check:
-- SELECT * FROM
--   `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_features_180d_20220101`
-- LIMIT 10;


-- ============================================================
-- Step 5: Train a logistic regression retention model (180-day)
-- ============================================================

CREATE OR REPLACE MODEL
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_180d_logreg`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['is_retained_180d'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  auto_class_weights = TRUE
) AS

SELECT
  -- Label
  is_retained_180d,

  -- Features
  past_num_orders,
  past_avg_order_value,
  past_total_spend,
  days_since_last_order,
  age,
  country,
  gender

FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_features_180d_20220101`;


-- ============================================================
-- Step 6: Evaluate the model (showing near-random performance)
-- ============================================================

SELECT
  *
FROM
  ML.EVALUATE(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_180d_logreg`
  );

-- In our experiment, we observed metrics like:
--   precision  ~ 0.08
--   recall     ~ 0.50
--   accuracy   ~ 0.50
--   roc_auc    ~ 0.50
--
-- This indicates almost no predictive power: the probabilities are
-- effectively random with respect to the retention label.


-- ============================================================
-- Step 7: Inspect model weights
-- ============================================================

SELECT
  *
FROM
  ML.WEIGHTS(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.user_retention_180d_logreg`
  )
ORDER BY
  ABS(weight) DESC;

-- The weights tend to be small and do not reveal strong signals,
-- reinforcing the conclusion that, with this dataset and features,
-- 180-day retention is not meaningfully predictable.
