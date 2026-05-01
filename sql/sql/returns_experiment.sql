-- Returns Prediction Experiment (Weak-Signal Case Study)
-- Dataset: bigquery-public-data.thelook_ecommerce
-- Target dataset: project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01
--
-- Goal:
--   1) Label orders as returned / not returned
--   2) Explore base return rates
--   3) Train a logistic regression model in BigQuery ML
--   4) Show that, with these features, returns are essentially not predictable

-- ============================================================
-- Step 1: Build labeled order dataset for returns
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_features` AS

WITH order_base AS (
  SELECT
    o.order_id,
    o.user_id,
    o.status,
    o.created_at,
    u.gender,
    u.country,
    u.age
  FROM
    `bigquery-public-data.thelook_ecommerce.orders` AS o
  JOIN
    `bigquery-public-data.thelook_ecommerce.users` AS u
  ON
    o.user_id = u.id
),

order_monetary AS (
  SELECT
    oi.order_id,
    COUNT(oi.id) AS num_items,
    SUM(oi.sale_price) AS order_total,
    AVG(oi.sale_price) AS avg_item_price
  FROM
    `bigquery-public-data.thelook_ecommerce.order_items` AS oi
  GROUP BY
    oi.order_id
)

SELECT
  ob.order_id,
  ob.user_id,
  ob.status,
  DATE(ob.created_at) AS order_date,
  ob.gender,
  ob.country,
  ob.age,
  om.num_items,
  om.order_total,
  om.avg_item_price,
  -- Label: 1 if the order status is 'Returned', else 0
  IF(ob.status = 'Returned', 1, 0) AS is_returned
FROM
  order_base AS ob
LEFT JOIN
  order_monetary AS om
USING (order_id);


-- ============================================================
-- Step 2: Explore base return rates
-- (Run these SELECTs one at a time in the BigQuery console)
-- ============================================================

-- Overall return rate
-- Expectation from exploration: ~10%

SELECT
  AVG(is_returned) AS overall_return_rate
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_features`;


-- Return rate by weekday
-- Expectation: fairly flat across weekdays (little signal)

SELECT
  FORMAT_DATE('%A', order_date) AS order_weekday,
  COUNT(*) AS n_orders,
  AVG(is_returned) AS return_rate
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_features`
GROUP BY
  order_weekday
ORDER BY
  order_weekday;


-- Return rate by rough order_total buckets
-- Expectation: returned vs non-returned orders have similar average totals

SELECT
  CASE
    WHEN order_total < 50 THEN '< 50'
    WHEN order_total < 100 THEN '50–100'
    WHEN order_total < 200 THEN '100–200'
    ELSE '>= 200'
  END AS order_total_bucket,
  COUNT(*) AS n_orders,
  AVG(is_returned) AS return_rate,
  AVG(order_total) AS avg_order_total
FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_features`
GROUP BY
  order_total_bucket
ORDER BY
  order_total_bucket;


-- ============================================================
-- Step 3: Train a logistic regression model for returns
-- ============================================================

CREATE OR REPLACE MODEL
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_logreg`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['is_returned'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  auto_class_weights = TRUE
) AS

SELECT
  -- Label
  is_returned,

  -- Features (simple set)
  gender,
  country,
  age,
  num_items,
  order_total,
  avg_item_price

FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_features`;


-- ============================================================
-- Step 4: Evaluate the model (shows it has almost no signal)
-- ============================================================

SELECT
  *
FROM
  ML.EVALUATE(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_logreg`
  );


-- ============================================================
-- Step 5: Inspect model weights (they will be small / uninformative)
-- ============================================================

SELECT
  *
FROM
  ML.WEIGHTS(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_returns_logreg`
  )
ORDER BY
  ABS(weight) DESC;
