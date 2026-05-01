-- Large Order Classification Workflow
-- Dataset: bigquery-public-data.thelook_ecommerce
-- Target dataset: project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01

-- ============================================================
-- Step 1: Build order-level features
-- ============================================================

CREATE OR REPLACE TABLE
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_features` AS
WITH order_level AS (
  SELECT
    o.order_id,
    o.user_id,
    DATE(o.created_at) AS order_date,
    COUNT(oi.id) AS num_items,
    SUM(oi.sale_price) AS order_total,
    AVG(oi.sale_price) AS avg_item_price
  FROM
    `bigquery-public-data.thelook_ecommerce.orders` AS o
  JOIN
    `bigquery-public-data.thelook_ecommerce.order_items` AS oi
  ON
    o.order_id = oi.order_id
  GROUP BY
    o.order_id,
    o.user_id,
    DATE(o.created_at)
)
SELECT
  *
FROM
  order_level;

-- Optional: quick sanity check
-- SELECT * FROM
--   `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_features`
-- LIMIT 10;


-- ============================================================
-- Step 2: Train logistic regression "large order" classifier
-- ============================================================

CREATE OR REPLACE MODEL
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.large_order_classifier_logreg`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['is_large_order'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2
) AS

-- Compute median order_total and label orders relative to it
WITH threshold AS (
  SELECT
    APPROX_QUANTILES(order_total, 101)[OFFSET(50)] AS median_order_total
  FROM
    `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_features`
)

SELECT
  -- Label: 1 if order_total is above median, else 0
  IF(of.order_total > t.median_order_total, 1, 0) AS is_large_order,

  -- Features
  of.num_items,
  of.avg_item_price,
  of.order_total

FROM
  `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.order_features` AS of,
  threshold AS t;


-- ============================================================
-- Step 3: Evaluate the model
-- ============================================================

SELECT
  *
FROM
  ML.EVALUATE(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.large_order_classifier_logreg`
  );

-- ============================================================
-- Step 4: Inspect feature importance via model weights
-- ============================================================

SELECT
  *
FROM
  ML.WEIGHTS(
    MODEL `project-8c6aef20-12f1-49aa-805.ml_workshop_e_commerce01.large_order_classifier_logreg`
  )
ORDER BY
  ABS(weight) DESC;
