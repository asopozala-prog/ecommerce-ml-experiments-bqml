# ecommerce-ml-experiments-bqml
Practical e-commerce machine learning experiments in BigQuery ML: large-order classification, returns analysis, and retention modeling.
# Large Order Classification and E‑Commerce ML Experiments (BigQuery ML)

This repository contains practical **e‑commerce machine learning experiments** built on top of the public dataset:

- `bigquery-public-data.thelook_ecommerce`

The focus is on **BigQuery ML** and on the *process* of exploring whether a business problem is actually predictable.

## Current experiments

### 1. Large Order Classification (successful)

We train a binary classifier to predict whether an order is **"large"** based on its characteristics.

- Target: `is_large_order`  
  Defined as: order total is **above the median** order total across all orders.

- Features:
  - `num_items` – number of items in the order
  - `avg_item_price` – average item sale price
  - `order_total` – total value of the order

- Workflow:
  1. Build order‑level features from `orders` + `order_items`
  2. Compute the median `order_total`
  3. Train a `LOGISTIC_REG` model in BigQuery ML
  4. Evaluate with `ML.EVALUATE`
  5. Inspect feature importance with `ML.WEIGHTS`

This experiment produces a **strong model** with high precision/recall and clear, interpretable feature weights.

The full SQL for this workflow is in:

- `sql/large_order_workflow.sql`

### 2. Returns and Retention (exploratory / low signal)

We also explored:

- **Returns prediction** – predicting whether an order will be returned
- **User retention** – predicting whether a user will place another order
  within 60 or 180 days after a cutoff date

Key findings:

- The overall **return rate** is ~10%, and with the available features
  (gender, number of items, order total, average price, etc.) there is
  almost **no predictive signal** (ROC AUC ≈ 0.5).
- For **retention**, we engineered user-level RFM-style features:
  - `past_num_orders`
  - `past_avg_order_value`
  - `past_total_spend`
  - `days_since_last_order`
- Even with a 180‑day window and these features, the resulting model also
  showed **no meaningful lift** (ROC AUC ≈ 0.5).

These “failed” experiments are **intentional and documented** here, because
they illustrate an important real‑world lesson:

> Not every seemingly good business question is actually predictable with the available data and features.

## Repository structure

Planned structure:
```text
.
├── README.md                      # This file
├── LICENSE                        # MIT license
└── sql/
    └── large_order_workflow.sql   # End-to-end large-order classification workflow
   ### 3. Daily Demand Forecasting (ARIMA_PLUS)

   We aggregate the thelook_ecommerce orders to a daily revenue time series,
   then train an `ARIMA_PLUS` model in BigQuery ML to forecast the next 90 days.

   - Target: `daily_revenue`
   - Training data: 2019-01-04 to 2026-01-31
   - Forecast horizon: 90 days (2026-02-01 to 2026-05-01)

   Results:
   - MAE ≈ 7.2k, RMSE ≈ 18.6k, MAPE ≈ 25%
   - The model tracks the general trend well but underestimates large
     end-of-month promotional spikes in April 2026.

   This experiment demonstrates both:
   - How to use `ARIMA_PLUS` for demand forecasting, and
   - The importance of external features (e.g., promotions) when
     forecasting sharp spikes in revenue.
