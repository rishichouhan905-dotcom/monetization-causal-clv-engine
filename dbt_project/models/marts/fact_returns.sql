{{ config(materialized='table') }}

/*
  Marts model: fact_returns
  Fact table specifically tracking order return transactions and return dollar values.
*/
SELECT
    transaction_id AS return_id,
    invoice_no,
    stock_code,
    customer_id,
    invoice_date,
    ABS(quantity) AS quantity_returned,
    unit_price,
    ABS(line_total) AS return_amount,
    country,
    is_cancelled,
    is_return
FROM {{ ref('int_orders') }}
WHERE is_return
