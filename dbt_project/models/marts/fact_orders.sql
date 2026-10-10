{{ config(materialized='table') }}

/*
  Marts model: fact_orders
  Fact table containing valid non-cancelled purchase order line items.
*/
SELECT
    transaction_id AS order_id,
    invoice_no,
    stock_code,
    customer_id,
    invoice_date,
    quantity,
    unit_price,
    line_total,
    country,
    is_cancelled,
    is_return
FROM {{ ref('int_orders') }}
WHERE NOT is_return
