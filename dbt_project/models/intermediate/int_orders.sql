{{ config(materialized='view') }}

/*
  Intermediate model: int_orders
  Enriches order line-items with gross/return metrics and transaction IDs.
*/
SELECT
    ROW_NUMBER() OVER (ORDER BY invoice_date, customer_id, stock_code) AS transaction_id,
    invoice_no,
    stock_code,
    description,
    customer_id,
    invoice_date,
    quantity,
    unit_price,
    line_total,
    country,
    is_cancelled,
    is_return,
    CASE WHEN NOT is_return THEN line_total ELSE 0.0 END AS gross_line_total,
    CASE WHEN is_return THEN ABS(line_total) ELSE 0.0 END AS return_line_total
FROM {{ ref('stg_orders') }}
