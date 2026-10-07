{{ config(materialized='table') }}

/*
  Transaction Fact Model: fact_transactions
  Line-item transactions with validated order totals, timestamps, and return flags.
*/

SELECT
    ROW_NUMBER() OVER (ORDER BY invoice_date, customer_id, stock_code) AS transaction_id,
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
FROM {{ ref('stg_transactions') }}
