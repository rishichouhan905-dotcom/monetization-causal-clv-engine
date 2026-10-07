{{ config(materialized='table') }}

/*
  Customer Dimension Model: dim_customers
  Aggregates customer lifetime metrics including gross spend, returns, and net spend.
*/

SELECT
    customer_id,
    MAX(country) AS country,
    MIN(invoice_date) AS first_purchase_date,
    MAX(invoice_date) AS last_purchase_date,
    COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END) AS total_orders,
    COUNT(DISTINCT CASE WHEN is_cancelled THEN invoice_no END) AS total_return_orders,
    SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) AS total_items_purchased,
    SUM(CASE WHEN quantity < 0 THEN ABS(quantity) ELSE 0 END) AS total_items_returned,
    ROUND(SUM(CASE WHEN quantity > 0 THEN line_total ELSE 0 END), 2) AS gross_spend,
    ROUND(SUM(CASE WHEN quantity < 0 THEN line_total ELSE 0 END), 2) AS return_spend,
    ROUND(SUM(line_total), 2) AS total_spend,
    ROUND(AVG(CASE WHEN quantity > 0 THEN line_total END), 2) AS avg_line_item_value,
    ROUND(SUM(line_total) / NULLIF(COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END), 0), 2) AS avg_order_value
FROM {{ ref('stg_transactions') }}
GROUP BY customer_id
