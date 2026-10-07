{{ config(materialized='table') }}

/*
  Product Dimension Model: dim_products
  Aggregates product catalog stats, quantities sold/returned, and net revenue.
*/

SELECT
    stock_code,
    MAX(description) AS description,
    ROUND(AVG(unit_price), 2) AS avg_unit_price,
    COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END) AS total_orders,
    SUM(quantity) AS total_quantity_sold,
    ROUND(SUM(line_total), 2) AS total_revenue
FROM {{ ref('stg_transactions') }}
GROUP BY stock_code
