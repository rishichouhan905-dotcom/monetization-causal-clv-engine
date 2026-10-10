{{ config(materialized='table') }}

/*
  Marts model: dim_products
  Product catalog dimension containing catalog statistics, average unit price, total items sold, and total revenue.
*/
SELECT
    stock_code,
    MAX(description) AS description,
    ROUND(AVG(unit_price), 2) AS avg_unit_price,
    COUNT(DISTINCT CASE WHEN NOT is_return THEN invoice_no END) AS total_orders,
    SUM(quantity) AS total_quantity_sold,
    ROUND(SUM(line_total), 2) AS total_revenue
FROM {{ ref('stg_orders') }}
GROUP BY stock_code
