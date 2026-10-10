{{ config(materialized='view') }}

/*
  Staging model: stg_customers
  Extracts distinct customer profiles and baseline order timestamps.
*/
SELECT
    customer_id,
    MAX(country) AS country,
    MIN(invoice_date) AS first_order_date,
    MAX(invoice_date) AS last_order_date
FROM {{ ref('stg_orders') }}
GROUP BY customer_id
