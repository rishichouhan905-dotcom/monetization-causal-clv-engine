{{ config(materialized='view') }}

/*
  Intermediate model: int_customer_week_panel
  Constructs customer-week panel aggregations for causal impact estimation.
*/
WITH base_tx AS (
    SELECT
        customer_id,
        country,
        DATE_TRUNC('week', invoice_date) AS week_date,
        invoice_no,
        quantity,
        line_total,
        is_return
    FROM {{ ref('stg_orders') }}
)
SELECT
    customer_id,
    country,
    week_date,
    SUM(line_total) AS weekly_spend,
    COUNT(DISTINCT invoice_no) AS weekly_orders,
    SUM(quantity) AS weekly_items,
    CASE WHEN country != 'United Kingdom' THEN 1 ELSE 0 END AS is_treated_cohort,
    CASE WHEN week_date >= TIMESTAMP '2010-06-01' THEN 1 ELSE 0 END AS is_post_period
FROM base_tx
GROUP BY customer_id, country, week_date
