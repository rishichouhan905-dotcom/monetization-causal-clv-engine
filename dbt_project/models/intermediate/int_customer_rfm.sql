{{ config(materialized='view') }}

/*
  Intermediate model: int_customer_rfm
  Computes customer Recency, Frequency, Monetary Value, and Age (T) for probabilistic CLV.
*/
WITH max_warehouse_date AS (
    SELECT MAX(invoice_date) AS max_date FROM {{ ref('stg_orders') }}
),
customer_orders AS (
    SELECT
        customer_id,
        invoice_no,
        MIN(invoice_date) AS order_date,
        SUM(line_total) AS order_value
    FROM {{ ref('stg_orders') }}
    WHERE NOT is_return
    GROUP BY customer_id, invoice_no
),
customer_summary AS (
    SELECT
        co.customer_id,
        COUNT(DISTINCT co.invoice_no) AS total_orders,
        COUNT(DISTINCT co.invoice_no) - 1 AS frequency,
        MIN(co.order_date) AS first_order_date,
        MAX(co.order_date) AS last_order_date,
        SUM(co.order_value) AS total_monetary,
        AVG(co.order_value) AS avg_monetary,
        CASE 
            WHEN COUNT(DISTINCT co.invoice_no) > 1 THEN 
                (SUM(co.order_value) - MIN(co.order_value)) / NULLIF(COUNT(DISTINCT co.invoice_no) - 1, 0)
            ELSE 0.0 
        END AS monetary_value_repeat
    FROM customer_orders co
    GROUP BY co.customer_id
)
SELECT
    cs.customer_id,
    cs.frequency,
    GREATEST(0.0, CAST(DATE_DIFF('day', cs.first_order_date, cs.last_order_date) AS DOUBLE)) AS recency,
    GREATEST(
        CAST(DATE_DIFF('day', cs.first_order_date, mwd.max_date) AS DOUBLE),
        GREATEST(0.0, CAST(DATE_DIFF('day', cs.first_order_date, cs.last_order_date) AS DOUBLE))
    ) AS T,
    ROUND(cs.avg_monetary, 2) AS monetary_value,
    ROUND(cs.monetary_value_repeat, 2) AS monetary_value_repeat,
    cs.total_orders,
    ROUND(cs.total_monetary, 2) AS total_monetary
FROM customer_summary cs
CROSS JOIN max_warehouse_date mwd
