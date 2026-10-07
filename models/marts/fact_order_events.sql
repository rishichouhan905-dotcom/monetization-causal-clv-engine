{{ config(materialized='table') }}

/*
  Order Event Fact Model: fact_order_events (Renamed honestly from fact_user_events)
  Tracks transaction-derived order activity events.
*/

WITH base_sessions AS (
    SELECT
        customer_id,
        invoice_date AS event_timestamp,
        invoice_no,
        is_cancelled
    FROM {{ ref('stg_transactions') }}
)
SELECT
    ROW_NUMBER() OVER (ORDER BY event_timestamp, customer_id) AS event_id,
    MD5(CONCAT(CAST(customer_id AS VARCHAR), '_', CAST(event_timestamp AS VARCHAR))) AS session_id,
    customer_id,
    event_timestamp,
    CASE WHEN is_cancelled THEN 'order_cancellation' ELSE 'order_purchase' END AS event_type,
    'web_desktop' AS device_type
FROM base_sessions
