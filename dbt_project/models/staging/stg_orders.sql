{{ config(materialized='view') }}

/*
  Staging model: stg_orders
  Preserves returns as rows with is_return flag based on C invoice prefix and negative quantity.
*/
SELECT
    CAST(Invoice AS VARCHAR) AS invoice_no,
    CAST(StockCode AS VARCHAR) AS stock_code,
    TRIM(CAST(Description AS VARCHAR)) AS description,
    CAST(Quantity AS INTEGER) AS quantity,
    CAST(InvoiceDate AS TIMESTAMP) AS invoice_date,
    CAST(Price AS DOUBLE) AS unit_price,
    CAST("Customer ID" AS BIGINT) AS customer_id,
    CAST(Country AS VARCHAR) AS country,
    CAST(Quantity AS DOUBLE) * CAST(Price AS DOUBLE) AS line_total,
    CASE 
        WHEN CAST(Invoice AS VARCHAR) LIKE 'C%' OR CAST(Quantity AS INTEGER) < 0 THEN TRUE 
        ELSE FALSE 
    END AS is_cancelled,
    CASE 
        WHEN CAST(Invoice AS VARCHAR) LIKE 'C%' OR CAST(Quantity AS INTEGER) < 0 THEN TRUE 
        ELSE FALSE 
    END AS is_return
FROM {{ source('raw_online_retail', 'raw_online_retail') }}
WHERE "Customer ID" IS NOT NULL
  AND Price > 0
