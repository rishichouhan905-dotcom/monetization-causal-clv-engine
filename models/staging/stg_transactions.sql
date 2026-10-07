{{ config(materialized='view') }}

/*
  Staging model for UCI Online Retail II raw transaction data.
  
  CANCELLATION & RETURN DECISION DOCUMENTATION:
  - We explicitly KEEP all cancellation and return records (where Quantity < 0 or Invoice starts with 'C') in staging.
  - We flag them with `is_cancelled` (TRUE if Invoice starts with 'C' or Quantity < 0) and `is_return` (TRUE if Quantity < 0).
  - Line total is calculated as Quantity * Unit Price, preserving negative values for returns.
  - Downstream models (dim_customers, dim_products) compute both GROSS and NET spend/revenue metrics,
    allowing analytical applications to inspect total returns while accurately assessing net revenue.
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
        WHEN CAST(Quantity AS INTEGER) < 0 THEN TRUE 
        ELSE FALSE 
    END AS is_return
FROM {{ source('raw', 'raw_online_retail') }}
WHERE "Customer ID" IS NOT NULL
  AND Price > 0
