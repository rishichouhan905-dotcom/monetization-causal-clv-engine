/*
  Singular Test: assert_net_revenue_equals_gross_minus_returns
  Verifies that for every customer in dim_customers, net spend (total_spend)
  equals gross_spend minus return_spend within floating point rounding threshold ($0.01).
  Returns any failing customer records.
*/
SELECT
    customer_id,
    gross_spend,
    return_spend,
    total_spend,
    ROUND(gross_spend - return_spend, 2) AS calculated_net_spend
FROM {{ ref('dim_customers') }}
WHERE ABS(total_spend - ROUND(gross_spend - return_spend, 2)) > 0.01
