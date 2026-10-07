import os
import duckdb
import pandas as pd
import numpy as np
from clv_causal.config import DB_PATH

def extract_rfm_matrix(cutoff_date=None):
    """
    Extracts Recency, Frequency, Monetary, and Age (T) per customer from DuckDB warehouse.
    - frequency: number of REPEAT orders (total non-cancelled orders - 1)
    - recency: time between first and last purchase (in days)
    - T (age): time between first purchase and cutoff date (in days)
    - monetary_value_repeat: average spend per repeat purchase
    """
    conn = duckdb.connect(DB_PATH)
    
    if cutoff_date is None:
        max_date_res = conn.execute("SELECT MAX(invoice_date) FROM fact_transactions").fetchone()[0]
        cutoff_date = max_date_res
    
    print(f"Extracting RFM data using cutoff observation date: {cutoff_date}")
    
    query = f"""
    WITH customer_orders AS (
        SELECT
            customer_id,
            invoice_no,
            MIN(invoice_date) AS order_date,
            SUM(line_total) AS order_value
        FROM fact_transactions
        WHERE invoice_date <= '{cutoff_date}'
          AND NOT is_cancelled
        GROUP BY customer_id, invoice_no
    ),
    customer_summary AS (
        SELECT
            customer_id,
            COUNT(DISTINCT invoice_no) AS total_orders,
            COUNT(DISTINCT invoice_no) - 1 AS frequency,
            MIN(order_date) AS first_order_date,
            MAX(order_date) AS last_order_date,
            SUM(order_value) AS total_monetary,
            AVG(order_value) AS avg_monetary,
            -- Calculate monetary_value strictly over repeat orders if frequency > 0
            CASE 
                WHEN COUNT(DISTINCT invoice_no) > 1 THEN 
                    (SUM(order_value) - MIN(order_value)) / NULLIF(COUNT(DISTINCT invoice_no) - 1, 0)
                ELSE 0 
            END AS monetary_value_repeat
        FROM customer_orders
        GROUP BY customer_id
    )
    SELECT
        customer_id,
        frequency,
        CAST(DATE_DIFF('day', first_order_date, last_order_date) AS DOUBLE) AS recency,
        CAST(DATE_DIFF('day', first_order_date, TIMESTAMP '{cutoff_date}') AS DOUBLE) AS T,
        ROUND(avg_monetary, 2) AS monetary_value,
        ROUND(monetary_value_repeat, 2) AS monetary_value_repeat,
        total_orders,
        ROUND(total_monetary, 2) AS total_monetary
    FROM customer_summary;
    """
    
    rfm_df = conn.execute(query).df()
    conn.close()
    
    rfm_df["recency"] = np.maximum(rfm_df["recency"], 0.0)
    rfm_df["T"] = np.maximum(rfm_df["T"], rfm_df["recency"])
    
    print(f"Extracted RFM metrics for {len(rfm_df):,} distinct customers.")
    return rfm_df

def extract_rfm_holdout_split(observation_end_date="2011-06-01", holdout_end_date=None):
    """
    Splits historical transactions into Observation/Calibration Period (<= observation_end_date)
    and Holdout Period (> observation_end_date and <= holdout_end_date) for out-of-sample validation.
    Returns calibration RFM metrics merged with actual holdout repeat purchases & monetary spend.
    """
    conn = duckdb.connect(DB_PATH)
    
    if holdout_end_date is None:
        max_date_res = conn.execute("SELECT MAX(invoice_date) FROM fact_transactions").fetchone()[0]
        holdout_end_date = max_date_res
        
    print(f"Extracting Holdout Split: Calibration <= {observation_end_date} | Holdout <= {holdout_end_date}")
    
    # 1. Calibration RFM
    obs_query = f"""
    WITH customer_orders AS (
        SELECT
            customer_id,
            invoice_no,
            MIN(invoice_date) AS order_date,
            SUM(line_total) AS order_value
        FROM fact_transactions
        WHERE invoice_date <= '{observation_end_date}'
          AND NOT is_cancelled
        GROUP BY customer_id, invoice_no
    ),
    customer_summary AS (
        SELECT
            customer_id,
            COUNT(DISTINCT invoice_no) AS total_orders_obs,
            COUNT(DISTINCT invoice_no) - 1 AS frequency_obs,
            MIN(order_date) AS first_order_date,
            MAX(order_date) AS last_order_date,
            SUM(order_value) AS total_monetary_obs,
            AVG(order_value) AS avg_monetary_obs,
            CASE 
                WHEN COUNT(DISTINCT invoice_no) > 1 THEN 
                    (SUM(order_value) - MIN(order_value)) / NULLIF(COUNT(DISTINCT invoice_no) - 1, 0)
                ELSE 0 
            END AS monetary_value_repeat_obs
        FROM customer_orders
        GROUP BY customer_id
    )
    SELECT
        customer_id,
        frequency_obs AS frequency,
        CAST(DATE_DIFF('day', first_order_date, last_order_date) AS DOUBLE) AS recency,
        CAST(DATE_DIFF('day', first_order_date, TIMESTAMP '{observation_end_date}') AS DOUBLE) AS T,
        ROUND(avg_monetary_obs, 2) AS monetary_value,
        ROUND(monetary_value_repeat_obs, 2) AS monetary_value_repeat,
        ROUND(total_monetary_obs, 2) AS total_monetary
    FROM customer_summary;
    """
    
    obs_df = conn.execute(obs_query).df()
    obs_df["recency"] = np.maximum(obs_df["recency"], 0.0)
    obs_df["T"] = np.maximum(obs_df["T"], obs_df["recency"])
    
    # 2. Holdout Actuals
    holdout_query = f"""
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS actual_holdout_purchases,
        ROUND(SUM(line_total), 2) AS actual_holdout_spend
    FROM fact_transactions
    WHERE invoice_date > '{observation_end_date}'
      AND invoice_date <= '{holdout_end_date}'
      AND NOT is_cancelled
    GROUP BY customer_id;
    """
    
    holdout_df = conn.execute(holdout_query).df()
    conn.close()
    
    merged = pd.merge(obs_df, holdout_df, on='customer_id', how='left')
    merged['actual_holdout_purchases'] = merged['actual_holdout_purchases'].fillna(0).astype(int)
    merged['actual_holdout_spend'] = merged['actual_holdout_spend'].fillna(0.0)
    
    holdout_days = (pd.to_datetime(holdout_end_date) - pd.to_datetime(observation_end_date)).days
    merged['holdout_duration_days'] = holdout_days
    
    print(f"Holdout dataset created: {len(merged):,} calibration customers, {holdout_days} holdout days.")
    return merged

if __name__ == "__main__":
    df = extract_rfm_matrix()
    print(df.head(10))
