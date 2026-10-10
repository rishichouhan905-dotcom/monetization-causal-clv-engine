"""RFM Matrix Extraction and Time-Based Holdout Dataset Builder.

Extracts Recency, Frequency, Monetary Value (repeat-only spend), and Customer Age (T)
from the DuckDB analytical warehouse int_customer_rfm and fact_orders tables.
Implements a time-based split: fit on first calibration period, evaluate on holdout period.
"""

from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from clv_causal.config import CALIBRATION_END_DATE, DB_PATH, HOLDOUT_END_DATE


def extract_rfm_matrix(cutoff_date: Optional[str] = None) -> pd.DataFrame:
    """Extracts Recency, Frequency, Monetary, and Age (T) per customer from int_customer_rfm table.

    Args:
        cutoff_date: Optional ISO date string cutoff. If None, queries full int_customer_rfm.

    Returns:
        DataFrame containing customer_id, frequency, recency, T, monetary_value,
        monetary_value_repeat, total_orders, and total_monetary.
    """
    conn = duckdb.connect(DB_PATH)

    if cutoff_date is not None:
        query = f"""
        WITH customer_orders AS (
            SELECT
                customer_id,
                invoice_no,
                MIN(invoice_date) AS order_date,
                SUM(line_total) AS order_value
            FROM fact_orders
            WHERE invoice_date <= '{cutoff_date}'
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
                CASE 
                    WHEN COUNT(DISTINCT invoice_no) > 1 THEN 
                        (SUM(order_value) - MIN(order_value)) / NULLIF(COUNT(DISTINCT invoice_no) - 1, 0)
                    ELSE 0.0 
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
        rfm_df: pd.DataFrame = conn.execute(query).df()
    else:
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        if "int_customer_rfm" in tables:
            rfm_df = conn.execute("SELECT * FROM int_customer_rfm").df()
        else:
            max_date_res = conn.execute("SELECT MAX(invoice_date) FROM fact_orders").fetchone()[0]
            cutoff_date = str(max_date_res)
            return extract_rfm_matrix(cutoff_date=cutoff_date)

    conn.close()

    rfm_df["recency"] = np.maximum(rfm_df["recency"], 0.0)
    rfm_df["T"] = np.maximum(rfm_df["T"], rfm_df["recency"])

    print(f"Extracted RFM metrics for {len(rfm_df):,} distinct customers.")
    return rfm_df


def extract_rfm_holdout_split(
    calibration_end_date: Optional[str] = CALIBRATION_END_DATE,
    holdout_end_date: Optional[str] = HOLDOUT_END_DATE,
) -> pd.DataFrame:
    """Splits transactions into Calibration period and Holdout period based on time.

    Args:
        calibration_end_date: Cutoff date for fitting (e.g. '2011-06-01' or '2010-12-01').
        holdout_end_date: Cutoff date for evaluation (e.g. '2011-12-09').

    Returns:
        DataFrame merging calibration period RFM metrics with actual holdout purchases and spend.
    """
    conn = duckdb.connect(DB_PATH)

    cal_date: str = calibration_end_date or CALIBRATION_END_DATE

    if holdout_end_date is None:
        max_date_res = conn.execute("SELECT MAX(invoice_date) FROM fact_orders").fetchone()[0]
        holdout_date = str(max_date_res)
    else:
        holdout_date = holdout_end_date

    # Check if cal_date yields customers; if 0, auto-adjust to dataset midpoint
    min_date = conn.execute("SELECT MIN(invoice_date) FROM fact_orders").fetchone()[0]
    max_date = conn.execute("SELECT MAX(invoice_date) FROM fact_orders").fetchone()[0]

    cnt_cal = conn.execute(f"SELECT COUNT(*) FROM fact_orders WHERE invoice_date <= '{cal_date}'").fetchone()[0]
    if cnt_cal == 0 and min_date is not None and max_date is not None:
        midpoint = min_date + (max_date - min_date) / 2
        cal_date = midpoint.strftime("%Y-%m-%d")
        print(f"Notice: Adjusted calibration cutoff date to dataset midpoint: {cal_date}")

    print(f"Extracting Time-Based Holdout Split: Calibration <= {cal_date} | Holdout > {cal_date} AND <= {holdout_date}")

    obs_query = f"""
    WITH customer_orders AS (
        SELECT
            customer_id,
            invoice_no,
            MIN(invoice_date) AS order_date,
            SUM(line_total) AS order_value
        FROM fact_orders
        WHERE invoice_date <= '{cal_date}'
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
                ELSE 0.0 
            END AS monetary_value_repeat_obs
        FROM customer_orders
        GROUP BY customer_id
    )
    SELECT
        customer_id,
        frequency_obs AS frequency,
        CAST(DATE_DIFF('day', first_order_date, last_order_date) AS DOUBLE) AS recency,
        CAST(DATE_DIFF('day', first_order_date, TIMESTAMP '{cal_date}') AS DOUBLE) AS T,
        ROUND(avg_monetary_obs, 2) AS monetary_value,
        ROUND(monetary_value_repeat_obs, 2) AS monetary_value_repeat,
        ROUND(total_monetary_obs, 2) AS total_monetary,
        total_orders_obs AS total_orders
    FROM customer_summary;
    """

    obs_df: pd.DataFrame = conn.execute(obs_query).df()
    obs_df["recency"] = np.maximum(obs_df["recency"], 0.0)
    obs_df["T"] = np.maximum(obs_df["T"], obs_df["recency"])

    holdout_query = f"""
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS actual_holdout_purchases,
        ROUND(SUM(line_total), 2) AS actual_holdout_spend
    FROM fact_orders
    WHERE invoice_date > '{cal_date}'
      AND invoice_date <= '{holdout_date}'
    GROUP BY customer_id;
    """

    holdout_df: pd.DataFrame = conn.execute(holdout_query).df()
    conn.close()

    merged = pd.merge(obs_df, holdout_df, on="customer_id", how="left")
    merged["actual_holdout_purchases"] = merged["actual_holdout_purchases"].fillna(0).astype(int)
    merged["actual_holdout_spend"] = merged["actual_holdout_spend"].fillna(0.0)

    holdout_days = (pd.to_datetime(holdout_date) - pd.to_datetime(cal_date)).days
    merged["holdout_duration_days"] = holdout_days
    merged["calibration_end_date"] = cal_date
    merged["holdout_end_date"] = holdout_date

    print(f"Time-based holdout dataset created: {len(merged):,} calibration customers, {holdout_days} holdout days.")
    return merged


if __name__ == "__main__":
    df = extract_rfm_matrix()
    print(df.head(10))
