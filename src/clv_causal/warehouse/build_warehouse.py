import os
import sys
import subprocess
import duckdb
from clv_causal.config import DB_PATH, PROJECT_ROOT

def build_star_schema_sql_fallback():
    print("Executing SQL fallback transformations in DuckDB Analytical Warehouse...")
    conn = duckdb.connect(DB_PATH)
    
    # 1. Staging Layer: stg_transactions
    conn.execute("""
    CREATE OR REPLACE VIEW stg_transactions AS
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
        CASE WHEN CAST(Invoice AS VARCHAR) LIKE 'C%' OR CAST(Quantity AS INTEGER) < 0 THEN TRUE ELSE FALSE END AS is_cancelled,
        CASE WHEN CAST(Quantity AS INTEGER) < 0 THEN TRUE ELSE FALSE END AS is_return
    FROM raw_online_retail
    WHERE "Customer ID" IS NOT NULL
      AND Price > 0;
    """)

    # 2. Dimension Table: dim_customers
    conn.execute("""
    CREATE OR REPLACE TABLE dim_customers AS
    SELECT
        customer_id,
        MAX(country) AS country,
        MIN(invoice_date) AS first_purchase_date,
        MAX(invoice_date) AS last_purchase_date,
        COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END) AS total_orders,
        COUNT(DISTINCT CASE WHEN is_cancelled THEN invoice_no END) AS total_return_orders,
        SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END) AS total_items_purchased,
        SUM(CASE WHEN quantity < 0 THEN ABS(quantity) ELSE 0 END) AS total_items_returned,
        ROUND(SUM(CASE WHEN quantity > 0 THEN line_total ELSE 0 END), 2) AS gross_spend,
        ROUND(SUM(CASE WHEN quantity < 0 THEN line_total ELSE 0 END), 2) AS return_spend,
        ROUND(SUM(line_total), 2) AS total_spend,
        ROUND(AVG(CASE WHEN quantity > 0 THEN line_total END), 2) AS avg_line_item_value,
        ROUND(SUM(line_total) / NULLIF(COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END), 0), 2) AS avg_order_value
    FROM stg_transactions
    GROUP BY customer_id;
    """)

    # 3. Dimension Table: dim_products
    conn.execute("""
    CREATE OR REPLACE TABLE dim_products AS
    SELECT
        stock_code,
        MAX(description) AS description,
        ROUND(AVG(unit_price), 2) AS avg_unit_price,
        COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN invoice_no END) AS total_orders,
        SUM(quantity) AS total_quantity_sold,
        ROUND(SUM(line_total), 2) AS total_revenue
    FROM stg_transactions
    GROUP BY stock_code;
    """)

    # 4. Fact Table: fact_transactions
    conn.execute("""
    CREATE OR REPLACE TABLE fact_transactions AS
    SELECT
        ROW_NUMBER() OVER (ORDER BY invoice_date, customer_id, stock_code) AS transaction_id,
        invoice_no,
        stock_code,
        customer_id,
        invoice_date,
        quantity,
        unit_price,
        line_total,
        country,
        is_cancelled,
        is_return
    FROM stg_transactions;
    """)

    # 5. Fact Table: fact_order_events
    conn.execute("""
    CREATE OR REPLACE TABLE fact_order_events AS
    WITH base_sessions AS (
        SELECT
            customer_id,
            invoice_date AS event_timestamp,
            invoice_no,
            is_cancelled
        FROM stg_transactions
    )
    SELECT
        ROW_NUMBER() OVER (ORDER BY event_timestamp, customer_id) AS event_id,
        MD5(CONCAT(CAST(customer_id AS VARCHAR), '_', CAST(event_timestamp AS VARCHAR))) AS session_id,
        customer_id,
        event_timestamp,
        CASE WHEN is_cancelled THEN 'order_cancellation' ELSE 'order_purchase' END AS event_type,
        'web_desktop' AS device_type
    FROM base_sessions;
    """)
    conn.close()

def build_star_schema():
    print("Step 2: Building Star Schema & SQL Transformations via dbt (dbt-duckdb)...")
    
    dbt_cmd = [sys.executable, "-m", "dbt.cli.main", "build", "--profiles-dir", str(PROJECT_ROOT)]
    print(f"Executing dbt command: {' '.join(dbt_cmd)}")
    
    dbt_success = False
    try:
        res = subprocess.run(dbt_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout)
            dbt_success = True
        else:
            print(f"dbt build failed (exit code {res.returncode}):\n{res.stdout}\n{res.stderr}")
            print("Falling back to direct DuckDB SQL model execution...")
            build_star_schema_sql_fallback()
    except Exception as e:
        print(f"dbt build exception ({e}). Falling back to direct DuckDB SQL execution...")
        build_star_schema_sql_fallback()

    # Schema Data Quality Assertions Verification
    conn = duckdb.connect(DB_PATH)
    print("\n--- Running Data Quality & Constraint Verification Assertions ---")
    
    null_cust_count = conn.execute("SELECT COUNT(*) FROM fact_transactions WHERE customer_id IS NULL").fetchone()[0]
    assert null_cust_count == 0, f"DATA ASSERTION FAILURE: Found {null_cust_count} NULL customer_ids in fact_transactions!"
    print("[PASS] Assertion 1: fact_transactions customer_id is 100% Non-Null.")

    orphan_cust_count = conn.execute("""
        SELECT COUNT(*) FROM fact_transactions ft 
        LEFT JOIN dim_customers dc ON ft.customer_id = dc.customer_id 
        WHERE dc.customer_id IS NULL
    """).fetchone()[0]
    assert orphan_cust_count == 0, f"DATA ASSERTION FAILURE: Found {orphan_cust_count} orphan customer records!"
    print("[PASS] Assertion 2: Foreign Key integrity between fact_transactions and dim_customers validated.")

    print("\nData Warehouse Star Schema successfully constructed and verified!")
    conn.close()

if __name__ == "__main__":
    build_star_schema()
