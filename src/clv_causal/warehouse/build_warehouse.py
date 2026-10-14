"""Star Schema Data Warehouse Builder using dbt-duckdb.

Executes dbt build for the dbt_project directory (staging, intermediate, marts models,
and data quality schema / singular tests) against the DuckDB analytical warehouse.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import duckdb

from clv_causal.config import DB_PATH, PROJECT_ROOT

DBT_PROJECT_DIR: Path = PROJECT_ROOT / "dbt_project"


def build_star_schema() -> None:
    """Executes dbt build for dbt_project and verifies warehouse star schema tables and constraints."""
    print("Step 2: Building Star Schema & SQL Transformations via dbt (dbt-duckdb)...")

    # If SAMPLE=1, ensure raw_online_retail is populated from data/sample/online_retail_II_sample.parquet
    sample_mode = os.environ.get("SAMPLE") == "1"
    if sample_mode:
        sample_path = PROJECT_ROOT / "data" / "sample" / "online_retail_II_sample.parquet"
        if not sample_path.exists():
            from scripts.create_sample import create_sample
            create_sample(sample_size=5000)
        conn = duckdb.connect(DB_PATH)
        conn.execute(f"CREATE OR REPLACE TABLE raw_online_retail AS SELECT * FROM '{sample_path.as_posix()}'")
        row_count = conn.execute("SELECT COUNT(*) FROM raw_online_retail").fetchone()[0]
        conn.close()
        print(f"SAMPLE=1 active: Populated 'raw_online_retail' with {row_count:,} sample rows.")

    dbt_cmd: List[str] = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        "build",
        "--project-dir",
        str(DBT_PROJECT_DIR),
        "--profiles-dir",
        str(DBT_PROJECT_DIR),
    ]
    print(f"Executing dbt command: {' '.join(dbt_cmd)}")

    env_vars = dict(os.environ, DB_PATH=DB_PATH)
    if sample_mode:
        env_vars["SAMPLE"] = "1"
    res = subprocess.run(dbt_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env_vars)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"dbt build failed with exit code {res.returncode}")

    # Schema Data Quality Verification
    conn = duckdb.connect(DB_PATH)
    print("\n--- Running Warehouse Table Verification Assertions ---")

    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    expected_marts = ["dim_customers", "dim_products", "fact_orders", "fact_returns"]
    for table in expected_marts:
        assert table in tables, f"DATA ASSERTION FAILURE: Missing mart table '{table}' in DuckDB!"

    null_cust_count: int = conn.execute("SELECT COUNT(*) FROM fact_orders WHERE customer_id IS NULL").fetchone()[0]
    assert null_cust_count == 0, f"DATA ASSERTION FAILURE: Found {null_cust_count} NULL customer_ids in fact_orders!"
    print("[PASS] Assertion 1: fact_orders customer_id is 100% Non-Null.")

    orphan_cust_count: int = conn.execute("""
        SELECT COUNT(*) FROM fact_orders fo 
        LEFT JOIN dim_customers dc ON fo.customer_id = dc.customer_id 
        WHERE dc.customer_id IS NULL
    """).fetchone()[0]
    assert orphan_cust_count == 0, "DATA ASSERTION FAILURE: Foreign key mismatch in fact_orders!"
    print("[PASS] Assertion 2: Foreign Key integrity between fact_orders and dim_customers validated.")

    print("\nData Warehouse Star Schema successfully constructed and verified via dbt-duckdb!")
    conn.close()


if __name__ == "__main__":
    build_star_schema()
