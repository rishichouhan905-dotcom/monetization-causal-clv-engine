"""Unit tests for DuckDB analytical data warehouse and star schema tables."""

import os
import duckdb
import pytest
from clv_causal.config import DB_PATH


def test_warehouse_tables_exist() -> None:
    """Tests existence of star schema tables in DuckDB analytical warehouse."""
    assert os.path.exists(DB_PATH), "Database file does not exist!"
    conn = duckdb.connect(DB_PATH)
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    conn.close()

    expected = ["dim_customers", "dim_products", "fact_orders", "fact_returns"]
    for t in expected:
        assert t in tables, f"Missing table {t} in warehouse!"


def test_non_null_customer_ids() -> None:
    """Tests that customer_id is 100% non-null in fact_orders."""
    conn = duckdb.connect(DB_PATH)
    null_count = conn.execute("SELECT COUNT(*) FROM fact_orders WHERE customer_id IS NULL").fetchone()[0]
    conn.close()
    assert null_count == 0, f"Found {null_count} null customer IDs in fact_orders!"


def test_foreign_key_integrity() -> None:
    """Tests referential integrity between fact_orders and dim_customers."""
    conn = duckdb.connect(DB_PATH)
    orphan_count = conn.execute("""
        SELECT COUNT(*) FROM fact_orders fo
        LEFT JOIN dim_customers dc ON fo.customer_id = dc.customer_id
        WHERE dc.customer_id IS NULL
    """).fetchone()[0]
    conn.close()
    assert orphan_count == 0, f"Found {orphan_count} orphan customer records!"


def test_raw_warehouse_row_count_and_date_bounds() -> None:
    """Tests that raw_online_retail row count >= 1,000,000 and max date >= 2011-12-01."""
    import pandas as pd
    assert os.path.exists(DB_PATH), "Database file does not exist!"
    conn = duckdb.connect(DB_PATH)

    row_count = conn.execute("SELECT COUNT(*) FROM raw_online_retail").fetchone()[0]

    inv_col = "InvoiceDate" if "InvoiceDate" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "invoice_date"
    max_date_val = conn.execute(f"SELECT MAX({inv_col}) FROM raw_online_retail").fetchone()[0]
    max_date = pd.to_datetime(max_date_val)
    conn.close()

    assert row_count >= 1000000, f"Raw row count ({row_count:,}) is under 1,000,000 requirement!"
    assert max_date >= pd.to_datetime("2011-12-01"), f"Max date ({max_date}) is before 2011-12-01 requirement!"

