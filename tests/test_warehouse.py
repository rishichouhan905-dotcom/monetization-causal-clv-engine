import pytest
import os
import duckdb
from clv_causal.config import DB_PATH

def test_warehouse_tables_exist():
    assert os.path.exists(DB_PATH), "Database file does not exist!"
    conn = duckdb.connect(DB_PATH)
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    conn.close()
    
    expected = ["dim_customers", "dim_products", "fact_transactions", "fact_order_events"]
    for t in expected:
        assert t in tables, f"Missing table {t} in warehouse!"

def test_non_null_customer_ids():
    conn = duckdb.connect(DB_PATH)
    null_count = conn.execute("SELECT COUNT(*) FROM fact_transactions WHERE customer_id IS NULL").fetchone()[0]
    conn.close()
    assert null_count == 0, f"Found {null_count} null customer IDs in fact_transactions!"

def test_foreign_key_integrity():
    conn = duckdb.connect(DB_PATH)
    orphan_count = conn.execute("""
        SELECT COUNT(*) FROM fact_transactions ft
        LEFT JOIN dim_customers dc ON ft.customer_id = dc.customer_id
        WHERE dc.customer_id IS NULL
    """).fetchone()[0]
    conn.close()
    assert orphan_count == 0, f"Found {orphan_count} orphan customer records!"
