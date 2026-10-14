"""Database connection provider for the Streamlit application.

Provides a unified connection interface that reads from the local DuckDB warehouse
(data/retail_warehouse.duckdb) if present, or transparently falls back to in-memory
views over app_data/*.parquet files if the warehouse file is absent.
"""

from pathlib import Path
import duckdb

from clv_causal.config import DB_PATH, PROJECT_ROOT


def get_app_connection() -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection, falling back to app_data/*.parquet in memory if DB_PATH is absent."""
    db_file = Path(DB_PATH)
    if db_file.exists():
        return duckdb.connect(db_file.as_posix(), read_only=True)

    # Fallback to in-memory DuckDB loaded with app_data Parquet files
    conn = duckdb.connect(":memory:")
    app_data_dir = PROJECT_ROOT / "app_data"

    if app_data_dir.exists():
        for parquet_path in sorted(app_data_dir.glob("*.parquet")):
            tbl = parquet_path.stem
            conn.execute(f"CREATE VIEW {tbl} AS SELECT * FROM '{parquet_path.as_posix()}'")

    return conn
