"""Exports small Parquet extracts of warehouse tables to app_data/ for lightweight fallback and demo mode."""

import os
import duckdb

from clv_causal.config import DB_PATH, PROJECT_ROOT


def export_app_data() -> None:
    app_data_dir = PROJECT_ROOT / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DB_PATH, read_only=True)

    tables = [
        "dim_customers",
        "fact_orders",
        "fact_returns",
        "dim_products",
        "pred_clv_summary",
        "decision_segment_summary",
        "clv_holdout_metrics",
    ]

    existing_tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    print(f"Exporting tables to {app_data_dir}...")

    total_bytes = 0
    for tbl in tables:
        if tbl in existing_tables:
            out_file = app_data_dir / f"{tbl}.parquet"
            # Use ZSTD compression to keep file sizes very compact
            conn.execute(f"COPY {tbl} TO '{out_file.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            sz = os.path.getsize(out_file)
            total_bytes += sz
            print(f"  - {tbl:<25} -> {out_file.name:<30} ({sz / (1024*1024):.2f} MB)")
        else:
            print(f"  - {tbl:<25} (SKIPPED: table not found in DuckDB)")

    conn.close()

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal app_data Parquet size: {total_mb:.2f} MB (Constraint: < 20 MB)")
    assert total_mb < 20.0, f"app_data Parquet size {total_mb:.2f} MB exceeds 20 MB limit!"


if __name__ == "__main__":
    export_app_data()
