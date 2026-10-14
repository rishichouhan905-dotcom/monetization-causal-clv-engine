"""Exports small Parquet extracts of warehouse tables to app_data/ for lightweight fallback and demo mode."""

import os
from pathlib import Path
import shutil
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb
from clv_causal.config import DB_PATH, PROJECT_ROOT


def export_app_data() -> None:
    app_data_dir = PROJECT_ROOT / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export tables to compact ZSTD Parquet files
    conn = duckdb.connect(DB_PATH, read_only=True)

    tables = [
        "dim_customers",
        "fact_orders",
        "fact_returns",
        "dim_products",
        "pred_clv_summary",
        "clv_scores",
        "clv_holdout_deciles",
        "clv_holdout_metrics",
        "decision_segment_summary",
        "decision_scenarios",
        "causal_att_vs_truth",
        "causal_clv_heterogeneity",
        "causal_treatment_control_diagnostics",
        "causal_experiment_data",
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
            print(f"  - {tbl:<38} -> {out_file.name:<38} ({sz / (1024*1024):.2f} MB)")
        else:
            print(f"  - {tbl:<38} (SKIPPED: table not found in DuckDB)")

    conn.close()

    # 2. Also ensure dbt_run_results.json is copied if target exists
    dbt_target_results = PROJECT_ROOT / "dbt_project" / "target" / "run_results.json"
    if dbt_target_results.exists():
        shutil.copy2(dbt_target_results, app_data_dir / "dbt_run_results.json")
        print("  - Copied dbt_project/target/run_results.json -> app_data/dbt_run_results.json")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal app_data Parquet size: {total_mb:.2f} MB (Constraint: < 25 MB)")
    assert total_mb < 25.0, f"app_data Parquet size {total_mb:.2f} MB exceeds 25 MB limit!"


if __name__ == "__main__":
    export_app_data()
