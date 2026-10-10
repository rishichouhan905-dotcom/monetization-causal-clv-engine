"""Causal Treatment Effect Heterogeneity by CLV Decile.

Evaluates TWFE DiD treatment effects across customer CLV deciles (D01 - D10)
and compares estimated ATTs against true ground-truth ATTs per decile.
"""

from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd

from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH


def analyze_clv_heterogeneity(
    panel_df: Optional[pd.DataFrame] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Evaluates treatment effect heterogeneity across customer CLV deciles.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        db_path: Path to DuckDB database.

    Returns:
        DataFrame containing decile, true_att, estimated_did_att, std_err, p_value, customer_count.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()

    # Fetch CLV scores to assign customer deciles D01 - D10
    conn = duckdb.connect(db_path)
    try:
        clv_df = conn.execute("SELECT customer_id, clv_12m FROM clv_scores").df()
    except Exception:
        clv_df = conn.execute("""
            SELECT customer_id, SUM(line_total) AS clv_12m
            FROM fact_orders
            GROUP BY customer_id
        """).df()
    conn.close()

    if len(clv_df) > 0 and "clv_12m" in clv_df.columns:
        try:
            clv_df["decile"] = pd.qcut(
                clv_df["clv_12m"], q=10, labels=[f"D{i:02d}" for i in range(1, 11)], duplicates="drop"
            )
        except Exception:
            clv_df["decile"] = pd.cut(
                clv_df["clv_12m"], bins=10, labels=[f"D{i:02d}" for i in range(1, 11)]
            )
    else:
        clv_df = pd.DataFrame({"customer_id": df["customer_id"].unique(), "decile": "D01"})

    # Merge deciles into panel DataFrame
    df = df.merge(clv_df[["customer_id", "decile"]], on="customer_id", how="left")
    df["decile"] = df["decile"].fillna("D01")

    decile_results = []
    unique_deciles = sorted(df["decile"].unique())

    print("\n=================== Causal Heterogeneity by CLV Decile ===================")
    print(f"{'Decile':<8} | {'Cust Count':<10} | {'True ATT ($)':<12} | {'TWFE DiD ATT ($)':<16} | {'Std Error':<10} | {'p-value':<10}")
    print("-" * 78)

    for dec in unique_deciles:
        sub = df[df["decile"] == dec].copy()
        cust_cnt = sub["customer_id"].nunique()

        # Compute Ground-Truth ATT for treated customer-weeks post-t0 in this decile
        post_treated_sub = sub[(sub["is_treated"] == 1) & (sub["is_post_period"] == 1)]
        true_att = float(post_treated_sub["true_effect_spend"].mean()) if len(post_treated_sub) > 0 else 0.0

        # Estimate TWFE DiD for this decile
        twfe_res = estimate_twfe_did(panel_df=sub)
        did_att = float(twfe_res["estimated_att"])
        se = float(twfe_res["std_err"])
        pval = float(twfe_res["p_value"])

        decile_results.append({
            "decile": str(dec),
            "customer_count": int(cust_cnt),
            "true_att": np.round(true_att, 4),
            "estimated_did_att": np.round(did_att, 4),
            "std_err": np.round(se, 4),
            "p_value": pval,
        })

        print(f"{str(dec):<8} | {int(cust_cnt):<10} | ${true_att:<11.4f} | ${did_att:<15.4f} | ${se:<9.4f} | {pval:.4f}")

    print("=========================================================================\n")

    res_df = pd.DataFrame(decile_results)

    # Save to DuckDB
    conn = duckdb.connect(db_path)
    conn.execute("CREATE OR REPLACE TABLE causal_clv_heterogeneity AS SELECT * FROM res_df")
    conn.close()

    return res_df


if __name__ == "__main__":
    analyze_clv_heterogeneity()
