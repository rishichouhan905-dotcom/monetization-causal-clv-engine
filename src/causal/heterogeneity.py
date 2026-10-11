"""Causal Treatment Effect Heterogeneity by CLV Decile.

Evaluates TWFE DiD treatment effects across customer CLV deciles (D01 - D10)
and compares estimated ATTs against true ground-truth ATTs per decile.
Requires minimum cell size (>= 10 treated customers per decile) and flags
degenerate cells with degenerate = True and NaN instead of returning 0.0.
"""

from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd

from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH


def analyze_clv_heterogeneity(
    panel_df: Optional[pd.DataFrame] = None,
    min_treated_customers: int = 10,
    outcome_col: str = "observed_weekly_spend_winsor",
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Evaluates treatment effect heterogeneity across customer CLV deciles.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        min_treated_customers: Minimum treated customer count required to fit DiD (default 10).
        outcome_col: Dependent variable column name (default observed_weekly_spend_winsor).
        db_path: Path to DuckDB database.

    Returns:
        DataFrame containing decile, true_att, estimated_did_att, std_err, p_value, customer_count, treated_count, degenerate.
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
                clv_df["clv_12m"].rank(method="first"),
                q=10,
                labels=[f"D{i:02d}" for i in range(1, 11)],
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
    print(f"{'Decile':<8} | {'Total Cust':<10} | {'Treated':<8} | {'True ATT ($)':<12} | {'TWFE DiD ($)':<14} | {'Std Error':<10} | {'p-value':<9} | {'Degenerate'}")
    print("-" * 92)

    for dec in unique_deciles:
        sub = df[df["decile"] == dec].copy()
        cust_cnt = sub["customer_id"].nunique()
        treated_cnt = sub[sub["is_treated"] == 1]["customer_id"].nunique()

        # Compute Ground-Truth ATT for treated customer-weeks post-t0 in this decile
        post_treated_sub = sub[(sub["is_treated"] == 1) & (sub["is_post_period"] == 1)]
        effect_col = "realized_effect_spend_winsor" if "realized_effect_spend_winsor" in sub.columns else "true_effect_spend"
        true_att = float(post_treated_sub[effect_col].mean()) if len(post_treated_sub) > 0 else 0.0

        # Check minimum cell size and treatment variation
        if treated_cnt < min_treated_customers or sub["is_treated"].nunique() < 2:
            did_att = np.nan
            se = np.nan
            pval = np.nan
            is_degenerate = True
        else:
            twfe_res = estimate_twfe_did(panel_df=sub, outcome_col=outcome_col)
            if twfe_res.get("degenerate", False):
                did_att = np.nan
                se = np.nan
                pval = np.nan
                is_degenerate = True
            else:
                did_att = float(twfe_res["estimated_att"])
                se = float(twfe_res["std_err"])
                pval = float(twfe_res["p_value"])
                is_degenerate = False

        decile_results.append({
            "decile": str(dec),
            "customer_count": int(cust_cnt),
            "treated_count": int(treated_cnt),
            "true_att": np.round(true_att, 4),
            "estimated_did_att": np.round(did_att, 4) if not np.isnan(did_att) else np.nan,
            "std_err": np.round(se, 4) if not np.isnan(se) else np.nan,
            "p_value": np.round(pval, 4) if not np.isnan(pval) else np.nan,
            "degenerate": is_degenerate,
        })

        att_str = f"${did_att:<13.4f}" if not np.isnan(did_att) else "NaN          "
        se_str = f"${se:<9.4f}" if not np.isnan(se) else "NaN      "
        pv_str = f"{pval:.4f}" if not np.isnan(pval) else "NaN   "

        print(f"{str(dec):<8} | {int(cust_cnt):<10} | {int(treated_cnt):<8} | ${true_att:<11.4f} | {att_str} | {se_str} | {pv_str:<9} | {is_degenerate}")

    print("=" * 92 + "\n")

    res_df = pd.DataFrame(decile_results)

    # Save to DuckDB
    try:
        conn = duckdb.connect(db_path)
        conn.execute("CREATE OR REPLACE TABLE causal_clv_heterogeneity AS SELECT * FROM res_df")
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to persist heterogeneity table to DuckDB: {e}")

    return res_df


if __name__ == "__main__":
    analyze_clv_heterogeneity()
