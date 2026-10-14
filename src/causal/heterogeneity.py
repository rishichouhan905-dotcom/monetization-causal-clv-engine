"""Causal Treatment Effect Heterogeneity by CLV Decile.

Evaluates TWFE DiD treatment effects across customer CLV deciles (D01 - D10)
and compares estimated ATTs against true ground-truth ATTs per decile.
Requires minimum cell size (>= 10 treated customers per decile) and flags
degenerate cells with degenerate = True and NaN instead of returning 0.0.
"""

from typing import Optional
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

    req_cols = ["customer_id", "week_date", "is_treated", "is_post_period", "observed_weekly_spend", outcome_col]
    if "realized_effect_spend_winsor" in panel_df.columns:
        req_cols.append("realized_effect_spend_winsor")
    elif "true_effect_spend" in panel_df.columns:
        req_cols.append("true_effect_spend")
    df = panel_df[req_cols].copy()

    # Define customer deciles D01 - D10 strictly from observed pre-period spend (before t_0)
    pre_spend_df = (
        df[df["is_post_period"] == 0]
        .groupby("customer_id")["observed_weekly_spend"]
        .sum()
        .reset_index()
        .rename(columns={"observed_weekly_spend": "pre_spend"})
    )
    pre_spend_df["decile"] = pd.qcut(
        pre_spend_df["pre_spend"].rank(method="first"),
        q=10,
        labels=[f"D{i:02d}" for i in range(1, 11)],
    )

    # Merge deciles into panel DataFrame
    df = df.merge(pre_spend_df[["customer_id", "decile"]], on="customer_id", how="left")
    df["decile"] = df["decile"].fillna("D01")

    decile_results = []
    unique_deciles = sorted(df["decile"].unique())

    print("\n=================== Causal Heterogeneity by Pre-Period Spend Decile ===================")
    print(f"{'Decile':<8} | {'Total Cust':<10} | {'Treated':<8} | {'True ATT ($)':<12} | {'TWFE DiD ($)':<14} | {'Std Error':<10} | {'p (raw)':<8} | {'p (Bonf)':<8} | {'Degenerate'}")
    print("-" * 105)

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
            pval_bonf = np.nan
            is_degenerate = True
        else:
            twfe_res = estimate_twfe_did(panel_df=sub, outcome_col=outcome_col)
            if twfe_res.get("degenerate", False):
                did_att = np.nan
                se = np.nan
                pval = np.nan
                pval_bonf = np.nan
                is_degenerate = True
            else:
                did_att = float(twfe_res["estimated_att"])
                se = float(twfe_res["std_err"])
                pval = float(twfe_res["p_value"])
                # Bonferroni correction for 10 simultaneous decile hypotheses
                pval_bonf = min(1.0, pval * 10.0)
                is_degenerate = False

        decile_results.append({
            "decile": str(dec),
            "customer_count": int(cust_cnt),
            "treated_count": int(treated_cnt),
            "true_att": np.round(true_att, 4),
            "estimated_did_att": np.round(did_att, 4) if not np.isnan(did_att) else np.nan,
            "std_err": np.round(se, 4) if not np.isnan(se) else np.nan,
            "p_value": np.round(pval, 4) if not np.isnan(pval) else np.nan,
            "p_value_bonferroni": np.round(pval_bonf, 4) if not np.isnan(pval_bonf) else np.nan,
            "degenerate": is_degenerate,
        })

        att_str = f"${did_att:<13.4f}" if not np.isnan(did_att) else "NaN          "
        se_str = f"${se:<9.4f}" if not np.isnan(se) else "NaN      "
        pv_str = f"{pval:.4f}" if not np.isnan(pval) else "NaN   "
        pv_b_str = f"{pval_bonf:.4f}" if not np.isnan(pval_bonf) else "NaN   "

        print(f"{str(dec):<8} | {int(cust_cnt):<10} | {int(treated_cnt):<8} | ${true_att:<11.4f} | {att_str} | {se_str} | {pv_str:<8} | {pv_b_str:<8} | {is_degenerate}")

    print("=" * 105 + "\n")

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
