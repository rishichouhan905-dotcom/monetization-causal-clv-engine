"""Causal Model Evaluation & Ground-Truth Comparison Engine.

Executes all causal estimators (Event Study, TWFE DiD, Synthetic Control, Refutations,
and CLV Heterogeneity), compares estimates against ground-truth ATT, exports
results/att_vs_truth.csv, and prints the summary comparison table.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd

from causal.event_study import run_event_study
from causal.heterogeneity import analyze_clv_heterogeneity
from causal.refutations import run_all_refutations, run_placebo_date_refutation
from causal.simulate_policy import simulate_policy_intervention
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH, PROJECT_ROOT


def run_full_causal_evaluation(
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Runs all causal estimators and evaluates against ground-truth ATT.

    Args:
        db_path: Absolute path to DuckDB database.

    Returns:
        Dictionary containing evaluation DataFrame, estimators results, and csv_path.
    """
    print("\n=== Running Comprehensive Causal Estimator Suite & Evaluation ===")

    # 1. Fetch / Generate Step 4 Panel
    panel_df = simulate_policy_intervention(db_path=db_path)

    # Calculate Ground-Truth ATT for treated customer-weeks post-t0
    post_treated = panel_df[(panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)]
    true_att = float(post_treated["true_effect_spend"].mean()) if len(post_treated) > 0 else 0.0

    # 2. Run Estimators
    event_res = run_event_study(panel_df=panel_df, db_path=db_path)
    twfe_res = estimate_twfe_did(panel_df=panel_df, db_path=db_path)
    sc_res = estimate_synthetic_control(panel_df=panel_df, db_path=db_path)
    refute_res = run_all_refutations(panel_df=panel_df)
    het_df = analyze_clv_heterogeneity(panel_df=panel_df, db_path=db_path)

    twfe_att = float(twfe_res["estimated_att"])
    sc_att = float(sc_res["estimated_att"])
    placebo_att = float(refute_res["placebo_date"]["placebo_att"])

    # Build Comparison Summary Table
    comparison_rows = [
        {
            "Estimator_Model": "Ground-Truth ATT (Benchmark)",
            "Estimated_ATT": true_att,
            "True_ATT": true_att,
            "Absolute_Error": 0.0,
            "Absolute_Error_Pct": 0.0,
            "Standard_Error": 0.0,
            "p_value": 0.0,
            "Status": "TRUE BENCHMARK",
        },
        {
            "Estimator_Model": "Two-Way Fixed Effects (TWFE) DiD",
            "Estimated_ATT": twfe_att,
            "True_ATT": true_att,
            "Absolute_Error": abs(twfe_att - true_att),
            "Absolute_Error_Pct": abs(twfe_att - true_att) / max(1e-5, abs(true_att)) * 100.0,
            "Standard_Error": float(twfe_res["std_err"]),
            "p_value": float(twfe_res["p_value"]),
            "Status": "PRIMARY ESTIMATOR",
        },
        {
            "Estimator_Model": "Synthetic Control (Convex Donors)",
            "Estimated_ATT": sc_att,
            "True_ATT": true_att,
            "Absolute_Error": abs(sc_att - true_att),
            "Absolute_Error_Pct": abs(sc_att - true_att) / max(1e-5, abs(true_att)) * 100.0,
            "Standard_Error": float(sc_res["pre_rmspe"]),
            "p_value": float(sc_res["placebo_pval"]),
            "Status": "SYNTHETIC CONTROL",
        },
        {
            "Estimator_Model": "Refutation: Placebo Treatment Date",
            "Estimated_ATT": placebo_att,
            "True_ATT": 0.0,  # Expected true effect under placebo is 0
            "Absolute_Error": abs(placebo_att - 0.0),
            "Absolute_Error_Pct": 0.0,
            "Standard_Error": float(refute_res["placebo_date"]["std_err"]),
            "p_value": float(refute_res["placebo_date"]["p_value"]),
            "Status": "FALSIFICATION DIAGNOSTIC",
        },
    ]

    for _, r in het_df.iterrows():
        dec_att = float(r["estimated_did_att"])
        dec_true = float(r["true_att"])
        comparison_rows.append({
            "Estimator_Model": f"TWFE DiD - Decile {r['decile']}",
            "Estimated_ATT": dec_att,
            "True_ATT": dec_true,
            "Absolute_Error": abs(dec_att - dec_true),
            "Absolute_Error_Pct": abs(dec_att - dec_true) / max(1e-5, abs(dec_true)) * 100.0,
            "Standard_Error": float(r["std_err"]),
            "p_value": float(r["p_value"]),
            "Status": "DECIDILE HETEROGENEITY",
        })

    eval_df = pd.DataFrame(comparison_rows)

    # 3. Export results/att_vs_truth.csv
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "att_vs_truth.csv"
    eval_df.to_csv(csv_path, index=False)
    print(f"Exported evaluation table to: {csv_path}")

    # Save to DuckDB
    conn = duckdb.connect(db_path)
    conn.execute("CREATE OR REPLACE TABLE causal_att_vs_truth AS SELECT * FROM eval_df")
    conn.close()

    # 4. Display Table
    print("\n=================== Causal ATT vs. Ground-Truth Summary ===================")
    disp_df = eval_df[["Estimator_Model", "Estimated_ATT", "True_ATT", "Absolute_Error", "Standard_Error", "p_value"]].copy()
    disp_df["Estimated_ATT"] = disp_df["Estimated_ATT"].map("${:+.4f}".format)
    disp_df["True_ATT"] = disp_df["True_ATT"].map("${:+.4f}".format)
    disp_df["Absolute_Error"] = disp_df["Absolute_Error"].map("${:.4f}".format)
    disp_df["Standard_Error"] = disp_df["Standard_Error"].map("${:.4f}".format)
    disp_df["p_value"] = disp_df["p_value"].map("{:.4f}".format)
    print(disp_df.to_string(index=False))
    print("===========================================================================\n")

    return {
        "evaluation_df": eval_df,
        "true_att": true_att,
        "twfe_att": twfe_att,
        "sc_att": sc_att,
        "csv_path": str(csv_path),
        "event_res": event_res,
        "twfe_res": twfe_res,
        "sc_res": sc_res,
        "refute_res": refute_res,
        "het_df": het_df,
    }


if __name__ == "__main__":
    run_full_causal_evaluation()
