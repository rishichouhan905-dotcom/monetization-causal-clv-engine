"""Causal Model Evaluation & Ground-Truth Comparison Engine.

Executes all causal estimators (Event Study, TWFE DiD, Synthetic Control, Refutations,
and CLV Heterogeneity), compares estimates against ground-truth ATT for identical
outcomes (weekly_orders, weekly_spend_winsor, active_last_90d) overall and by quintile
(Q1-Q4 pooled, Q5), exports results/att_vs_truth.csv, and prints summary comparison tables.
"""

from typing import Any, Dict, List, Optional
import duckdb
import numpy as np
import pandas as pd

from causal.event_study import run_event_study
from causal.heterogeneity import analyze_clv_heterogeneity
from causal.refutations import run_all_refutations
from causal.simulate_policy import (
    get_ground_truth_effects,
    simulate_policy_intervention,
)
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH, PROJECT_ROOT


def run_full_causal_evaluation(
    db_path: str = DB_PATH,
    random_seed: int = 42,
    churn_scenario: str = "baseline",
    top_spend_churn_cumulative: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs all causal estimators and evaluates against ground-truth ATT across outcomes and cohorts.

    Args:
        db_path: Absolute path to DuckDB database.
        random_seed: Random seed for simulation and estimators.
        churn_scenario: Simulation churn scenario ('baseline' or 'hidden_degradation').
        top_spend_churn_cumulative: Cumulative churn probability over post-period for top spenders.

    Returns:
        Dictionary containing evaluation DataFrame, estimator results, and csv_path.
    """
    if top_spend_churn_cumulative is None:
        top_spend_churn_cumulative = 0.40 if churn_scenario == "hidden_degradation" else 0.20

    print("\n================================================================================")
    print(f"      STARTING CAUSAL EVALUATION & BENCHMARKING (SCENARIO: {churn_scenario.upper()})    ")
    print("================================================================================")

    # 1. Generate & Persist Main Panel for this Scenario
    panel_df = simulate_policy_intervention(
        db_path=db_path,
        churn_scenario=churn_scenario,
        top_spend_churn_cumulative=top_spend_churn_cumulative,
        random_seed=random_seed,
    )

    tr_post = (panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)
    q14_mask = tr_post & (panel_df["spend_quintile"].isin([1, 2, 3, 4]))
    q5_mask = tr_post & (panel_df["spend_quintile"] == 5)

    gt_effects = get_ground_truth_effects(panel_df)

    # Sub-panels for cohort evaluations (subset columns to reduce memory)
    sub_cols = [
        "customer_id",
        "week_date",
        "is_treated",
        "is_post_period",
        "observed_weekly_orders",
        "observed_weekly_spend_winsor",
        "observed_active_last_90d",
    ]
    sub_q14 = panel_df.loc[panel_df["spend_quintile"].isin([1, 2, 3, 4]), sub_cols].copy()
    sub_q5 = panel_df.loc[panel_df["spend_quintile"] == 5, sub_cols].copy()

    # 2. Run Estimators across Outcomes & Cohorts
    rows: List[Dict[str, Any]] = []

    # --- Benchmark: Ground Truth Rows ---
    for cohort_name, mask_label in [
        ("Overall (All Treated)", "overall"),
        ("Q1-Q4 Pooled", "q1_to_q4_pooled"),
        ("Quintile 5 (Top Spend)", "q5"),
    ]:
        rows.append({
            "Estimator_Model": "Ground-Truth (Realized)",
            "Cohort": cohort_name,
            "Outcome_Metric": "weekly_orders",
            "Estimated_ATT": gt_effects[mask_label]["realized_orders"],
            "True_ATT": gt_effects[mask_label]["realized_orders"],
            "Absolute_Error": 0.0,
            "Standard_Error": 0.0,
            "ci_lower": gt_effects[mask_label]["realized_orders"],
            "ci_upper": gt_effects[mask_label]["realized_orders"],
            "p_value": 0.0,
            "Status": "BENCHMARK_TRUTH",
        })
        rows.append({
            "Estimator_Model": "Ground-Truth (Expected)",
            "Cohort": cohort_name,
            "Outcome_Metric": "weekly_orders",
            "Estimated_ATT": gt_effects[mask_label]["expected_orders"],
            "True_ATT": gt_effects[mask_label]["realized_orders"],
            "Absolute_Error": abs(gt_effects[mask_label]["expected_orders"] - gt_effects[mask_label]["realized_orders"]),
            "Standard_Error": 0.0,
            "ci_lower": gt_effects[mask_label]["expected_orders"],
            "ci_upper": gt_effects[mask_label]["expected_orders"],
            "p_value": 0.0,
            "Status": "EXPECTED_THEORY",
        })
        rows.append({
            "Estimator_Model": "Ground-Truth (Realized)",
            "Cohort": cohort_name,
            "Outcome_Metric": "weekly_spend_winsor",
            "Estimated_ATT": gt_effects[mask_label]["realized_spend_winsor"],
            "True_ATT": gt_effects[mask_label]["realized_spend_winsor"],
            "Absolute_Error": 0.0,
            "Standard_Error": 0.0,
            "ci_lower": gt_effects[mask_label]["realized_spend_winsor"],
            "ci_upper": gt_effects[mask_label]["realized_spend_winsor"],
            "p_value": 0.0,
            "Status": "BENCHMARK_TRUTH",
        })
        rows.append({
            "Estimator_Model": "Ground-Truth (Expected)",
            "Cohort": cohort_name,
            "Outcome_Metric": "weekly_spend_winsor",
            "Estimated_ATT": gt_effects[mask_label]["expected_spend_winsor"],
            "True_ATT": gt_effects[mask_label]["realized_spend_winsor"],
            "Absolute_Error": abs(gt_effects[mask_label]["expected_spend_winsor"] - gt_effects[mask_label]["realized_spend_winsor"]),
            "Standard_Error": 0.0,
            "ci_lower": gt_effects[mask_label]["expected_spend_winsor"],
            "ci_upper": gt_effects[mask_label]["expected_spend_winsor"],
            "p_value": 0.0,
            "Status": "EXPECTED_THEORY",
        })
        rows.append({
            "Estimator_Model": "Ground-Truth (Realized)",
            "Cohort": cohort_name,
            "Outcome_Metric": "active_last_90d",
            "Estimated_ATT": gt_effects[mask_label]["realized_active_90d"],
            "True_ATT": gt_effects[mask_label]["realized_active_90d"],
            "Absolute_Error": 0.0,
            "Standard_Error": 0.0,
            "ci_lower": gt_effects[mask_label]["realized_active_90d"],
            "ci_upper": gt_effects[mask_label]["realized_active_90d"],
            "p_value": 0.0,
            "Status": "BENCHMARK_TRUTH",
        })

    # --- TWFE DiD Models ---
    did_targets = [
        ("Overall (All Treated)", panel_df, "overall"),
        ("Q1-Q4 Pooled", sub_q14, "q1_to_q4_pooled"),
        ("Quintile 5 (Top Spend)", sub_q5, "q5"),
    ]

    for cohort_name, p_sub, mask_label in did_targets:
        for out_col, metric_name, true_metric_key in [
            ("observed_weekly_orders", "weekly_orders", "realized_orders"),
            ("observed_weekly_spend_winsor", "weekly_spend_winsor", "realized_spend_winsor"),
            ("observed_active_last_90d", "active_last_90d", "realized_active_90d"),
        ]:
            true_val = gt_effects[mask_label][true_metric_key]
            did_res = estimate_twfe_did(panel_df=p_sub, outcome_col=out_col)
            est_att = float(did_res["estimated_att"])
            se = float(did_res["std_err"])
            pval = float(did_res["p_value"])
            ci_l = float(did_res["ci_lower"])
            ci_u = float(did_res["ci_upper"])

            rows.append({
                "Estimator_Model": "TWFE DiD (PanelOLS)",
                "Cohort": cohort_name,
                "Outcome_Metric": metric_name,
                "Estimated_ATT": est_att,
                "True_ATT": true_val,
                "Absolute_Error": abs(est_att - true_val),
                "Standard_Error": se,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "p_value": pval,
                "Status": "PRIMARY_ESTIMATOR",
            })

    del sub_q14, sub_q5
    import gc
    gc.collect()

    # --- Synthetic Control Models (50 Donor Cohorts) ---
    for out_col, metric_name, true_metric_key in [
        ("observed_weekly_spend_winsor", "weekly_spend_winsor", "realized_spend_winsor"),
        ("observed_weekly_orders", "weekly_orders", "realized_orders"),
    ]:
        true_val = gt_effects["overall"][true_metric_key]
        sc_res = estimate_synthetic_control(
            panel_df=panel_df, outcome_col=out_col, n_donors=50, random_seed=random_seed
        )
        sc_att = float(sc_res["estimated_att"])
        rows.append({
            "Estimator_Model": "Synthetic Control (50 Donors)",
            "Cohort": "Overall (All Treated)",
            "Outcome_Metric": metric_name,
            "Estimated_ATT": sc_att,
            "True_ATT": true_val,
            "Absolute_Error": abs(sc_att - true_val),
            "Standard_Error": float(sc_res["pre_rmspe"]),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": float(sc_res["placebo_pval"]),
            "Status": "SYNTHETIC_CONTROL",
        })

    # --- Event Study (Dynamic Lead/Lag) ---
    es_plot_name = f"event_study_plot_{churn_scenario}.png"
    es_res = run_event_study(
        panel_df=panel_df,
        outcome_col="observed_weekly_spend_winsor",
        plot_filename=es_plot_name,
    )
    post_es = es_res["event_study_df"][es_res["event_study_df"]["is_post"] == 1]
    avg_post_es = float(es_res["mean_post_att"])
    true_spend_win = gt_effects["overall"]["realized_spend_winsor"]
    post_lin_se = float(es_res["mean_post_se"])

    rows.append({
        "Estimator_Model": "Event Study (Dynamic Mean)",
        "Cohort": "Overall (All Treated)",
        "Outcome_Metric": "weekly_spend_winsor",
        "Estimated_ATT": avg_post_es,
        "True_ATT": true_spend_win,
        "Absolute_Error": abs(avg_post_es - true_spend_win),
        "Standard_Error": post_lin_se,
        "ci_lower": avg_post_es - 1.96 * post_lin_se,
        "ci_upper": avg_post_es + 1.96 * post_lin_se,
        "p_value": float(es_res["joint_test_pval"]),
        "Status": "DYNAMIC_EVENT_STUDY",
    })

    # --- Refutations: Placebo Treatment Date & Leave-One-Donor-Out ---
    gc.collect()
    ref_res = run_all_refutations(panel_df=panel_df)
    placebo_att = float(ref_res["placebo_date"]["placebo_att"])
    rows.append({
        "Estimator_Model": "Refutation: Placebo Rollout Date",
        "Cohort": "Overall (Pre-Period Placebo)",
        "Outcome_Metric": "weekly_spend_winsor",
        "Estimated_ATT": placebo_att,
        "True_ATT": 0.0,
        "Absolute_Error": abs(placebo_att - 0.0),
        "Standard_Error": float(ref_res["placebo_date"]["std_err"]),
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": float(ref_res["placebo_date"]["p_value"]),
        "Status": "FALSIFICATION_DIAGNOSTIC",
    })

    loo_mean = float(ref_res["leave_one_out"]["mean_att"])
    rows.append({
        "Estimator_Model": "Refutation: Leave-One-Donor-Out SC",
        "Cohort": "Overall (50 LOO Runs)",
        "Outcome_Metric": "weekly_spend_winsor",
        "Estimated_ATT": loo_mean,
        "True_ATT": true_spend_win,
        "Absolute_Error": abs(loo_mean - true_spend_win),
        "Standard_Error": float(ref_res["leave_one_out"]["std_att"]),
        "ci_lower": float(ref_res["leave_one_out"]["min_att"]),
        "ci_upper": float(ref_res["leave_one_out"]["max_att"]),
        "p_value": 0.0,
        "Status": "SENSITIVITY_DIAGNOSTIC",
    })

    # --- CLV Decile Heterogeneity (D01 - D10) ---
    gc.collect()
    het_df = analyze_clv_heterogeneity(panel_df=panel_df)
    for _, r in het_df.iterrows():
        dec_att = float(r["estimated_did_att"]) if not np.isnan(r["estimated_did_att"]) else np.nan
        dec_true = float(r["true_att"])
        dec_se = float(r["std_err"]) if not np.isnan(r["std_err"]) else np.nan
        dec_pval = float(r["p_value"]) if not np.isnan(r["p_value"]) else np.nan
        err = abs(dec_att - dec_true) if not np.isnan(dec_att) else np.nan

        rows.append({
            "Estimator_Model": f"TWFE DiD - Decile {r['decile']}",
            "Cohort": f"CLV Decile {r['decile']} (N={int(r['customer_count'])})",
            "Outcome_Metric": "weekly_spend_winsor",
            "Estimated_ATT": dec_att,
            "True_ATT": dec_true,
            "Absolute_Error": err,
            "Standard_Error": dec_se,
            "ci_lower": dec_att - 1.96 * dec_se if not np.isnan(dec_att) and not np.isnan(dec_se) else np.nan,
            "ci_upper": dec_att + 1.96 * dec_se if not np.isnan(dec_att) and not np.isnan(dec_se) else np.nan,
            "p_value": dec_pval,
            "Status": "DEGENERATE" if r["degenerate"] else "DECILE_HETEROGENEITY",
        })

    eval_df = pd.DataFrame(rows)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Save Scenario-Specific Event Study CSV
    es_csv_path = results_dir / f"event_study_{churn_scenario}.csv"
    es_res["event_study_df"].to_csv(es_csv_path, index=False)

    # 4. Save Scenario-Specific Decile Heterogeneity CSV
    het_csv_path = results_dir / f"heterogeneity_{churn_scenario}.csv"
    het_df.to_csv(het_csv_path, index=False)

    # 5. Save Scenario-Specific Benchmark Evaluation CSV
    scen_csv_path = results_dir / f"att_vs_truth_{churn_scenario}.csv"
    eval_df.to_csv(scen_csv_path, index=False)
    print(f"\n[OK] Exported {churn_scenario} evaluation table to: {scen_csv_path}")

    # If baseline scenario, also write default att_vs_truth.csv and DuckDB table
    if churn_scenario == "baseline":
        default_csv = results_dir / "att_vs_truth.csv"
        eval_df.to_csv(default_csv, index=False)
        try:
            conn = duckdb.connect(db_path)
            conn.execute("CREATE OR REPLACE TABLE causal_att_vs_truth AS SELECT * FROM eval_df")
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to save causal_att_vs_truth to DuckDB: {e}")

    # 6. Display Formatted Summary Table
    print("\n" + "=" * 105)
    print(f"          CAUSAL ESTIMATORS VS. GROUND TRUTH ATT SUMMARY (SCENARIO: {churn_scenario.upper()})           ")
    print("=" * 105)
    print(f"{'Estimator / Model':<35} | {'Cohort':<22} | {'Outcome':<18} | {'Est ATT':<10} | {'True ATT':<10} | {'Abs Err':<9} | {'SE':<8} | {'p-val'}")
    print("-" * 105)

    key_sub = eval_df[
        eval_df["Status"].isin(["BENCHMARK_TRUTH", "EXPECTED_THEORY", "PRIMARY_ESTIMATOR", "SYNTHETIC_CONTROL", "DYNAMIC_EVENT_STUDY", "FALSIFICATION_DIAGNOSTIC"])
    ]

    for _, r in key_sub.iterrows():
        est_s = f"{r['Estimated_ATT']:+.4f}" if not np.isnan(r["Estimated_ATT"]) else "NaN"
        tru_s = f"{r['True_ATT']:+.4f}" if not np.isnan(r["True_ATT"]) else "NaN"
        err_s = f"{r['Absolute_Error']:.4f}" if not np.isnan(r["Absolute_Error"]) else "NaN"
        se_s = f"{r['Standard_Error']:.4f}" if not np.isnan(r["Standard_Error"]) else "NaN"
        pv_s = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "NaN"

        print(f"{r['Estimator_Model']:<35} | {r['Cohort']:<22} | {r['Outcome_Metric']:<18} | {est_s:<10} | {tru_s:<10} | {err_s:<9} | {se_s:<8} | {pv_s}")

    print("=" * 105 + "\n")

    return {
        "evaluation_df": eval_df,
        "csv_path": str(scen_csv_path),
        "gt_effects": gt_effects,
        "event_res": es_res,
        "sc_res": sc_res,
        "ref_res": ref_res,
        "het_df": het_df,
    }


def run_both_causal_scenarios(db_path: str = DB_PATH, random_seed: int = 42) -> Dict[str, Any]:
    """Runs full causal evaluation for BOTH baseline and hidden_degradation scenarios.

    Saves:
        - results/att_vs_truth_baseline.csv & results/att_vs_truth_hidden_degradation.csv
        - results/event_study_baseline.csv & results/event_study_hidden_degradation.csv
        - results/heterogeneity_baseline.csv & results/heterogeneity_hidden_degradation.csv
        - results/event_study_plot_baseline.png & results/event_study_plot_hidden_degradation.png
    Prints single-run TWFE estimates with CIs for Q5 in hidden_degradation.
    Restores baseline simulation panel in DuckDB.
    """
    res_base = run_full_causal_evaluation(
        db_path=db_path,
        random_seed=random_seed,
        churn_scenario="baseline",
        top_spend_churn_cumulative=0.20,
    )
    res_hid = run_full_causal_evaluation(
        db_path=db_path,
        random_seed=random_seed,
        churn_scenario="hidden_degradation",
        top_spend_churn_cumulative=0.40,
    )

    # Print single-run TWFE estimates with CIs for Q5 in hidden_degradation
    df_hid = res_hid["evaluation_df"]
    q5_hid = df_hid[(df_hid["Cohort"] == "Quintile 5 (Top Spend)") & (df_hid["Estimator_Model"] == "TWFE DiD (PanelOLS)")]

    print("\n=========================================================================================")
    print("      SINGLE-RUN TWFE ESTIMATES & 95% CONFIDENCE INTERVALS FOR QUINTILE 5 (TOP SPEND)    ")
    print("                               (SCENARIO: HIDDEN_DEGRADATION)                            ")
    print("=========================================================================================")
    print(f"{'Outcome Metric':<25} | {'Estimated ATT':<15} | {'Std Error':<12} | {'95% CI Lower':<14} | {'95% CI Upper':<14} | {'p-value'}")
    print("-" * 100)
    for _, row in q5_hid.iterrows():
        print(
            f"{row['Outcome_Metric']:<25} | "
            f"{row['Estimated_ATT']:<+15.4f} | "
            f"{row['Standard_Error']:<12.4f} | "
            f"{row['ci_lower']:<+14.4f} | "
            f"{row['ci_upper']:<+14.4f} | "
            f"{row['p_value']:.4e}"
        )
    print("=========================================================================================\n")

    # Restore DuckDB to baseline simulation panel
    print("Restoring DuckDB table 'causal_experiment_data' to baseline scenario...")
    simulate_policy_intervention(db_path=db_path, churn_scenario="baseline", random_seed=random_seed)

    return {"baseline": res_base, "hidden_degradation": res_hid}


if __name__ == "__main__":
    run_both_causal_scenarios()
