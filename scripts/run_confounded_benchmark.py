#!/usr/bin/env python3
"""Run Causal Assignment Benchmarks on Real Data Panel.

Re-runs both the confounded assignment stress test and the stratified random
baseline assignment on the real customer panel using the current covariance-based
Wald test, TWFE DiD, and Synthetic Control models.

Saves:
  - results/confounded_assignment.csv
  - results/baseline_assignment.csv
with full provenance: code version, timestamp, source script name, and panel row count.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal.event_study import run_event_study
from causal.simulate_policy import simulate_policy_intervention
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did


def get_git_code_version() -> str:
    """Returns the current git HEAD commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def run_benchmark(assignment_mode: str) -> tuple[dict, pd.DataFrame]:
    """Runs full causal benchmark for a given assignment mode on real panel.

    Args:
        assignment_mode: 'confounded' or 'stratified_random'

    Returns:
        Tuple of (results_dict, panel_df).
    """
    print("\n=====================================================================")
    print(f"      RUNNING REAL PANEL BENCHMARK: {assignment_mode.upper()}")
    print("=====================================================================")

    # 1. Simulate policy on real data
    print(f"1. Simulating policy with assignment_mode='{assignment_mode}'...")
    panel = simulate_policy_intervention(
        assignment_mode=assignment_mode,
        churn_scenario="baseline",
        random_seed=42,
        persist_to_db=False,
    )
    panel_row_count = len(panel)

    # 2. Pre-trend Wald test via Event Study (covariance-based)
    print("2. Running event study leads & lags for covariance-based Wald test...")
    tmp_plot = f"_temp_{assignment_mode}_es.png"
    res_es = run_event_study(
        panel_df=panel,
        outcome_col="observed_weekly_spend_winsor",
        plot_filename=tmp_plot,
    )
    (PROJECT_ROOT / "results" / tmp_plot).unlink(missing_ok=True)

    wald_chi2 = float(res_es["joint_test_stat"])
    wald_df = 9  # 9 pre-period leads (-10 to -2)
    wald_p = float(res_es["joint_test_pval"])

    # 3. Ground-truth Realized ATT
    t0 = pd.to_datetime("2011-06-01")
    post_tr_mask = (panel["is_treated"] == 1) & (pd.to_datetime(panel["week_date"]) >= t0)
    true_att = float(panel.loc[post_tr_mask, "realized_effect_spend_winsor"].mean())

    # 4. TWFE DiD
    print("3. Estimating Two-Way Fixed Effects DiD...")
    res_twfe = estimate_twfe_did(
        panel_df=panel,
        outcome_col="observed_weekly_spend_winsor",
    )
    twfe_att = float(res_twfe["estimated_att"])
    twfe_se = float(res_twfe["std_err"])
    twfe_diff = twfe_att - true_att

    # 5. Synthetic Control
    print("4. Estimating Synthetic Control with 50 donor cohorts...")
    res_sc = estimate_synthetic_control(
        panel_df=panel,
        outcome_col="observed_weekly_spend_winsor",
        random_seed=42,
    )
    sc_att = float(res_sc["estimated_att"])
    sc_pre_rmspe = float(res_sc["pre_rmspe"])

    # 6. Spend growth metrics
    spend_growth_def = "(late_spend - early_spend) / (late_spend + early_spend + 10.0)"
    tr_growth = float(panel[panel["is_treated"] == 1]["spend_growth"].mean())
    ct_growth = float(panel[panel["is_treated"] == 0]["spend_growth"].mean())
    growth_gap = tr_growth - ct_growth

    code_version = get_git_code_version()
    timestamp_str = datetime.now(timezone.utc).isoformat()
    source_script_name = "scripts/run_confounded_benchmark.py"

    diff_label = "twfe_bias" if assignment_mode == "confounded" else "single_draw_error"
    diff_desc = (
        "TWFE DiD bias (estimate - true_att) from confounded selection"
        if assignment_mode == "confounded"
        else "Single-draw estimation error (estimate - true_att) under random assignment"
    )

    records = [
        {"metric": "code_version", "value": code_version, "description": "Git commit SHA at execution time"},
        {"metric": "timestamp", "value": timestamp_str, "description": "UTC timestamp of benchmark execution"},
        {"metric": "source_script", "value": source_script_name, "description": "Source script that generated this benchmark"},
        {"metric": "panel_row_count", "value": str(panel_row_count), "description": "Number of rows in the balanced customer-week panel"},
        {"metric": "assignment_mode", "value": assignment_mode, "description": "Treatment assignment mechanism"},
        {"metric": "pre_trend_wald_chi2", "value": f"{wald_chi2:.4f}", "description": "Joint pre-period Wald parallel trends chi2 test statistic"},
        {"metric": "pre_trend_wald_df", "value": f"{wald_df}", "description": "Degrees of freedom for pre-period leads (-10 to -2)"},
        {"metric": "pre_trend_wald_p", "value": f"{wald_p:.6e}", "description": "P-value for joint pre-trend Wald test"},
        {"metric": "true_att", "value": f"{true_att:.4f}", "description": "Realized true ATT for weekly winsorized spend ($/cust/wk)"},
        {"metric": "twfe_estimate", "value": f"{twfe_att:.4f}", "description": "TWFE DiD ATT point estimate ($/cust/wk)"},
        {"metric": "twfe_se", "value": f"{twfe_se:.4f}", "description": "TWFE DiD cluster-robust standard error"},
        {"metric": diff_label, "value": f"{twfe_diff:.4f}", "description": diff_desc},
        {"metric": "sc_estimate", "value": f"{sc_att:.4f}", "description": "Synthetic Control ATT point estimate ($/cust/wk)"},
        {"metric": "sc_pre_rmspe", "value": f"{sc_pre_rmspe:.4f}", "description": "Synthetic Control pre-treatment RMSPE ($)"},
        {"metric": "spend_growth_definition", "value": spend_growth_def, "description": "Normalized formula for customer pre-period spend growth: (late - early) / (late + early + 10.0)"},
        {"metric": "treated_mean_spend_growth", "value": f"{tr_growth:.4f}", "description": "Mean pre-period spend growth for treated cohort"},
        {"metric": "control_mean_spend_growth", "value": f"{ct_growth:.4f}", "description": "Mean pre-period spend growth for control cohort"},
        {"metric": "spend_growth_gap", "value": f"{growth_gap:.4f}", "description": "Pre-period spend growth gap (treated - control)"},
    ]

    out_df = pd.DataFrame(records)
    return records, out_df


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Baseline Random Assignment
    records_base, df_base = run_benchmark(assignment_mode="stratified_random")
    base_path = results_dir / "baseline_assignment.csv"
    df_base.to_csv(base_path, index=False)
    print(f"\n[OK] Saved baseline random-assignment outputs to: {base_path}")

    # 2. Run Confounded Assignment
    records_conf, df_conf = run_benchmark(assignment_mode="confounded")
    conf_path = results_dir / "confounded_assignment.csv"
    df_conf.to_csv(conf_path, index=False)
    print(f"\n[OK] Saved confounded assignment outputs to: {conf_path}")

    print("\n=====================================================================")
    print("                    BENCHMARK OUTPUT SUMMARY                         ")
    print("=====================================================================")
    base_dict = dict(zip(df_base["metric"], df_base["value"]))
    conf_dict = dict(zip(df_conf["metric"], df_conf["value"]))

    print(f"{'Metric':<30} | {'Baseline (Random)':<22} | {'Confounded (Stress Test)':<22}")
    print("-" * 78)
    common_metrics = [
        "code_version", "panel_row_count", "pre_trend_wald_chi2", "pre_trend_wald_p",
        "true_att", "twfe_estimate", "twfe_se",
        "sc_estimate", "sc_pre_rmspe", "treated_mean_spend_growth",
        "control_mean_spend_growth", "spend_growth_gap"
    ]
    for m in common_metrics:
        b_val = base_dict.get(m, "—")
        c_val = conf_dict.get(m, "—")
        print(f"{m:<30} | {b_val:<22} | {c_val:<22}")
    print(f"{'single_draw_error / bias':<30} | {base_dict.get('single_draw_error', '—'):<22} | {conf_dict.get('twfe_bias', '—'):<22}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
