#!/usr/bin/env python3
"""Run Confounded Assignment Benchmark on Real Data Panel.

Re-runs the policy intervention simulation under confounded assignment on the
real customer panel, performs Event Study pre-trend Wald test, TWFE DiD, and
Synthetic Control estimation, and saves all outputs to results/confounded_assignment.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal.event_study import run_event_study
from causal.simulate_policy import simulate_policy_intervention
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did


def main() -> None:
    print("=====================================================================")
    print("      RUNNING CONFOUNDED ASSIGNMENT BENCHMARK ON REAL PANEL          ")
    print("=====================================================================\n")

    # 1. Simulate policy with confounded assignment on real data
    print("1. Simulating policy intervention with assignment_mode='confounded'...")
    panel_conf = simulate_policy_intervention(
        assignment_mode="confounded",
        churn_scenario="baseline",
        random_seed=42,
        persist_to_db=False,
    )

    # 2. Pre-trend Wald test via Event Study
    print("2. Running event study leads & lags for pre-trend Wald test...")
    tmp_plot = "_temp_conf_es.png"
    res_es = run_event_study(
        panel_df=panel_conf,
        outcome_col="observed_weekly_spend_winsor",
        plot_filename=tmp_plot,
    )
    (PROJECT_ROOT / "results" / tmp_plot).unlink(missing_ok=True)

    wald_chi2 = float(res_es["joint_test_stat"])
    wald_df = 9  # 9 pre-period leads (-10 through -2 relative weeks)
    wald_p = float(res_es["joint_test_pval"])

    # 3. Ground-truth Realized ATT
    t0 = pd.to_datetime("2011-06-01")
    post_tr_mask = (panel_conf["is_treated"] == 1) & (pd.to_datetime(panel_conf["week_date"]) >= t0)
    true_att = float(panel_conf.loc[post_tr_mask, "realized_effect_spend_winsor"].mean())

    # 4. TWFE DiD
    print("3. Estimating Two-Way Fixed Effects DiD...")
    res_twfe = estimate_twfe_did(
        panel_df=panel_conf,
        outcome_col="observed_weekly_spend_winsor",
    )
    twfe_att = float(res_twfe["estimated_att"])
    twfe_se = float(res_twfe["std_err"])
    twfe_bias = twfe_att - true_att

    # 5. Synthetic Control
    print("4. Estimating Synthetic Control with 50 donor cohorts...")
    res_sc = estimate_synthetic_control(
        panel_df=panel_conf,
        outcome_col="observed_weekly_spend_winsor",
        random_seed=42,
    )
    sc_att = float(res_sc["estimated_att"])
    sc_pre_rmspe = float(res_sc["pre_rmspe"])

    # 6. Spend growth metrics & definition
    spend_growth_def = "(late_spend - early_spend) / (late_spend + early_spend + 10.0)"
    tr_growth = float(panel_conf[panel_conf["is_treated"] == 1]["spend_growth"].mean())
    ct_growth = float(panel_conf[panel_conf["is_treated"] == 0]["spend_growth"].mean())
    growth_gap = tr_growth - ct_growth

    # 7. Construct output DataFrame
    records = [
        {"metric": "pre_trend_wald_chi2", "value": f"{wald_chi2:.4f}", "description": "Joint pre-period Wald parallel trends chi2 test statistic"},
        {"metric": "pre_trend_wald_df", "value": f"{wald_df}", "description": "Degrees of freedom for pre-period leads (-10 to -2)"},
        {"metric": "pre_trend_wald_p", "value": f"{wald_p:.6e}", "description": "P-value for joint pre-trend Wald test (rejects parallel trends)"},
        {"metric": "true_att", "value": f"{true_att:.4f}", "description": "Realized true ATT for weekly winsorized spend ($/cust/wk)"},
        {"metric": "twfe_estimate", "value": f"{twfe_att:.4f}", "description": "TWFE DiD ATT point estimate ($/cust/wk)"},
        {"metric": "twfe_se", "value": f"{twfe_se:.4f}", "description": "TWFE DiD cluster-robust standard error"},
        {"metric": "twfe_bias", "value": f"{twfe_bias:.4f}", "description": "TWFE DiD bias (estimate - true_att)"},
        {"metric": "sc_estimate", "value": f"{sc_att:.4f}", "description": "Synthetic Control ATT point estimate ($/cust/wk)"},
        {"metric": "sc_pre_rmspe", "value": f"{sc_pre_rmspe:.4f}", "description": "Synthetic Control pre-treatment RMSPE ($)"},
        {"metric": "spend_growth_definition", "value": spend_growth_def, "description": "Normalized formula for customer pre-period spend growth: (late - early) / (late + early + 10.0)"},
        {"metric": "treated_mean_spend_growth", "value": f"{tr_growth:.4f}", "description": "Mean pre-period spend growth for treated cohort"},
        {"metric": "control_mean_spend_growth", "value": f"{ct_growth:.4f}", "description": "Mean pre-period spend growth for control cohort"},
        {"metric": "spend_growth_gap", "value": f"{growth_gap:.4f}", "description": "Pre-period spend growth gap (treated - control)"},
    ]

    out_df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / "results" / "confounded_assignment.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved confounded assignment outputs to: {out_path}\n")

    print("=====================================================================")
    print("               CONFOUNDED ASSIGNMENT RESULTS SUMMARY                 ")
    print("=====================================================================")
    for _, r in out_df.iterrows():
        print(f"  {r['metric']:<28} = {r['value']:<20} ({r['description']})")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
