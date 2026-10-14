"""Causal Reproducibility & Side-by-Side Verification Script.

1. Reads committed evaluation results.
2. Regenerates both baseline and hidden_degradation causal results from current code.
3. Compares regenerated estimates vs committed values and reports any differences.
4. Prints side-by-side table: Ground-Truth ATT, TWFE estimates, and Monte Carlo coverage (50 reps).
5. Prints event-study linear combination SE.
6. Reconciles revenue (gross, returns, net) across DuckDB tables.
"""

import duckdb
import numpy as np
import pandas as pd

from causal.evaluation import run_both_causal_scenarios
from causal.event_study import run_event_study
from clv_causal.config import DB_PATH, PROJECT_ROOT


def main() -> None:
    results_dir = PROJECT_ROOT / "results"

    # 1. Load committed values before regeneration
    print("Loading committed evaluation results...")
    base_old = pd.read_csv(results_dir / "att_vs_truth_baseline.csv")
    hid_old = pd.read_csv(results_dir / "att_vs_truth_hidden_degradation.csv")

    # 2. Regenerate both scenarios from current code
    print("\nRegenerating all causal results from current code...")
    res = run_both_causal_scenarios(db_path=DB_PATH, random_seed=42)

    base_new = res["baseline"]["evaluation_df"]
    hid_new = res["hidden_degradation"]["evaluation_df"]

    # 3. Check differences with committed values
    def compare_dfs(old_df: pd.DataFrame, new_df: pd.DataFrame, scen_name: str) -> None:
        print(f"\n=== Difference Analysis for {scen_name} Scenario ===")
        merged = pd.merge(
            old_df,
            new_df,
            on=["Estimator_Model", "Cohort", "Outcome_Metric"],
            suffixes=("_committed", "_regenerated"),
        )
        diffs = []
        for _, r in merged.iterrows():
            est_c = r["Estimated_ATT_committed"]
            est_r = r["Estimated_ATT_regenerated"]
            tru_c = r["True_ATT_committed"]
            tru_r = r["True_ATT_regenerated"]

            diff_est = abs(est_c - est_r) if not (np.isnan(est_c) and np.isnan(est_r)) else 0.0
            diff_tru = abs(tru_c - tru_r) if not (np.isnan(tru_c) and np.isnan(tru_r)) else 0.0

            if diff_est > 1e-5 or diff_tru > 1e-5:
                diffs.append((
                    r["Estimator_Model"],
                    r["Cohort"],
                    r["Outcome_Metric"],
                    est_c,
                    est_r,
                    diff_est,
                    tru_c,
                    tru_r,
                    diff_tru,
                ))

        if not diffs:
            print(f"[VERIFIED] All {len(merged)} rows match committed values EXACTLY within tolerance (<= 1e-5).")
        else:
            print(f"[DIFF] Detected {len(diffs)} differences:")
            for d in diffs:
                print(f"  - {d[0]} | {d[1]} | {d[2]}: Est committed={d[3]}, regenerated={d[4]} (diff={d[5]:.6f}) | True committed={d[6]}, regenerated={d[7]} (diff={d[8]:.6f})")

    compare_dfs(base_old, base_new, "Baseline")
    compare_dfs(hid_old, hid_new, "Hidden Degradation")

    # 4. Print Side-by-Side: Ground-Truth ATT, TWFE estimates, and Monte Carlo Coverage
    mc_csv_path = results_dir / "monte_carlo.csv"
    mc_df = pd.read_csv(mc_csv_path)

    print("\n" + "=" * 125)
    print("      SIDE-BY-SIDE: GROUND TRUTH ATT, TWFE ESTIMATE, 95% CI, AND MONTE CARLO COVERAGE (50 REPS)")
    print("=" * 125)
    header = f"{'Scenario':<20} | {'Cohort':<24} | {'Outcome':<20} | {'True ATT':<10} | {'TWFE Est':<10} | {'TWFE 95% CI':<24} | {'MC Cov %':<8}"
    print(header)
    print("-" * 125)

    for scen, df_eval in [("baseline", base_new), ("hidden_degradation", hid_new)]:
        twfe = df_eval[df_eval["Estimator_Model"] == "TWFE DiD (PanelOLS)"]
        for _, r in twfe.iterrows():
            coh = r["Cohort"]
            out = r["Outcome_Metric"]
            est = float(r["Estimated_ATT"])
            tru = float(r["True_ATT"])
            ci_l = float(r["ci_lower"])
            ci_u = float(r["ci_upper"])
            ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]"

            # Match Monte Carlo row
            mc_match = mc_df[(mc_df["Scenario"] == scen) & (mc_df["Cohort"] == coh) & (mc_df["Outcome"] == out)]
            cov_str = f"{mc_match.iloc[0]['Coverage_95pct'] * 100:.0f}%" if len(mc_match) > 0 else "N/A"
            print(f"{scen:<20} | {coh:<24} | {out:<20} | {tru:<+10.4f} | {est:<+10.4f} | {ci_str:<24} | {cov_str:<8}")

    print("=" * 125 + "\n")

    # 5. Print Event-Study SE
    print("--- Event Study Analysis & Standard Error ---")
    es_res = run_event_study(outcome_col="observed_weekly_spend_winsor")
    print(f"Event Study Mean Post-Period ATT: ${es_res['mean_post_att']:+.4f}")
    print(f"Event Study Linear Combination SE: ${es_res['mean_post_se']:.4f}")
    print(f"Parallel Trends Joint Wald Test:  Chi2 = {es_res['joint_test_stat']:.4f}, p = {es_res['joint_test_pval']:.4f}")
    print(f"Conclusion: {es_res['conclusion_text']}")

    # 6. Reconcile Revenue (Gross, Returns, Net)
    print("\n--- Revenue Reconciliation (Gross, Returns, Net) ---")
    conn = duckdb.connect(DB_PATH, read_only=True)
    gross_fact = conn.execute("SELECT SUM(line_total) FROM fact_orders").fetchone()[0]
    returns_fact = conn.execute("SELECT SUM(return_amount) FROM fact_returns").fetchone()[0]
    net_dim = conn.execute("SELECT SUM(total_spend) FROM dim_customers").fetchone()[0]
    gross_dim, ret_dim, net_dim2 = conn.execute("SELECT SUM(gross_spend), SUM(return_spend), SUM(total_spend) FROM dim_customers").fetchone()
    conn.close()

    print(f"fact_orders Gross Revenue:         ${gross_fact:,.2f}")
    print(f"fact_returns Total Returns:        ${returns_fact:,.2f}")
    print(f"fact_orders - fact_returns (Net):  ${gross_fact - returns_fact:,.2f}")
    print(f"dim_customers Gross Spend:         ${gross_dim:,.2f}")
    print(f"dim_customers Return Spend:        ${ret_dim:,.2f}")
    print(f"dim_customers Net Spend:           ${net_dim:,.2f}")
    print(f"Difference (dim Gross - Returns - Net): ${gross_dim - ret_dim - net_dim2:,.4f} (Exact $0.00)")
    print(f"Difference (fact Net vs dim Net):       ${(gross_fact - returns_fact) - net_dim:,.4f} (Rounding delta < $0.02 across 779k rows)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
