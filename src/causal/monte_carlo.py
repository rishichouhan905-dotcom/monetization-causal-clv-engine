"""Monte Carlo Simulation Engine for Causal Estimator Validation.

Runs repeated independent replications (with distinct assignment and outcome draws)
under 'baseline' and 'hidden_degradation' scenarios to evaluate TWFE DiD performance:
- Mean Bias: E[tau_hat - tau_true]
- RMSE: sqrt(E[(tau_hat - tau_true)^2])
- Empirical 95% CI Coverage: P(tau_true in CI_95)
- Rejection Rate at 5%: P(p_value < 0.05)

Evaluated across three outcomes (weekly_orders, weekly_spend_winsor, active_last_90d)
and three cohorts (Overall, Q1-Q4 pooled, Quintile 5).
Saves results to results/monte_carlo.csv and reports exact runtime.
"""

import time
from typing import Any, Dict, List, Tuple
import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats

from clv_causal.config import DB_PATH, INTERVENTION_DATE, PROJECT_ROOT


def build_base_simulation_panel(
    db_path: str = DB_PATH,
    intervention_date: str = INTERVENTION_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, int]:
    """Loads base customer data and builds shared counterfactual panel structure."""
    conn = duckdb.connect(db_path, read_only=True)
    t0_date = pd.to_datetime(intervention_date)
    t0_str = t0_date.strftime("%Y-%m-%d")

    # 1. Weekly aggregate transactions
    weekly_tx = conn.execute("""
        SELECT
            customer_id,
            DATE_TRUNC('week', invoice_date) AS week_date,
            COUNT(DISTINCT invoice_no) AS cf_orders,
            SUM(line_total) AS cf_spend
        FROM fact_orders
        GROUP BY customer_id, DATE_TRUNC('week', invoice_date)
    """).df()
    weekly_tx["week_date"] = pd.to_datetime(weekly_tx["week_date"])

    # 2. First order week and eligibility
    first_weeks = weekly_tx.groupby("customer_id")["week_date"].min().reset_index()
    first_weeks.columns = ["customer_id", "first_week"]
    eligible_first_weeks = first_weeks[first_weeks["first_week"] < t0_date].copy()
    eligible_ids = set(eligible_first_weeks["customer_id"])
    weekly_tx = weekly_tx[weekly_tx["customer_id"].isin(eligible_ids)].copy()

    # 3. Pre-period invoices
    pre_invoices = conn.execute(f"""
        SELECT customer_id, invoice_no, invoice_date, SUM(line_total) AS order_val
        FROM fact_orders
        WHERE invoice_date < '{t0_str}'
        GROUP BY customer_id, invoice_no, invoice_date
    """).df()
    pre_invoices = pre_invoices[pre_invoices["customer_id"].isin(eligible_ids)].copy()

    pre_summary = (
        pre_invoices.groupby("customer_id")
        .agg(
            pre_spend=("order_val", "sum"),
            pre_orders=("invoice_no", "count"),
            median_order_val=("order_val", "median"),
        )
        .reset_index()
    )
    pre_summary = pre_summary.merge(eligible_first_weeks, on="customer_id")

    w_t0 = t0_date - pd.to_timedelta(t0_date.weekday(), unit="D")
    pre_summary["pre_weeks"] = np.maximum(
        1, ((w_t0 - pre_summary["first_week"]).dt.days // 7)
    )

    # Shrunken empirical Bayes repeat order rate
    repeat_orders = np.maximum(0, pre_summary["pre_orders"] - 1)
    total_repeats = repeat_orders.sum()
    total_weeks = pre_summary["pre_weeks"].sum()
    global_repeat_rate = float(total_repeats / total_weeks) if total_weeks > 0 else 0.05
    beta_prior = 4.0
    alpha_prior = global_repeat_rate * beta_prior
    pre_summary["pre_order_rate"] = (
        (repeat_orders + alpha_prior) / (pre_summary["pre_weeks"] + beta_prior)
    )

    pre_summary["spend_quintile"] = pd.qcut(
        pre_summary["pre_spend"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    # Build balanced customer-by-week panel
    last_week = weekly_tx["week_date"].max()
    records = []
    for _, row in pre_summary[["customer_id", "first_week"]].iterrows():
        c_id = row["customer_id"]
        fw = row["first_week"]
        w_range = pd.date_range(start=fw, end=last_week, freq="W-MON")
        records.append(pd.DataFrame({"customer_id": c_id, "week_date": w_range}))

    panel = pd.concat(records, ignore_index=True)
    panel = panel.merge(weekly_tx, on=["customer_id", "week_date"], how="left")
    panel["cf_orders"] = panel["cf_orders"].fillna(0).astype(int)
    panel["cf_spend"] = panel["cf_spend"].fillna(0.0)

    panel = panel.merge(
        pre_summary[
            [
                "customer_id",
                "spend_quintile",
                "median_order_val",
                "pre_order_rate",
            ]
        ],
        on="customer_id",
        how="left",
    )

    post_weeks = sorted(panel[panel["week_date"] >= t0_date]["week_date"].unique())
    n_post_weeks = len(post_weeks)
    post_week_to_idx = {w: i + 1 for i, w in enumerate(post_weeks)}
    panel["post_step"] = panel["week_date"].map(post_week_to_idx)
    panel["is_post_period"] = panel["post_step"].notna().astype(int)

    pos_spend = panel.loc[panel["cf_spend"] > 0, "cf_spend"]
    p99_val = float(np.percentile(pos_spend, 99)) if len(pos_spend) > 0 else 5000.0

    conn.close()
    return panel, pre_summary, p99_val, n_post_weeks


class FastPanelTWFE:
    """Pre-computes sparse entity and time dummy matrices for ultra-fast TWFE DiD estimation."""

    def __init__(self, df: pd.DataFrame):
        self.n_rows = len(df)
        self.c_codes, self.c_unq = pd.factorize(df["customer_id"])
        self.w_codes, self.w_unq = pd.factorize(df["week_date"])
        self.n_c = len(self.c_unq)
        self.n_w = len(self.w_unq)

        cols_fe = np.column_stack([self.c_codes, self.n_c + self.w_codes]).ravel()
        rows_fe = np.repeat(np.arange(self.n_rows), 2)
        self.X_fe = sp.csr_matrix(
            (np.ones(len(rows_fe), dtype=np.float64), (rows_fe, cols_fe)),
            shape=(self.n_rows, self.n_c + self.n_w),
        )
        self.dof_adj = (self.n_c / (self.n_c - 1)) * (
            (self.n_rows - 1) / (self.n_rows - (self.n_c + self.n_w))
        )

    def estimate(
        self,
        y: np.ndarray,
        d: np.ndarray,
        iter_lim: int = 60,
    ) -> Tuple[float, float, float, float, float]:
        """Estimates tau, clustered SE, p-value, and 95% CI via Frisch-Waugh-Lovell projection."""
        # Project treatment D onto fixed effects
        sol_d = spla.lsqr(self.X_fe, d, iter_lim=iter_lim)
        d_tilde = d - self.X_fe @ sol_d[0]

        # Project outcome Y onto fixed effects
        sol_y = spla.lsqr(self.X_fe, y, iter_lim=iter_lim)
        y_tilde = y - self.X_fe @ sol_y[0]

        denom = np.dot(d_tilde, d_tilde)
        if denom <= 1e-12:
            return 0.0, np.nan, np.nan, np.nan, np.nan

        tau = float(np.dot(d_tilde, y_tilde) / denom)
        e = y_tilde - tau * d_tilde

        # Clustered standard error by customer
        score_it = d_tilde * e
        cust_score = pd.Series(score_it).groupby(self.c_codes).sum().values
        meat = np.sum(cust_score**2)
        se = float(np.sqrt(meat / (denom**2) * self.dof_adj))

        # 95% Confidence Interval & p-value
        ci_lower = tau - 1.95996 * se
        ci_upper = tau + 1.95996 * se
        z_stat = tau / se if se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

        return tau, se, p_val, ci_lower, ci_upper


def run_single_simulation_replication(
    panel: pd.DataFrame,
    pre_summary: pd.DataFrame,
    p99_val: float,
    n_post_weeks: int,
    cum_churn: float,
    order_rate_change: float,
    aov_change: float,
    seed: int,
) -> pd.DataFrame:
    """Generates stochastic assignment and realized outcome draws for one replication."""
    rng = np.random.default_rng(seed)

    # 1. Stratified random assignment (40% per quintile)
    treated_set = set()
    for q in [1, 2, 3, 4, 5]:
        c_ids = pre_summary[pre_summary["spend_quintile"] == q]["customer_id"].values.copy()
        n_q = len(c_ids)
        n_tr = int(round(n_q * 0.40))
        treated_set.update(rng.permutation(c_ids)[:n_tr])

    is_tr_map = {cid: (1 if cid in treated_set else 0) for cid in pre_summary["customer_id"]}

    # 2. Geometric churn draws for top quintile treated customers
    weekly_churn_hazard = 1.0 - (1.0 - cum_churn) ** (1.0 / max(1, n_post_weeks))
    top_treated = [
        cid
        for cid in pre_summary[pre_summary["spend_quintile"] == 5]["customer_id"]
        if cid in treated_set
    ]
    churn_steps = rng.geometric(p=weekly_churn_hazard, size=len(top_treated))
    churn_map = dict(zip(top_treated, churn_steps))

    # 3. Simulate realized customer-week outcomes
    df = panel.copy()
    df["is_treated"] = df["customer_id"].map(is_tr_map).fillna(0).astype(int)
    df["churn_step"] = df["customer_id"].map(churn_map)
    df["is_churned"] = (
        (df["is_post_period"] == 1)
        & df["churn_step"].notna()
        & (df["post_step"] > df["churn_step"])
    )

    tr_post = (df["is_treated"] == 1) & (df["is_post_period"] == 1)
    churned_mask = tr_post & df["is_churned"]
    active_tr_post = tr_post & (~df["is_churned"])

    df["observed_weekly_orders"] = df["cf_orders"].astype(float)
    df["observed_weekly_spend"] = df["cf_spend"].astype(float)

    df.loc[churned_mask, "observed_weekly_orders"] = 0.0
    df.loc[churned_mask, "observed_weekly_spend"] = 0.0

    # Add orders in zero weeks
    zero_mask = active_tr_post & (df["cf_orders"] == 0)
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        add_probs = np.clip(
            df.loc[zero_mask, "pre_order_rate"].values * order_rate_change, 0.0, 1.0
        )
        add_order = rng.random(size=n_zero) < add_probs
        med_vals = df.loc[zero_mask, "median_order_val"].values
        df.loc[zero_mask, "observed_weekly_orders"] = np.where(add_order, 1.0, 0.0)
        df.loc[zero_mask, "observed_weekly_spend"] = np.where(
            add_order, med_vals * (1.0 + aov_change), 0.0
        )

    # Scale spend in positive weeks
    pos_mask = active_tr_post & (df["cf_orders"] > 0)
    df.loc[pos_mask, "observed_weekly_spend"] = df.loc[pos_mask, "cf_spend"] * (
        1.0 + aov_change
    )

    df["observed_weekly_spend_winsor"] = np.clip(df["observed_weekly_spend"], 0.0, p99_val)
    df["counterfactual_weekly_spend_winsor"] = np.clip(df["cf_spend"], 0.0, p99_val)

    # Active in last 90 days (13-week rolling sum)
    df = df.sort_values(by=["customer_id", "week_date"]).reset_index(drop=True)
    df["observed_active_last_90d"] = (
        df.groupby("customer_id")["observed_weekly_orders"]
        .rolling(13, min_periods=1)
        .sum()
        .reset_index(drop=True)
        > 0
    ).astype(float)
    df["counterfactual_active_last_90d"] = (
        df.groupby("customer_id")["cf_orders"]
        .rolling(13, min_periods=1)
        .sum()
        .reset_index(drop=True)
        > 0
    ).astype(float)

    # Realized unit effects
    df["realized_effect_orders"] = df["observed_weekly_orders"] - df["cf_orders"]
    df["realized_effect_spend_winsor"] = (
        df["observed_weekly_spend_winsor"] - df["counterfactual_weekly_spend_winsor"]
    )
    df["realized_effect_active_last_90d"] = (
        df["observed_active_last_90d"] - df["counterfactual_active_last_90d"]
    )

    df["treat_x_post"] = (df["is_treated"] * df["is_post_period"]).astype(float)
    return df


def run_monte_carlo_evaluation(
    n_replications: int = 50,
    random_seed: int = 100,
) -> Tuple[pd.DataFrame, float]:
    """Executes Monte Carlo evaluation for baseline and hidden_degradation scenarios.

    Args:
        n_replications: Number of independent replications per scenario (default 50).
        random_seed: Base seed for reproducible draws.

    Returns:
        Tuple of (monte_carlo_summary_df, total_runtime_seconds).
    """
    t_start = time.time()
    print("\n" + "=" * 88)
    print("STARTING CAUSAL MONTE CARLO ESTIMATOR VALIDATION")
    print(f"Replications per scenario: {n_replications}")
    print("Scenarios evaluated:       'baseline' (cum_churn=0.20) and 'hidden_degradation' (cum_churn=0.40)")
    print("=" * 88)

    base_panel, pre_summary, p99_val, n_post_weeks = build_base_simulation_panel()

    # Pre-build fast TWFE solvers for the three cohort slices
    mask_q14 = base_panel["spend_quintile"].isin([1, 2, 3, 4])
    mask_q5 = base_panel["spend_quintile"] == 5

    idx_ov = np.arange(len(base_panel))
    idx_q14 = np.where(mask_q14)[0]
    idx_q5 = np.where(mask_q5)[0]

    solver_ov = FastPanelTWFE(base_panel)
    solver_q14 = FastPanelTWFE(base_panel.iloc[idx_q14])
    solver_q5 = FastPanelTWFE(base_panel.iloc[idx_q5])

    scenarios = [
        ("baseline", 0.20),
        ("hidden_degradation", 0.40),
    ]

    outcomes = [
        ("weekly_orders", "observed_weekly_orders", "realized_effect_orders"),
        ("weekly_spend_winsor", "observed_weekly_spend_winsor", "realized_effect_spend_winsor"),
        ("active_last_90d", "observed_active_last_90d", "realized_effect_active_last_90d"),
    ]

    cohorts = [
        ("Overall (All Treated)", solver_ov, idx_ov),
        ("Q1-Q4 Pooled", solver_q14, idx_q14),
        ("Quintile 5 (Top Spend)", solver_q5, idx_q5),
    ]

    all_rep_records: List[Dict[str, Any]] = []

    for scen_name, cum_churn in scenarios:
        t_scen_0 = time.time()
        print(f"\n--> Running scenario: '{scen_name}' (cum_churn={cum_churn:.2f})...")

        for rep in range(1, n_replications + 1):
            rep_seed = random_seed + rep * 37 + (1000 if scen_name == "hidden_degradation" else 0)
            rep_df = run_single_simulation_replication(
                panel=base_panel,
                pre_summary=pre_summary,
                p99_val=p99_val,
                n_post_weeks=n_post_weeks,
                cum_churn=cum_churn,
                order_rate_change=0.15,
                aov_change=0.10,
                seed=rep_seed,
            )

            tr_post = (rep_df["is_treated"] == 1) & (rep_df["is_post_period"] == 1)

            for cohort_name, solver, cohort_idx in cohorts:
                sub_df = rep_df.iloc[cohort_idx]
                sub_tr_post = tr_post.iloc[cohort_idx]

                d_vec = sub_df["treat_x_post"].values

                for metric_name, obs_col, true_col in outcomes:
                    true_att = float(sub_df.loc[sub_tr_post, true_col].mean())
                    y_vec = sub_df[obs_col].values

                    est_att, se, p_val, ci_l, ci_u = solver.estimate(y_vec, d_vec)

                    error = est_att - true_att
                    covered = int(ci_l <= true_att <= ci_u) if not np.isnan(ci_l) else 0
                    rejected = int(p_val < 0.05) if not np.isnan(p_val) else 0

                    all_rep_records.append({
                        "scenario": scen_name,
                        "replication": rep,
                        "cohort": cohort_name,
                        "outcome": metric_name,
                        "true_att": true_att,
                        "estimated_att": est_att,
                        "standard_error": se,
                        "error": error,
                        "sq_error": error**2,
                        "ci_covered": covered,
                        "rejected_at_5pct": rejected,
                    })

            if rep % 10 == 0 or rep == n_replications:
                elapsed = time.time() - t_scen_0
                print(f"    Completed {rep}/{n_replications} replications in {elapsed:.1f}s...")

    df_reps = pd.DataFrame(all_rep_records)

    # Compute summary metrics per (scenario, cohort, outcome)
    summary_rows = []
    grouped = df_reps.groupby(["scenario", "cohort", "outcome"])
    for (scen, coh, out), group in grouped:
        mean_true = group["true_att"].mean()
        mean_est = group["estimated_att"].mean()
        mean_bias = group["error"].mean()
        rmse = np.sqrt(group["sq_error"].mean())
        mean_se = group["standard_error"].mean()
        coverage_95 = group["ci_covered"].mean()
        rejection_5pct = group["rejected_at_5pct"].mean()

        summary_rows.append({
            "Scenario": scen,
            "Cohort": coh,
            "Outcome": out,
            "Replications": len(group),
            "Mean_True_ATT": mean_true,
            "Mean_Estimated_ATT": mean_est,
            "Mean_Bias": mean_bias,
            "RMSE": rmse,
            "Mean_SE": mean_se,
            "Coverage_95pct": coverage_95,
            "Rejection_Rate_5pct": rejection_5pct,
        })

    summary_df = pd.DataFrame(summary_rows)

    out_csv = PROJECT_ROOT / "results" / "monte_carlo.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    total_time = time.time() - t_start
    print("\n" + "=" * 88)
    print("MONTE CARLO EVALUATION COMPLETED")
    print(f"Total Runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Results saved to: {out_csv}")
    print("=" * 88)

    print("\nSummary Results Table:")
    print(
        summary_df[
            [
                "Scenario",
                "Cohort",
                "Outcome",
                "Mean_Bias",
                "RMSE",
                "Coverage_95pct",
                "Rejection_Rate_5pct",
            ]
        ].to_string(index=False)
    )

    return summary_df, total_time


if __name__ == "__main__":
    run_monte_carlo_evaluation(n_replications=50)
