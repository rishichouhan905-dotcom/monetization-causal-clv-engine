"""Event Study Estimator with Pre-Period Joint Test and Visual Plotting.

Estimates relative week leads and lags relative to policy date t_0, executes
a joint Wald test for pre-period parallel trends (H_0: beta_k = 0 for k < -1),
and generates event study plots.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from clv_causal.config import DB_PATH, INTERVENTION_DATE, PROJECT_ROOT


def run_event_study(
    panel_df: Optional[pd.DataFrame] = None,
    intervention_date: str = INTERVENTION_DATE,
    outcome_col: str = "observed_weekly_spend",
    treatment_col: str = "is_treated",
    customer_col: str = "customer_id",
    time_col: str = "week_date",
    max_leads: int = 10,
    max_lags: int = 10,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Runs event study model, performs joint pre-period parallel trend test, and saves plot.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        intervention_date: ISO date string for policy date t_0.
        outcome_col: Outcome column name.
        treatment_col: Treatment indicator column name.
        customer_col: Customer ID column name.
        time_col: Timestamp column name.
        max_leads: Maximum pre-treatment weeks to evaluate (default 10).
        max_lags: Maximum post-treatment weeks to evaluate (default 10).
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with event_study_df, joint_test_stat, joint_test_pval, parallel_trends_pass.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    t0_date = pd.to_datetime(intervention_date)

    # Relative week calculation
    df["rel_week"] = (df[time_col] - t0_date).dt.days // 7
    df["rel_week"] = np.clip(df["rel_week"], -max_leads, max_lags)

    # Weekly aggregate trajectory per treatment group for clean event-study estimation
    agg_df = (
        df.groupby([treatment_col, "rel_week", time_col])[outcome_col]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_spend"})
    )

    # OLS Event Study Regression on relative week dummies
    # Baseline omitted week is rel_week == -1
    rel_weeks = sorted([int(w) for w in agg_df["rel_week"].unique() if w != -1])
    pre_weeks = [w for w in rel_weeks if w < -1]
    post_weeks = [w for w in rel_weeks if w >= 0]

    X_list = [np.ones(len(agg_df)), agg_df[treatment_col].values]
    cols = ["intercept", "is_treated"]

    for w in rel_weeks:
        week_dummy = (agg_df["rel_week"] == w).astype(float).values
        inter_dummy = (agg_df[treatment_col] * (agg_df["rel_week"] == w)).astype(float).values
        X_list.append(week_dummy)
        cols.append(f"w_{w}")
        X_list.append(inter_dummy)
        cols.append(f"treat_x_w_{w}")

    X = np.column_stack(X_list)
    y = agg_df["avg_spend"].values

    XtX_inv = np.linalg.pinv(np.dot(X.T, X))
    beta = np.dot(XtX_inv, np.dot(X.T, y))
    residuals = y - np.dot(X, beta)
    deg_freedom = max(1, len(y) - X.shape[1])
    sigma_sq = np.sum(residuals**2) / deg_freedom
    vcov = sigma_sq * XtX_inv

    # Extract interaction estimates and SEs per relative week
    event_rows = []
    # Baseline omitted week k = -1
    event_rows.append({"rel_week": -1, "beta": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "is_post": 0})

    pre_indices = []
    for w in rel_weeks:
        param_name = f"treat_x_w_{w}"
        idx = cols.index(param_name)
        b_val = float(beta[idx])
        b_se = float(np.sqrt(max(0.0, vcov[idx, idx])))
        t_crit = float(stats.t.ppf(0.975, df=deg_freedom))

        if w < -1:
            pre_indices.append(idx)

        event_rows.append({
            "rel_week": w,
            "beta": b_val,
            "se": b_se,
            "ci_lower": b_val - t_crit * b_se,
            "ci_upper": b_val + t_crit * b_se,
            "is_post": 1 if w >= 0 else 0,
        })

    event_df = pd.DataFrame(event_rows).sort_values("rel_week").reset_index(drop=True)

    # Joint Wald Test for pre-period lead coefficients: H_0: beta_k = 0 for all k < -1
    if len(pre_indices) > 0:
        beta_pre = beta[pre_indices]
        vcov_pre = vcov[np.ix_(pre_indices, pre_indices)]
        vcov_pre_inv = np.linalg.pinv(vcov_pre)
        joint_stat = float(np.dot(beta_pre.T, np.dot(vcov_pre_inv, beta_pre)) / len(pre_indices))
        joint_pval = float(1.0 - stats.f.cdf(joint_stat, dfn=len(pre_indices), dfd=deg_freedom))
    else:
        joint_stat = 0.0
        joint_pval = 1.0

    parallel_trends_pass = bool(joint_pval > 0.05)

    print("\n=================== Event Study Leads & Lags Analysis ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Joint Pre-Period Parallel Trends Test: F-stat = {joint_stat:.4f} (p-value = {joint_pval:.4f})")
    if parallel_trends_pass:
        print("[PASS] Pre-period lead coefficients are jointly zero (statistically parallel baseline).")
    else:
        print("[NOTE] Pre-period lead coefficients indicate non-zero differential baseline movement.")
    print("=========================================================================\n")

    # Generate Event Study Plot
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.axvline(-0.5, color="red", linestyle=":", linewidth=1.5, label="Policy Rollout (t_0)")

        ax.errorbar(
            event_df["rel_week"],
            event_df["beta"],
            yerr=1.96 * event_df["se"],
            fmt="o-",
            color="#2b5c8f",
            ecolor="#6c9bd2",
            elinewidth=1.5,
            capsize=3,
            label="ATT Dynamic Lead/Lag (95% CI)",
        )

        ax.set_title("Event Study: Dynamic Treatment Effects Across Relative Weeks", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Weeks to Policy Intervention (k = 0 at t_0)", fontsize=10)
        ax.set_ylabel("Weekly Spend ATT ($)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        plot_path = results_dir / "event_study_plot.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved event study plot to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not render matplotlib plot image ({e}).")

    return {
        "event_study_df": event_df,
        "joint_test_stat": joint_stat,
        "joint_test_pval": joint_pval,
        "parallel_trends_pass": parallel_trends_pass,
    }


if __name__ == "__main__":
    run_event_study()
