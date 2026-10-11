"""Event Study Estimator with Pre-Period Joint Test and Visual Plotting.

Estimates relative week leads and lags relative to policy date t_0, executes
a joint Wald test for pre-period parallel trends (H_0: beta_k = 0 for k < -1),
and generates high-resolution publication-quality event study plots.
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
    outcome_col: str = "observed_weekly_spend_winsor",
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
        outcome_col: Outcome column name (default observed_weekly_spend_winsor).
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

    # Cell-weighted summary per treatment group and relative week
    agg = (
        df.groupby([treatment_col, "rel_week"])[outcome_col]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    piv = agg.pivot(index="rel_week", columns=treatment_col, values=["mean", "count", "std"])
    piv.columns = [f"{col[0]}_{col[1]}" for col in piv.columns]

    # Dynamic Difference in Differences relative to baseline week k = -1
    diff = piv["mean_1"] - piv["mean_0"]
    baseline_diff = diff.loc[-1] if -1 in diff.index else 0.0
    beta_series = diff - baseline_diff

    # Standard error via delta method on cell sample variances
    var_k = piv["std_1"] ** 2 / piv["count_1"] + piv["std_0"] ** 2 / piv["count_0"]
    var_base = var_k.loc[-1] if -1 in var_k.index else 0.0
    se_series = np.sqrt(var_k + var_base)
    if -1 in se_series.index:
        se_series.loc[-1] = 0.0

    event_rows = []
    rel_weeks = sorted(piv.index.unique())
    pre_weeks = [w for w in rel_weeks if w < -1]

    for w in rel_weeks:
        b_val = float(beta_series.loc[w])
        s_val = float(se_series.loc[w])
        ci_l = b_val - 1.96 * s_val if w != -1 else 0.0
        ci_u = b_val + 1.96 * s_val if w != -1 else 0.0
        event_rows.append({
            "rel_week": int(w),
            "beta": b_val,
            "se": s_val,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "is_post": 1 if w >= 0 else 0,
        })

    event_df = pd.DataFrame(event_rows).sort_values("rel_week").reset_index(drop=True)

    # Joint Wald Test for pre-period parallel trends: H_0: beta_k = 0 for all k < -1
    if len(pre_weeks) > 0:
        pre_betas = beta_series.loc[pre_weeks].values
        pre_ses = se_series.loc[pre_weeks].values
        chi2_stat = float(np.sum((pre_betas / np.maximum(1e-8, pre_ses)) ** 2))
        dfn = len(pre_weeks)
        joint_pval = float(1.0 - stats.chi2.cdf(chi2_stat, df=dfn))
    else:
        chi2_stat = 0.0
        joint_pval = 1.0

    parallel_trends_pass = bool(joint_pval > 0.05)

    print("\n=================== Event Study Leads & Lags Analysis ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Dependent Variable:                     {outcome_col}")
    print(f"Joint Pre-Period Parallel Trends Test:  Chi2 = {chi2_stat:.4f} (df={len(pre_weeks)}, p-value = {joint_pval:.4f})")
    if parallel_trends_pass:
        print("[PASS] Pre-trend results are consistent with parallel trends (under random assignment this holds by construction).")
    else:
        print("[NOTE] Pre-trend results reject parallel trends (indicates confounding or selection bias in assignment).")
    print("=========================================================================\n")

    # Generate Publication-Quality Event Study Plot
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

        # Baseline zero line and rollout dashed line
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(-0.5, color="#ef4444", linestyle="-.", linewidth=1.8, label=f"Policy Rollout $t_0$ ({intervention_date})")

        # Shaded confidence band and error bars
        ax.fill_between(
            event_df["rel_week"],
            event_df["ci_lower"],
            event_df["ci_upper"],
            color="#3b82f6",
            alpha=0.15,
            label="95% Confidence Band",
        )

        ax.errorbar(
            event_df["rel_week"],
            event_df["beta"],
            yerr=1.96 * event_df["se"],
            fmt="o-",
            color="#1d4ed8",
            ecolor="#93c5fd",
            elinewidth=2.0,
            capsize=4,
            capthick=1.5,
            markersize=6,
            markerfacecolor="#ffffff",
            markeredgewidth=2.0,
            markeredgecolor="#1d4ed8",
            label="Dynamic ATT Estimate ($\\hat{\\beta}_k$)",
        )

        # Annotations
        p_text = f"Pre-trend test: $\\chi^2$={chi2_stat:.2f}, p={joint_pval:.3f}"
        status_box = "Parallel Trends: PASS" if parallel_trends_pass else "Parallel Trends: CHECK"
        ax.text(
            0.03,
            0.92,
            f"{status_box}\n{p_text}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="semibold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        y_label = "Weekly Winsorized Spend ATT ($)" if "spend" in outcome_col else "Weekly Orders ATT"
        ax.set_title(
            f"Event Study: Dynamic Treatment Effects Around Policy Rollout\nDependent Variable: {outcome_col}",
            fontsize=13,
            fontweight="bold",
            pad=12,
            color="#0f172a",
        )
        ax.set_xlabel("Weeks Relative to Policy Intervention ($k = 0$ at rollout week)", fontsize=11, fontweight="medium", labelpad=8)
        ax.set_ylabel(y_label, fontsize=11, fontweight="medium", labelpad=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.5)

        plot_path = results_dir / "event_study_plot.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved event study plot to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not render matplotlib plot image ({e}).")

    return {
        "event_study_df": event_df,
        "joint_test_stat": chi2_stat,
        "joint_test_pval": joint_pval,
        "parallel_trends_pass": parallel_trends_pass,
    }


if __name__ == "__main__":
    run_event_study()
