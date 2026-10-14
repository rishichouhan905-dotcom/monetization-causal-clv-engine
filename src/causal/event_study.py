"""Event Study Estimator with Pre-Period Joint Test and Visual Plotting.

Estimates relative week leads and lags relative to policy date t_0, executes
a joint Wald test for pre-period parallel trends (H_0: beta_k = 0 for k < -1),
and generates high-resolution publication-quality event study plots.
"""

from typing import Any, Dict, Optional
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
    max_lags: int = 26,
    plot_filename: str = "event_study_plot.png",
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
        max_lags: Maximum post-treatment weeks to evaluate (default 26 to cover all 27 post-weeks).
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with event_study_df, joint_test_stat, joint_test_pval, parallel_trends_pass,
        mean_post_att, mean_post_se, and conclusion_text.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        df = conn.execute(f"SELECT {time_col}, {treatment_col}, {outcome_col} FROM causal_experiment_data").df()
        conn.close()
    else:
        needed_cols = [time_col, treatment_col, outcome_col]
        present = [c for c in needed_cols if c in panel_df.columns]
        df = panel_df[present].copy()

    df[time_col] = pd.to_datetime(df[time_col])
    t0_date = pd.to_datetime(intervention_date)

    # Relative week calculation binned across specified lead/lag window
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

    diff = piv["mean_1"] - piv["mean_0"]
    v_k = piv["std_1"] ** 2 / piv["count_1"] + piv["std_0"] ** 2 / piv["count_0"]

    rel_weeks = sorted(piv.index.unique())
    pre_weeks = [w for w in rel_weeks if w < 0]
    post_weeks = [w for w in rel_weeks if w >= 0]
    K = len(pre_weeks)

    # Dynamic Difference in Differences relative to the average of all pre-period weeks
    if K > 0:
        base_ref = float(diff.loc[pre_weeks].mean())
        var_base = float(v_k.loc[pre_weeks].sum() / (K ** 2))
    else:
        base_ref = 0.0
        var_base = 0.0

    beta_series = diff - base_ref

    # Standard error via delta method:
    # Post-period: Var(beta_w) = Var(diff_w) + Var(base_ref)
    # Pre-period:  Var(beta_w) = (1 - 2/K)*Var(diff_w) + Var(base_ref)
    se_series = pd.Series(index=piv.index, dtype=float)
    for w in rel_weeks:
        if w >= 0:
            se_series.loc[w] = float(np.sqrt(v_k.loc[w] + var_base))
        else:
            var_w = (1.0 - 2.0 / K) * v_k.loc[w] + var_base if K > 1 else var_base
            se_series.loc[w] = float(np.sqrt(max(1e-8, var_w)))

    event_rows = []
    for w in rel_weeks:
        b_val = float(beta_series.loc[w])
        s_val = float(se_series.loc[w])
        ci_l = b_val - 1.96 * s_val
        ci_u = b_val + 1.96 * s_val
        event_rows.append({
            "rel_week": int(w),
            "beta": b_val,
            "se": s_val,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "is_post": 1 if w >= 0 else 0,
        })

    event_df = pd.DataFrame(event_rows).sort_values("rel_week").reset_index(drop=True)

    # Joint Wald Test for pre-period parallel trends: H_0: beta_k = 0 for all k in pre_weeks
    # Since sum(beta_k) = 0 by construction, degrees of freedom df = K - 1
    if K > 1:
        test_leads = pre_weeks[:-1]
        dfn = K - 1
        b_test = beta_series.loc[test_leads].values
        Sigma_pre = np.zeros((dfn, dfn))
        for i in range(dfn):
            for j in range(dfn):
                wi, wj = test_leads[i], test_leads[j]
                if i == j:
                    Sigma_pre[i, j] = (1.0 - 2.0 / K) * v_k.loc[wi] + var_base
                else:
                    Sigma_pre[i, j] = var_base - (v_k.loc[wi] + v_k.loc[wj]) / K
        chi2_stat = float(b_test @ np.linalg.pinv(Sigma_pre) @ b_test)
        joint_pval = float(1.0 - stats.chi2.cdf(chi2_stat, df=dfn))
    else:
        chi2_stat = 0.0
        dfn = 1
        joint_pval = 1.0

    parallel_trends_pass = bool(joint_pval >= 0.05)

    # Mean of post-period dynamic coefficients across all 27 post-weeks
    mean_post_att = float(beta_series.loc[post_weeks].mean()) if len(post_weeks) > 0 else 0.0

    # Full covariance matrix of post-period dynamic coefficients (Sigma_post)
    # Since each beta_k = Delta_k - bar(Delta)_pre, Cov(beta_i, beta_j) = var_base > 0 for i != j,
    # and Var(beta_i) = v_k[wi] + var_base for i == j.
    M_post = len(post_weeks)
    Sigma_post = np.zeros((M_post, M_post))
    for i in range(M_post):
        for j in range(M_post):
            wi, wj = post_weeks[i], post_weeks[j]
            if i == j:
                Sigma_post[i, j] = float(v_k.loc[wi] + var_base)
            else:
                Sigma_post[i, j] = float(var_base)

    w = np.full(M_post, 1.0 / M_post) if M_post > 0 else np.array([])
    var_post_quad = float(w @ Sigma_post @ w) if M_post > 0 else 0.0
    mean_post_se = float(np.sqrt(var_post_quad))

    # Conditional conclusion text based on p-value
    if parallel_trends_pass:
        conclusion_text = (
            f"[PASS] Pre-trend test: no evidence against parallel trends (low power; under random assignment this holds by construction) "
            f"(Chi2 = {chi2_stat:.4f}, df = {dfn}, p = {joint_pval:.4f} >= 0.05)."
        )
    else:
        conclusion_text = (
            f"[VIOLATION] Pre-trend test rejects parallel trends (Chi2 = {chi2_stat:.4f}, df = {dfn}, p = {joint_pval:.4e} < 0.05). "
            f"Statistically significant differential movement exists between treated and control groups prior to policy intervention, "
            f"indicating confounding or selection bias in treatment assignment."
        )

    print("\n=================== Event Study Leads & Lags Analysis ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Dependent Variable:                     {outcome_col}")
    print(f"Post-Intervention Window:               {len(post_weeks)} weeks (k in [{min(post_weeks)}, {max(post_weeks)}])")
    print(f"Mean Dynamic Post-Period ATT:           ${mean_post_att:+.4f} (Linear Combination SE = ${mean_post_se:.4f})")
    print(f"Joint Pre-Period Parallel Trends Test:  Chi2 = {chi2_stat:.4f} (df={dfn}, p-value = {joint_pval:.4f})")
    print(conclusion_text)
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
        p_text = f"Pre-trend Wald: $\\chi^2$={chi2_stat:.2f} (df={dfn}), p={joint_pval:.3f}"
        status_box = "Parallel Trends: PASS" if parallel_trends_pass else "Parallel Trends: VIOLATION"
        ax.text(
            0.03,
            0.92,
            f"{status_box}\n{p_text}\nMean Post ATT: ${mean_post_att:+.2f}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="semibold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        y_label = "Weekly Winsorized Spend ATT ($)" if "spend" in outcome_col else "Weekly Orders ATT"
        ax.set_title(
            f"Event Study: Dynamic Treatment Effects Across 27 Post-Weeks\nDependent Variable: {outcome_col}",
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

        plot_path = results_dir / plot_filename
        fig.tight_layout()
        fig.savefig(plot_path)
        if plot_filename != "event_study_plot.png":
            fig.savefig(results_dir / "event_study_plot.png")
        plt.close(fig)
        print(f"Saved event study plot to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not render matplotlib plot image ({e}).")

    return {
        "event_study_df": event_df,
        "joint_test_stat": chi2_stat,
        "joint_test_pval": joint_pval,
        "parallel_trends_pass": parallel_trends_pass,
        "mean_post_att": mean_post_att,
        "mean_post_se": mean_post_se,
        "conclusion_text": conclusion_text,
        "cov_matrix_post": Sigma_post,
        "post_weights": w,
    }


if __name__ == "__main__":
    run_event_study()
