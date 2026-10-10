"""Two-Way Fixed Effects (TWFE) Difference-in-Differences Estimator at Customer Level.

Fits customer-level TWFE model with customer fixed effects, week fixed effects,
and standard errors clustered by customer:
    y_{it} = alpha_i + gamma_t + beta_DiD (is_treated_i * is_post_{it}) + epsilon_{it}
"""

from typing import Any, Dict, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from clv_causal.config import DB_PATH


def estimate_twfe_did(
    panel_df: Optional[pd.DataFrame] = None,
    outcome_col: str = "observed_weekly_spend",
    treatment_col: str = "is_treated",
    post_col: str = "is_post_period",
    customer_col: str = "customer_id",
    time_col: str = "week_date",
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Estimates customer-level Two-Way Fixed Effects (TWFE) DiD with customer-clustered SEs.

    Args:
        panel_df: DataFrame customer-by-week panel. If None, queries causal_experiment_data.
        outcome_col: Column name for dependent variable (observed_weekly_spend).
        treatment_col: Column name for treatment group indicator (is_treated).
        post_col: Column name for post-period indicator (is_post_period).
        customer_col: Column name for customer identifier.
        time_col: Column name for week timestamp.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with estimated_att, std_err, t_stat, p_value, ci_lower, ci_upper, n_obs, n_customers.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Construct treatment interaction term D_it = is_treated * is_post_period
    df["D_it"] = (df[treatment_col] * df[post_col]).astype(float)
    df["y_it"] = df[outcome_col].astype(float)

    N = len(df)
    unique_custs = df[customer_col].unique()
    G = len(unique_custs)

    if N == 0 or G == 0:
        return {
            "estimated_att": 0.0,
            "std_err": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_obs": 0,
            "n_customers": 0,
        }

    # Demeaning (Within Transformation) for Customer Fixed Effects and Week Fixed Effects
    # 1. Overall means
    y_bar = df["y_it"].mean()
    d_bar = df["D_it"].mean()

    # 2. Customer means
    cust_means = df.groupby(customer_col)[["y_it", "D_it"]].transform("mean")
    df["y_cust_mean"] = cust_means["y_it"]
    df["d_cust_mean"] = cust_means["D_it"]

    # 3. Week means
    week_means = df.groupby(time_col)[["y_it", "D_it"]].transform("mean")
    df["y_week_mean"] = week_means["y_it"]
    df["d_week_mean"] = week_means["D_it"]

    # 4. Demeaned variables: \tilde{z}_{it} = z_{it} - \bar{z}_i - \bar{z}_t + \bar{z}
    df["y_tilde"] = df["y_it"] - df["y_cust_mean"] - df["y_week_mean"] + y_bar
    df["d_tilde"] = df["D_it"] - df["d_cust_mean"] - df["d_week_mean"] + d_bar

    # OLS coefficient on demeaned data: beta_hat = sum(d_tilde * y_tilde) / sum(d_tilde^2)
    denom = (df["d_tilde"] ** 2).sum()
    if denom <= 1e-12:
        att = 0.0
    else:
        att = float((df["d_tilde"] * df["y_tilde"]).sum() / denom)

    # Compute Residuals: e_{it} = y_{it} - alpha_i - gamma_t - beta_hat * D_{it}
    df["e_hat"] = df["y_tilde"] - att * df["d_tilde"]

    # Cluster-robust standard error clustered by customer_id
    # Score for customer i: S_i = sum_t (d_tilde_{it} * e_hat_{it})
    cust_scores = df.groupby(customer_col).apply(
        lambda g: (g["d_tilde"] * g["e_hat"]).sum(), include_groups=False
    ).values

    df_correction = (G / max(1, G - 1)) * ((N - 1) / max(1, N - 2))
    meat = np.sum(cust_scores**2)
    var_att = df_correction * meat / (denom**2) if denom > 1e-12 else 0.0
    se = float(np.sqrt(max(0.0, var_att)))

    t_stat = float(att / se) if se > 0 else 0.0
    dof = max(1, G - 1)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
    t_crit = float(stats.t.ppf(0.975, df=dof))

    ci_l = float(att - t_crit * se)
    ci_u = float(att + t_crit * se)

    res = {
        "estimated_att": att,
        "std_err": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "n_obs": N,
        "n_customers": G,
    }

    print("\n=================== Two-Way Fixed Effects (TWFE) DiD Results ===================")
    print(f"Customer-Level TWFE ATT Estimate:        ${att:+.4f} / week")
    print(f"Cluster-Robust Standard Error (SE):     ${se:.4f} (clustered by {G:,} customers)")
    print(f"t-statistic:                           {t_stat:.4f} (p-value = {p_val:.4e})")
    print(f"95% Cluster-Robust Confidence Interval:  [${ci_l:.4f}, ${ci_u:.4f}]")
    print("=================================================================================\n")

    return res


if __name__ == "__main__":
    estimate_twfe_did()
