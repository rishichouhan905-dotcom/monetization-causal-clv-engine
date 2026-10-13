"""Two-Way Fixed Effects (TWFE) Difference-in-Differences Estimator at Customer Level.

Fits customer-level TWFE model with customer fixed effects, week fixed effects,
and standard errors clustered by customer using linearmodels.PanelOLS:
    y_{it} = alpha_i + gamma_t + beta_DiD (is_treated_i * is_post_{it}) + epsilon_{it}
"""

from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from clv_causal.config import DB_PATH


def estimate_twfe_did(
    panel_df: Optional[pd.DataFrame] = None,
    outcome_col: str = "observed_weekly_spend_winsor",
    treatment_col: str = "is_treated",
    post_col: str = "is_post_period",
    customer_col: str = "customer_id",
    time_col: str = "week_date",
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Estimates customer-level Two-Way Fixed Effects (TWFE) DiD using linearmodels.PanelOLS with customer-clustered SEs.

    Args:
        panel_df: DataFrame customer-by-week panel. If None, queries causal_experiment_data.
        outcome_col: Column name for dependent variable (observed_weekly_spend_winsor, observed_weekly_orders, etc.).
        treatment_col: Column name for treatment group indicator (is_treated).
        post_col: Column name for post-period indicator (is_post_period).
        customer_col: Column name for customer identifier.
        time_col: Column name for week timestamp.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with estimated_att, std_err, t_stat, p_value, ci_lower, ci_upper, n_obs, n_customers, degenerate.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        df = conn.execute(f"SELECT {customer_col}, {time_col}, {treatment_col}, {post_col}, {outcome_col} FROM causal_experiment_data").df()
        conn.close()
    else:
        req_cols = [customer_col, time_col, treatment_col, post_col, outcome_col]
        df = panel_df[req_cols].copy()

    df[time_col] = pd.to_datetime(df[time_col])

    # Construct treatment interaction term D_it = is_treated * is_post_period
    df["treat_x_post"] = (df[treatment_col] * df[post_col]).astype(float)
    df["y_it"] = df[outcome_col].astype(float)

    N = len(df)
    unique_custs = df[customer_col].unique()
    G = len(unique_custs)

    if N == 0 or G == 0 or df["treat_x_post"].nunique() < 2:
        return {
            "estimated_att": np.nan,
            "std_err": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": N,
            "n_customers": G,
            "degenerate": True,
        }

    # For large panels (N > 100_000), PanelOLS exhausts memory constructing dense projection matrices.
    # Route directly to the mathematically equivalent memory-efficient sparse FWL solver.
    if N > 100_000:
        return _estimate_twfe_did_sparse(
            df=df,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            post_col=post_col,
            customer_col=customer_col,
            time_col=time_col,
        )

    # Prepare MultiIndex panel dataframe for linearmodels
    df_panel = df[[customer_col, time_col, "treat_x_post", "y_it"]].set_index([customer_col, time_col])

    try:
        mod = PanelOLS(

            dependent=df_panel["y_it"],
            exog=df_panel[["treat_x_post"]],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True, low_memory=True)

        att = float(res.params["treat_x_post"])
        se = float(res.std_errors["treat_x_post"])
        t_stat = float(res.tstats["treat_x_post"])
        p_val = float(res.pvalues["treat_x_post"])
        ci_lower = float(res.conf_int().iloc[0, 0])
        ci_upper = float(res.conf_int().iloc[0, 1])
        degenerate = False
    except Exception as e:
        print(f"PanelOLS fitting encountered {e}; invoking memory-efficient sparse FWL solver...")
        del df_panel
        import gc
        gc.collect()
        try:
            return _estimate_twfe_did_sparse(
                df=df,
                outcome_col=outcome_col,
                treatment_col=treatment_col,
                post_col=post_col,
                customer_col=customer_col,
                time_col=time_col,
            )
        except Exception as sparse_e:
            print(f"Sparse TWFE fallback error: {sparse_e}")

            return {
                "estimated_att": np.nan,
                "std_err": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_obs": N,
                "n_customers": G,
                "degenerate": True,
            }

    res_dict = {
        "estimated_att": att,
        "std_err": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": N,
        "n_customers": G,
        "degenerate": degenerate,
    }

    print("\n=================== Two-Way Fixed Effects (TWFE) DiD Results ===================")
    print(f"Dependent Variable:                     {outcome_col}")
    print(f"Customer-Level TWFE ATT Estimate:        {att:+.4f} / week")
    print(f"Cluster-Robust Standard Error (SE):     {se:.4f} (clustered by {G:,} customers)")
    print(f"t-statistic:                           {t_stat:.4f} (p-value = {p_val:.4e})")
    print(f"95% Cluster-Robust Confidence Interval:  [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print("=================================================================================\n")

    return res_dict


def _estimate_twfe_did_sparse(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "is_treated",
    post_col: str = "is_post_period",
    customer_col: str = "customer_id",
    time_col: str = "week_date",
) -> Dict[str, Any]:
    """Memory-efficient Frisch-Waugh-Lovell projection for TWFE DiD using scipy.sparse LSQR."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy import stats

    d = (df[treatment_col] * df[post_col]).astype(float).values
    y = df[outcome_col].astype(float).values

    n_rows = len(df)
    c_codes, c_unq = pd.factorize(df[customer_col])
    w_codes, w_unq = pd.factorize(df[time_col])
    n_c = len(c_unq)
    n_w = len(w_unq)

    # Exact zero-memory Frisch-Waugh-Lovell alternating within projection
    c_cnt = np.bincount(c_codes)
    w_cnt = np.bincount(w_codes)

    def _demean_2way(v: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
        v_res = v.copy()
        for _ in range(max_iter):
            v_old = v_res.copy()
            v_res -= (np.bincount(c_codes, weights=v_res) / c_cnt)[c_codes]
            v_res -= (np.bincount(w_codes, weights=v_res) / w_cnt)[w_codes]
            if np.max(np.abs(v_res - v_old)) < tol:
                break
        return v_res

    d_tilde = _demean_2way(d)
    y_tilde = _demean_2way(y)

    extra_df = n_c + n_w - 1
    dof_resid = max(1, n_rows - extra_df - 1)
    scale = n_rows / dof_resid

    denom = np.dot(d_tilde, d_tilde)
    if denom <= 1e-12:
        return {
            "estimated_att": np.nan,
            "std_err": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": n_rows,
            "n_customers": n_c,
            "degenerate": True,
        }

    tau = float(np.dot(d_tilde, y_tilde) / denom)
    e = y_tilde - tau * d_tilde

    cust_score = np.bincount(c_codes, weights=d_tilde * e)
    meat = np.sum(cust_score**2)
    se = float(np.sqrt(meat / (denom**2) * scale))

    ci_lower = tau - 1.95996 * se
    ci_upper = tau + 1.95996 * se
    t_stat = tau / se if se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

    return {
        "estimated_att": tau,
        "std_err": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_rows,
        "n_customers": n_c,
        "degenerate": False,
    }


if __name__ == "__main__":
    estimate_twfe_did()
