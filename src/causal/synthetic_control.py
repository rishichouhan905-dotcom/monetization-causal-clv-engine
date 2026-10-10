"""Synthetic Control Estimator with Non-Negative Convex Weights, RMSPE, and Placebo Permutation.

Constructs an optimal synthetic control unit from donor pool cohorts using convex
optimization (w_j >= 0, sum(w_j) = 1), calculates pre-period RMSPE, and computes
placebo-in-space permutation p-values.
"""

from typing import Any, Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clv_causal.config import DB_PATH, INTERVENTION_DATE


def solve_donor_weights(
    y_target_pre: np.ndarray, Y_donor_pre: np.ndarray
) -> np.ndarray:
    """Solves constrained optimization for non-negative donor weights summing to 1.

    Args:
        y_target_pre: Target trajectory pre-intervention vector of shape (T0,).
        Y_donor_pre: Matrix of donor trajectories pre-intervention of shape (T0, J).

    Returns:
        Weight vector of shape (J,) with w_j >= 0 and sum(w_j) = 1.0.
    """
    T0, J = Y_donor_pre.shape
    if J == 0:
        return np.array([])

    def loss_func(w: np.ndarray) -> float:
        pred = np.dot(Y_donor_pre, w)
        return float(np.mean((y_target_pre - pred) ** 2))

    # Initial uniform weights
    w0 = np.full(J, 1.0 / J)
    bounds = [(0.0, 1.0) for _ in range(J)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = res.x if res.success else w0
    weights = np.maximum(0.0, weights)
    weights /= max(1e-12, np.sum(weights))

    return weights


def estimate_synthetic_control(
    panel_df: Optional[pd.DataFrame] = None,
    intervention_date: str = INTERVENTION_DATE,
    n_donors: int = 5,
    outcome_col: str = "observed_weekly_spend",
    treatment_col: str = "is_treated",
    time_col: str = "week_date",
    customer_col: str = "customer_id",
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Fits Synthetic Control model, evaluates pre-period RMSPE, and runs placebo-in-space permutations.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        intervention_date: Policy rollout date ISO string.
        n_donors: Number of donor cohorts to partition control customers into.
        outcome_col: Outcome column name.
        treatment_col: Treatment indicator column name.
        time_col: Week timestamp column name.
        customer_col: Customer identifier column name.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary containing estimated_att, pre_rmspe, donor_weights_df, placebo_pval, placebo_effects.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    t0_date = pd.to_datetime(intervention_date)
    df["is_post"] = np.where(df[time_col] >= t0_date, 1, 0)

    # Aggregate target treated trajectory (weekly mean across treated customers)
    treated_panel = df[df[treatment_col] == 1]
    target_traj = (
        treated_panel.groupby([time_col, "is_post"])[outcome_col]
        .mean()
        .reset_index()
        .sort_values(time_col)
    )

    # Partition control customers into n_donors deterministic donor cohorts
    control_panel = df[df[treatment_col] == 0].copy()
    control_custs = np.sort(control_panel[customer_col].unique())

    if len(control_custs) == 0:
        return {
            "estimated_att": 0.0,
            "pre_rmspe": 0.0,
            "donor_weights_df": pd.DataFrame(),
            "placebo_pval": 1.0,
            "placebo_effects": [],
        }

    # Assign donor cohort ID (0 to n_donors - 1)
    donor_assignments = {cid: idx % n_donors for idx, cid in enumerate(control_custs)}
    control_panel["donor_cohort"] = control_panel[customer_col].map(donor_assignments)

    donor_trajs = (
        control_panel.groupby([time_col, "donor_cohort"])[outcome_col]
        .mean()
        .unstack(level="donor_cohort")
        .reindex(target_traj[time_col])
        .fillna(0.0)
    )

    # Split pre and post periods
    pre_mask = (target_traj["is_post"] == 0).values
    post_mask = (target_traj["is_post"] == 1).values

    y_target_pre = target_traj.loc[pre_mask, outcome_col].values
    y_target_post = target_traj.loc[post_mask, outcome_col].values

    Y_donor_pre = donor_trajs.loc[pre_mask].values
    Y_donor_post = donor_trajs.loc[post_mask].values

    # Optimize weights w_j >= 0, sum(w_j) = 1
    weights = solve_donor_weights(y_target_pre, Y_donor_pre)

    # Synthetic control trajectories
    synth_pre = np.dot(Y_donor_pre, weights)
    synth_post = np.dot(Y_donor_post, weights)

    # Calculate pre-period RMSPE
    pre_rmspe = float(np.sqrt(np.mean((y_target_pre - synth_pre) ** 2)))

    # Estimated ATT post-t0
    att_synth = float(np.mean(y_target_post - synth_post))

    # Construct Donor Weights Table
    weights_df = pd.DataFrame({
        "donor_cohort": [f"Donor_Cohort_{j+1}" for j in range(len(weights))],
        "weight": np.round(weights, 4),
    })

    # Placebo-in-Space Permutation Test across donor cohorts
    placebo_effects = []
    J = Y_donor_pre.shape[1]

    for j in range(J):
        # Leave cohort j as pseudo-target
        y_pseudo_pre = Y_donor_pre[:, j]
        y_pseudo_post = Y_donor_post[:, j]

        # Other donors as donor pool
        other_indices = [idx for idx in range(J) if idx != j]
        if len(other_indices) == 0:
            continue

        Y_other_pre = Y_donor_pre[:, other_indices]
        Y_other_post = Y_donor_post[:, other_indices]

        w_p = solve_donor_weights(y_pseudo_pre, Y_other_pre)
        synth_pseudo_post = np.dot(Y_other_post, w_p)
        eff_p = float(np.mean(y_pseudo_post - synth_pseudo_post))
        placebo_effects.append(eff_p)

    # Permutation p-value
    if len(placebo_effects) > 0:
        placebo_pval = float((np.sum(np.abs(placebo_effects) >= abs(att_synth)) + 1) / (len(placebo_effects) + 1))
    else:
        placebo_pval = 1.0

    print("\n=================== Synthetic Control Estimation Results ===================")
    print(f"Synthetic Control ATT Estimate:          ${att_synth:+.4f} / week")
    print(f"Pre-Treatment RMSPE:                    ${pre_rmspe:.4f}")
    print(f"Placebo-in-Space Permutation p-value:    {placebo_pval:.4f}")
    print("\nDonor Cohort Weights:")
    print(weights_df.to_string(index=False))
    print("============================================================================\n")

    return {
        "estimated_att": att_synth,
        "pre_rmspe": pre_rmspe,
        "donor_weights_df": weights_df,
        "placebo_pval": placebo_pval,
        "placebo_effects": placebo_effects,
    }


if __name__ == "__main__":
    estimate_synthetic_control()
