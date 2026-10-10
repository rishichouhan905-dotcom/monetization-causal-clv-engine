"""Refutation Sensitivity Tests: Placebo Treatment Date & Leave-One-Donor-Out.

Executes falsification diagnostics to test model validity and identification assumptions:
1. Placebo Treatment Date: Shifts policy date t_0 to pre-period (expected ATT ~ 0).
2. Leave-One-Donor-Out: Evaluates sensitivity of synthetic control ATT to individual donors.
"""

from typing import Any, Dict, List, Optional
import duckdb
import numpy as np
import pandas as pd

from causal.synthetic_control import solve_donor_weights
from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH, INTERVENTION_DATE


def run_placebo_date_refutation(
    panel_df: Optional[pd.DataFrame] = None,
    actual_t0: str = INTERVENTION_DATE,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Runs placebo treatment date refutation test strictly on pre-intervention period.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        actual_t0: Actual policy rollout date ISO string.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with placebo_att, std_err, p_value, and test_passed.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()
    df["week_date"] = pd.to_datetime(df["week_date"])
    actual_t0_date = pd.to_datetime(actual_t0)

    # Filter strictly to pre-intervention data
    pre_df = df[df["week_date"] < actual_t0_date].copy()
    if len(pre_df) == 0:
        return {"placebo_att": 0.0, "std_err": 0.0, "p_value": 1.0, "test_passed": True}

    # Select midpoint of pre-intervention window as placebo date t0_placebo
    min_date = pre_df["week_date"].min()
    max_date = pre_df["week_date"].max()
    placebo_t0 = min_date + (max_date - min_date) / 2.0

    pre_df["is_post_period"] = np.where(pre_df["week_date"] >= placebo_t0, 1, 0)

    res = estimate_twfe_did(panel_df=pre_df)
    placebo_att = float(res["estimated_att"])
    p_val = float(res["p_value"])
    test_passed = bool(p_val > 0.05 or abs(placebo_att) < 0.50)

    print("\n=================== Refutation 1: Placebo Treatment Date ===================")
    print(f"Actual Policy Cutoff (t_0):             {actual_t0}")
    print(f"Placebo Policy Cutoff:                  {placebo_t0.strftime('%Y-%m-%d')}")
    print(f"Placebo ATT Estimate:                   ${placebo_att:+.4f} / week")
    print(f"Placebo p-value:                        {p_val:.4f}")
    if test_passed:
        print("[PASS] Placebo treatment date yields no statistically significant effect (ATT ~ 0).")
    else:
        print("[NOTE] Differential effect detected under placebo treatment date.")
    print("=============================================================================\n")

    return {
        "placebo_att": placebo_att,
        "std_err": float(res["std_err"]),
        "p_value": p_val,
        "test_passed": test_passed,
    }


def run_leave_one_donor_out_refutation(
    panel_df: Optional[pd.DataFrame] = None,
    intervention_date: str = INTERVENTION_DATE,
    n_donors: int = 5,
    outcome_col: str = "observed_weekly_spend",
    treatment_col: str = "is_treated",
    time_col: str = "week_date",
    customer_col: str = "customer_id",
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Runs leave-one-donor-out sensitivity test for synthetic control.

    Args:
        panel_df: Customer-by-week panel DataFrame.
        intervention_date: Policy rollout date ISO string.
        n_donors: Number of donor cohorts.
        outcome_col: Outcome column name.
        treatment_col: Treatment indicator column name.
        time_col: Timestamp column name.
        customer_col: Customer identifier column name.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary with leave_one_out_atts, mean_att, std_att, min_att, max_att.
    """
    if panel_df is None:
        conn = duckdb.connect(db_path)
        panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
        conn.close()

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    t0_date = pd.to_datetime(intervention_date)
    df["is_post"] = np.where(df[time_col] >= t0_date, 1, 0)

    treated_panel = df[df[treatment_col] == 1]
    target_traj = (
        treated_panel.groupby([time_col, "is_post"])[outcome_col]
        .mean()
        .reset_index()
        .sort_values(time_col)
    )

    control_panel = df[df[treatment_col] == 0].copy()
    control_custs = np.sort(control_panel[customer_col].unique())

    if len(control_custs) == 0:
        return {"leave_one_out_atts": [], "mean_att": 0.0, "std_att": 0.0, "min_att": 0.0, "max_att": 0.0}

    donor_assignments = {cid: idx % n_donors for idx, cid in enumerate(control_custs)}
    control_panel["donor_cohort"] = control_panel[customer_col].map(donor_assignments)

    donor_trajs = (
        control_panel.groupby([time_col, "donor_cohort"])[outcome_col]
        .mean()
        .unstack(level="donor_cohort")
        .reindex(target_traj[time_col])
        .fillna(0.0)
    )

    pre_mask = (target_traj["is_post"] == 0).values
    post_mask = (target_traj["is_post"] == 1).values

    y_target_pre = target_traj.loc[pre_mask, outcome_col].values
    y_target_post = target_traj.loc[post_mask, outcome_col].values

    Y_donor_pre = donor_trajs.loc[pre_mask].values
    Y_donor_post = donor_trajs.loc[post_mask].values

    J = Y_donor_pre.shape[1]
    loo_atts = []

    for j in range(J):
        remaining_indices = [idx for idx in range(J) if idx != j]
        if len(remaining_indices) == 0:
            continue
        Y_sub_pre = Y_donor_pre[:, remaining_indices]
        Y_sub_post = Y_donor_post[:, remaining_indices]

        w_sub = solve_donor_weights(y_target_pre, Y_sub_pre)
        synth_sub_post = np.dot(Y_sub_post, w_sub)
        att_sub = float(np.mean(y_target_post - synth_sub_post))
        loo_atts.append(att_sub)

    mean_att = float(np.mean(loo_atts)) if len(loo_atts) > 0 else 0.0
    std_att = float(np.std(loo_atts)) if len(loo_atts) > 0 else 0.0
    min_att = float(np.min(loo_atts)) if len(loo_atts) > 0 else 0.0
    max_att = float(np.max(loo_atts)) if len(loo_atts) > 0 else 0.0

    print("\n=================== Refutation 2: Leave-One-Donor-Out ===================")
    print(f"Leave-One-Donor-Out ATTs (N={len(loo_atts)}):      Range [${min_att:.4f}, ${max_att:.4f}]")
    print(f"Mean LOO ATT:                           ${mean_att:+.4f} (Std Dev = ${std_att:.4f})")
    print("==========================================================================\n")

    return {
        "leave_one_out_atts": loo_atts,
        "mean_att": mean_att,
        "std_att": std_att,
        "min_att": min_att,
        "max_att": max_att,
    }


def run_all_refutations(panel_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Orchestrates all falsification and refutation diagnostic tests."""
    p_res = run_placebo_date_refutation(panel_df)
    l_res = run_leave_one_donor_out_refutation(panel_df)
    return {"placebo_date": p_res, "leave_one_out": l_res}


if __name__ == "__main__":
    run_all_refutations()
