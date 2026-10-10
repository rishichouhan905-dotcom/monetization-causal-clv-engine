"""Unit tests for causal estimators in src/causal/ and ground-truth tolerance check."""

from pathlib import Path
import duckdb
import numpy as np
import pytest
import pandas as pd

from causal.evaluation import run_full_causal_evaluation
from causal.event_study import run_event_study
from causal.heterogeneity import analyze_clv_heterogeneity
from causal.refutations import run_placebo_date_refutation, run_leave_one_donor_out_refutation
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did
from clv_causal.config import DB_PATH, PROJECT_ROOT


def test_twfe_did_matches_truth_within_tolerance() -> None:
    """Tests that customer-level TWFE DiD estimate matches ground-truth ATT within standard error tolerance."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    # Ground-truth ATT for treated customer-weeks post-t0
    post_treated = panel_df[(panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)]
    true_att = float(post_treated["true_effect_spend"].mean()) if len(post_treated) > 0 else 0.0

    twfe_res = estimate_twfe_did(panel_df=panel_df)
    estimated_att = float(twfe_res["estimated_att"])
    se = float(twfe_res["std_err"])

    # Tolerance bound: 2.5 * Standard Error or $10/week limit
    tolerance = max(10.0, 2.5 * se)
    error = abs(estimated_att - true_att)

    assert error <= tolerance, (
        f"DiD Estimate (${estimated_att:.4f}) missed True ATT (${true_att:.4f}) "
        f"by ${error:.4f}, exceeding tolerance ${tolerance:.4f}"
    )


def test_event_study_pre_period_joint_test() -> None:
    """Tests event study leads/lags estimation and joint pre-period parallel trend test."""
    res = run_event_study()
    assert "event_study_df" in res
    assert "joint_test_stat" in res
    assert "joint_test_pval" in res
    assert len(res["event_study_df"]) > 0


def test_synthetic_control_donor_weights() -> None:
    """Tests Synthetic Control non-negative weights summing to 1 and RMSPE evaluation."""
    sc_res = estimate_synthetic_control()
    assert "donor_weights_df" in sc_res
    assert "pre_rmspe" in sc_res
    assert "placebo_pval" in sc_res

    weights = sc_res["donor_weights_df"]["weight"].values
    if len(weights) > 0:
        assert (weights >= 0.0).all()
        assert abs(weights.sum() - 1.0) < 1e-3


def test_refutations_and_heterogeneity() -> None:
    """Tests placebo date refutation, leave-one-donor-out, and CLV decile heterogeneity."""
    p_res = run_placebo_date_refutation()
    assert "placebo_att" in p_res
    assert "p_value" in p_res

    l_res = run_leave_one_donor_out_refutation()
    assert "leave_one_out_atts" in l_res

    het_df = analyze_clv_heterogeneity()
    assert len(het_df) > 0
    assert "decile" in het_df.columns
    assert "true_att" in het_df.columns
    assert "estimated_did_att" in het_df.columns


def test_full_causal_evaluation_export() -> None:
    """Tests full causal evaluation pipeline and att_vs_truth.csv file creation."""
    res = run_full_causal_evaluation()
    csv_path = Path(res["csv_path"])
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    df = pd.read_csv(csv_path)
    assert len(df) > 0
    assert "Estimator_Model" in df.columns
    assert "Estimated_ATT" in df.columns
    assert "True_ATT" in df.columns
