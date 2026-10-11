"""Unit tests for causal estimators in src/causal/ and ground-truth tolerance checks."""

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


def test_did_confidence_interval_covers_true_effect_weekly_orders() -> None:
    """Asserts that the customer-level TWFE DiD 95% CI covers the true population ATT on weekly_orders."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    tr_post = (panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)
    true_att_orders = float(panel_df.loc[tr_post, "realized_effect_orders"].mean())

    did_res = estimate_twfe_did(panel_df=panel_df, outcome_col="observed_weekly_orders")
    ci_lower = float(did_res["ci_lower"])
    ci_upper = float(did_res["ci_upper"])
    est_att = float(did_res["estimated_att"])

    assert ci_lower <= true_att_orders <= ci_upper, (
        f"DiD 95% CI [{ci_lower:.5f}, {ci_upper:.5f}] does not cover True ATT ({true_att_orders:.5f}) "
        f"on weekly_orders (Est={est_att:.5f})"
    )


def test_twfe_did_matches_truth_within_tolerance() -> None:
    """Tests that customer-level TWFE DiD estimate matches ground-truth ATT on winsorized spend within tolerance."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    tr_post = (panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)
    true_att = float(panel_df.loc[tr_post, "realized_effect_spend_winsor"].mean())

    twfe_res = estimate_twfe_did(panel_df=panel_df, outcome_col="observed_weekly_spend_winsor")
    estimated_att = float(twfe_res["estimated_att"])
    se = float(twfe_res["std_err"])

    # 95% confidence tolerance or 2.5 * SE
    tolerance = max(10.0, 2.5 * se)
    error = abs(estimated_att - true_att)

    assert error <= tolerance, (
        f"DiD Estimate (${estimated_att:.4f}) missed True ATT (${true_att:.4f}) "
        f"by ${error:.4f}, exceeding tolerance ${tolerance:.4f}"
    )


def test_event_study_pre_period_joint_test() -> None:
    """Tests event study leads/lags estimation and joint pre-period parallel trend test."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    res = run_event_study(panel_df=panel_df, outcome_col="observed_weekly_spend_winsor")
    assert "event_study_df" in res
    assert "joint_test_stat" in res
    assert "joint_test_pval" in res
    assert len(res["event_study_df"]) > 0

    plot_file = PROJECT_ROOT / "results" / "event_study_plot.png"
    assert plot_file.exists(), f"Expected {plot_file} to exist"


def test_synthetic_control_donor_weights() -> None:
    """Tests Synthetic Control non-negative weights summing to 1 and RMSPE evaluation with >= 40 donors."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    sc_res = estimate_synthetic_control(panel_df=panel_df, outcome_col="observed_weekly_spend_winsor", n_donors=50)
    assert "donor_weights_df" in sc_res
    assert "pre_rmspe" in sc_res
    assert "placebo_pval" in sc_res
    assert len(sc_res["donor_weights_df"]) >= 40

    weights = sc_res["donor_weights_df"]["weight"].values
    if len(weights) > 0:
        assert (weights >= 0.0).all()
        assert abs(weights.sum() - 1.0) < 1e-3


def test_refutations_and_heterogeneity() -> None:
    """Tests placebo date refutation, leave-one-donor-out, and CLV decile heterogeneity with minimum cell size."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
    conn.close()

    p_res = run_placebo_date_refutation(panel_df=panel_df, outcome_col="observed_weekly_spend_winsor")
    assert "placebo_att" in p_res
    assert "p_value" in p_res

    l_res = run_leave_one_donor_out_refutation(panel_df=panel_df, outcome_col="observed_weekly_spend_winsor", n_donors=50)
    assert "leave_one_out_atts" in l_res
    assert len(l_res["leave_one_out_atts"]) >= 40

    het_df = analyze_clv_heterogeneity(panel_df=panel_df, min_treated_customers=10)
    assert len(het_df) == 10
    assert "decile" in het_df.columns
    assert "true_att" in het_df.columns
    assert "estimated_did_att" in het_df.columns
    assert "degenerate" in het_df.columns


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
    assert "Cohort" in df.columns
    assert "Outcome_Metric" in df.columns
