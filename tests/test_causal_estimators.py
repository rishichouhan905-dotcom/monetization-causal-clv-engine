"""Unit tests for causal estimators in src/causal/ and ground-truth tolerance checks."""

from pathlib import Path
import duckdb
import numpy as np
import pytest
import pandas as pd

from linearmodels.panel import PanelOLS
from causal.evaluation import run_full_causal_evaluation
from causal.event_study import run_event_study
from causal.heterogeneity import analyze_clv_heterogeneity
from causal.refutations import run_placebo_date_refutation, run_leave_one_donor_out_refutation
from causal.synthetic_control import estimate_synthetic_control
from causal.twfe_did import estimate_twfe_did, _estimate_twfe_did_sparse
from clv_causal.config import DB_PATH, PROJECT_ROOT


def test_did_confidence_interval_covers_true_effect_weekly_orders() -> None:
    """Asserts that the customer-level TWFE DiD 95% CI covers the true population ATT on weekly_orders."""
    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute(
        "SELECT customer_id, week_date, is_treated, is_post_period, observed_weekly_orders, realized_effect_orders "
        "FROM causal_experiment_data"
    ).df()
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
    panel_df = conn.execute(
        "SELECT customer_id, week_date, is_treated, is_post_period, observed_weekly_spend_winsor, realized_effect_spend_winsor "
        "FROM causal_experiment_data"
    ).df()
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


def test_sparse_fallback_vs_panel_ols_subsample() -> None:
    """Compares the sparse fallback with PanelOLS on a 50,000-row subsample (estimates and SE within 1e-6)."""
    conn = duckdb.connect(DB_PATH)
    sub_df = conn.execute(
        "SELECT customer_id, week_date, is_treated, is_post_period, observed_weekly_spend_winsor "
        "FROM causal_experiment_data LIMIT 50000"
    ).df()
    conn.close()

    sub_df["week_date"] = pd.to_datetime(sub_df["week_date"])
    sub_df["treat_x_post"] = (sub_df["is_treated"] * sub_df["is_post_period"]).astype(float)
    sub_df["y_it"] = sub_df["observed_weekly_spend_winsor"].astype(float)

    # 1. Fit using linearmodels.PanelOLS
    df_panel = sub_df.set_index(["customer_id", "week_date"])
    mod = PanelOLS(
        df_panel["y_it"],
        df_panel[["treat_x_post"]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    panel_res = mod.fit(cov_type="clustered", cluster_entity=True, low_memory=True)
    panel_att = float(panel_res.params["treat_x_post"])
    panel_se = float(panel_res.std_errors["treat_x_post"])

    # 2. Fit using sparse FWL fallback
    sparse_res = _estimate_twfe_did_sparse(
        df=sub_df,
        outcome_col="observed_weekly_spend_winsor",
    )
    sparse_att = float(sparse_res["estimated_att"])
    sparse_se = float(sparse_res["std_err"])

    # 3. Assert estimates and standard errors match within 1e-6
    diff_att = abs(sparse_att - panel_att)
    diff_se = abs(sparse_se - panel_se)

    assert diff_att < 1e-6, f"ATT difference {diff_att:.2e} exceeds 1e-6 tolerance (Sparse={sparse_att}, PanelOLS={panel_att})"
    assert diff_se < 1e-6, f"SE difference {diff_se:.2e} exceeds 1e-6 tolerance (Sparse={sparse_se}, PanelOLS={panel_se})"


def test_q5_policy_simulator_formula() -> None:
    """Tests the Q5 fee-waiver Policy Simulator formula, extrapolation capping, and CI bounds."""
    from causal.decision_engine import simulate_q5_policy

    res_27 = simulate_q5_policy(scenario="hidden_degradation", waived_share=0.50, weeks=27)
    assert res_27["is_extrapolation"] is False
    assert res_27["n_q5"] == 395

    # Central net margin change = gross_margin * waived_share * N_Q5 * weeks * (-ATT_Q5_spend)
    expected_margin = 0.40 * 0.50 * 395 * 27 * (-res_27["att_q5_spend"])
    assert abs(res_27["net_margin_change_central"] - expected_margin) < 1e-4

    # Retention effect = (1 - waived_share) * ATT_Q5_active90
    expected_retention = 0.50 * res_27["att_q5_active90"]
    assert abs(res_27["q5_retention_effect"] - expected_retention) < 1e-4

    # 52 weeks is flagged as extrapolation
    res_52 = simulate_q5_policy(scenario="hidden_degradation", waived_share=0.50, weeks=52)
    assert res_52["is_extrapolation"] is True
    assert res_52["weeks"] == 52


def test_policy_simulator_zero_delta_and_monotonicity() -> None:
    """Verifies simulator produces zero delta at zero waiver, and is monotonic in waiver share."""
    from causal.decision_engine import simulate_q5_policy

    # 1. Zero delta at zero waiver share
    res_zero = simulate_q5_policy(scenario="hidden_degradation", waived_share=0.0)
    assert res_zero["net_margin_change_central"] == 0.0, "Central net margin must be exactly 0.0 at zero waiver"
    assert res_zero["net_margin_change_low"] == 0.0, "Low net margin must be exactly 0.0 at zero waiver"
    assert res_zero["net_margin_change_high"] == 0.0, "High net margin must be exactly 0.0 at zero waiver"

    # 2. Monotonicity in waiver share
    waiver_shares = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    sim_results = [
        simulate_q5_policy(scenario="hidden_degradation", waived_share=w, weeks=27)
        for w in waiver_shares
    ]

    margin_changes = [r["net_margin_change_central"] for r in sim_results]
    retention_effects = [r["q5_retention_effect"] for r in sim_results]

    # In hidden_degradation, ATT_Q5_spend is negative (-$3.26), so waiving fees strictly recovers margin
    for i in range(len(margin_changes) - 1):
        assert margin_changes[i] < margin_changes[i + 1], (
            f"Net margin change should strictly increase with waiver share: "
            f"share {waiver_shares[i]}={margin_changes[i]} vs share {waiver_shares[i+1]}={margin_changes[i+1]}"
        )

    # Active share retention effect = (1 - waived_share) * ATT_Q5_active90
    # Since ATT_Q5_active90 is negative (-3.7%), increasing waiver strictly reduces attrition (moves towards 0.0)
    for i in range(len(retention_effects) - 1):
        assert retention_effects[i] < retention_effects[i + 1], (
            f"Active share retention effect should strictly increase towards 0 with waiver share: "
            f"share {waiver_shares[i]}={retention_effects[i]} vs share {waiver_shares[i+1]}={retention_effects[i+1]}"
        )
    assert retention_effects[-1] == 0.0, "At 100% waiver, retention attrition shock must be 0.0"


def test_event_study_se_equals_quadratic_form_full_covariance() -> None:
    """Verifies event-study mean SE equals sqrt(w.T @ Sigma @ w) with nonzero off-diagonal correlation."""
    from causal.event_study import run_event_study

    conn = duckdb.connect(DB_PATH)
    panel_df = conn.execute("SELECT customer_id, week_date, is_treated, observed_weekly_spend_winsor FROM causal_experiment_data").df()
    conn.close()

    res = run_event_study(panel_df=panel_df)


    # 1. Check returned covariance matrix and weights
    assert "cov_matrix_post" in res, "Expected cov_matrix_post in event study result"
    assert "post_weights" in res, "Expected post_weights in event study result"
    assert "mean_post_se" in res, "Expected mean_post_se in event study result"

    Sigma_post = res["cov_matrix_post"]
    w = res["post_weights"]
    mean_post_se = res["mean_post_se"]

    M = len(w)
    assert M == 27, f"Expected 27 post-period weeks, got {M}"
    assert Sigma_post.shape == (M, M), f"Expected shape ({M}, {M}), got {Sigma_post.shape}"

    # 2. Check quadratic form identity: SE == sqrt(w.T @ Sigma_post @ w)
    var_quad = float(w @ Sigma_post @ w)
    se_quad = float(np.sqrt(var_quad))
    assert np.isclose(mean_post_se, se_quad, atol=1e-7), (
        f"Event-study SE ({mean_post_se}) must equal sqrt(w.T @ Sigma @ w) ({se_quad})"
    )

    # 3. Check nonzero off-diagonal correlation
    D = np.diag(1.0 / np.sqrt(np.diag(Sigma_post)))
    corr_matrix = D @ Sigma_post @ D
    off_diag_mask = ~np.eye(M, dtype=bool)
    off_diag_corrs = corr_matrix[off_diag_mask]

    assert len(off_diag_corrs) > 0
    assert (np.abs(off_diag_corrs) > 0.0).all(), "Expected all off-diagonal correlations to be strictly non-zero"
    assert off_diag_corrs.mean() > 0.05, f"Expected positive off-diagonal correlation, got {off_diag_corrs.mean()}"




