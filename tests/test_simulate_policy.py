"""Unit tests for randomized balanced panel policy simulator (src/causal/simulate_policy.py)."""

import duckdb
import numpy as np
import pytest
import pandas as pd

from causal.simulate_policy import (
    get_ground_truth_effects,
    simulate_policy_intervention,
)
from clv_causal.config import DB_PATH, INTERVENTION_DATE


def test_simulate_policy_intervention_balanced_panel_and_effects() -> None:
    """Tests balanced panel generation, stratified treatment assignment, and realized effects."""
    treatment_share = 0.40
    order_rate_change = 0.15
    aov_change = 0.10
    top_spend_churn_cumulative = 0.20
    seed = 42

    panel_df = simulate_policy_intervention(
        db_path=DB_PATH,
        intervention_date=INTERVENTION_DATE,
        treatment_share=treatment_share,
        order_rate_change=order_rate_change,
        aov_change=aov_change,
        churn_scenario="baseline",
        top_spend_churn_cumulative=top_spend_churn_cumulative,
        random_seed=seed,
    )

    # 1. Check panel structure and column presence
    assert len(panel_df) > 300_000, f"Expected balanced panel with >300k rows, got {len(panel_df):,}"
    expected_cols = [
        "customer_id",
        "week_date",
        "is_treated",
        "is_post_period",
        "spend_quintile",
        "counterfactual_weekly_orders",
        "counterfactual_weekly_spend",
        "observed_weekly_orders",
        "observed_weekly_spend",
        "realized_effect_orders",
        "realized_effect_spend",
        "expected_effect_orders",
        "expected_effect_spend",
        "counterfactual_weekly_spend_winsor",
        "observed_weekly_spend_winsor",
        "realized_effect_spend_winsor",
        "expected_effect_spend_winsor",
        "counterfactual_active_last_90d",
        "observed_active_last_90d",
        "realized_effect_active_last_90d",
        "expected_effect_active_last_90d",
        "weekly_orders",
        "weekly_spend",
        "weekly_spend_winsor",
        "active_last_90d",
    ]
    for col in expected_cols:
        assert col in panel_df.columns, f"Missing expected column: {col}"

    # 2. Check balanced panel contains zero-purchase weeks
    zero_cf_weeks = (panel_df["counterfactual_weekly_orders"] == 0).sum()
    assert zero_cf_weeks > 250_000, f"Expected balanced panel with zero-purchase weeks, got {zero_cf_weeks:,}"

    # 3. Check customer eligibility: distinct customers must equal eligible count (< t0)
    n_customers = panel_df["customer_id"].nunique()
    assert n_customers == 4933, f"Expected 4,933 eligible customers, found {n_customers}"

    # 4. Check stratified treatment assignment (exactly 40% in each quintile)
    cust_quintiles = (
        panel_df.groupby("customer_id")[["spend_quintile", "is_treated"]].first()
    )
    for q in [1, 2, 3, 4, 5]:
        q_share = float(cust_quintiles[cust_quintiles["spend_quintile"] == q]["is_treated"].mean())
        assert abs(q_share - treatment_share) < 0.01, (
            f"Quintile {q} treatment share {q_share:.4f} deviates from configured {treatment_share}"
        )

    # 5. Check pre-period and control non-interference (effects must be 0)
    pre_or_control = (panel_df["is_treated"] == 0) | (panel_df["is_post_period"] == 0)
    assert (panel_df.loc[pre_or_control, "realized_effect_orders"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "expected_effect_orders"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "realized_effect_spend"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "expected_effect_spend"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "realized_effect_spend_winsor"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "expected_effect_spend_winsor"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "realized_effect_active_last_90d"] == 0.0).all()

    # 6. Check active treated post-t0 zero-week order addition & AOV scaling
    active_tr_zero = (
        (panel_df["is_treated"] == 1)
        & (panel_df["is_post_period"] == 1)
        & (~panel_df["is_churned"])
        & (panel_df["counterfactual_weekly_orders"] == 0)
    )
    added_orders = panel_df.loc[active_tr_zero, "observed_weekly_orders"]
    assert added_orders.sum() > 0, "Expected new orders added to zero weeks"

    # For added orders, spend must equal median_order_val * (1 + aov_change)
    added_with_spend = active_tr_zero & (panel_df["observed_weekly_orders"] == 1.0)
    obs_spend_added = panel_df.loc[added_with_spend, "observed_weekly_spend"].values
    expected_spend_added = (
        panel_df.loc[added_with_spend, "median_order_val"].values * (1.0 + aov_change)
    )
    assert np.allclose(obs_spend_added, expected_spend_added, rtol=1e-3)

    # 7. Check positive purchase week AOV scaling: observed spend = cf_spend * (1 + aov_change)
    active_tr_pos = (
        (panel_df["is_treated"] == 1)
        & (panel_df["is_post_period"] == 1)
        & (~panel_df["is_churned"])
        & (panel_df["counterfactual_weekly_orders"] > 0)
    )
    assert active_tr_pos.sum() > 0
    obs_pos_spend = panel_df.loc[active_tr_pos, "observed_weekly_spend"].values
    cf_pos_spend = panel_df.loc[active_tr_pos, "counterfactual_weekly_spend"].values
    assert np.allclose(obs_pos_spend, cf_pos_spend * (1.0 + aov_change), rtol=1e-3)

    # 8. Check churned top spend quintile zeroing post-churn week
    churned_rows = (
        (panel_df["is_treated"] == 1)
        & (panel_df["is_post_period"] == 1)
        & panel_df["is_churned"]
    )
    assert churned_rows.sum() > 0, "Expected churned rows in top quintile"
    assert (panel_df.loc[churned_rows, "observed_weekly_orders"] == 0.0).all()
    assert (panel_df.loc[churned_rows, "observed_weekly_spend"] == 0.0).all()

    # 9. Check DuckDB persistence
    conn = duckdb.connect(DB_PATH)
    duck_count = conn.execute("SELECT COUNT(*) FROM causal_experiment_data").fetchone()[0]
    conn.close()
    assert duck_count == len(panel_df)

    # 10. Check ground truth ATT effects structure
    effects = get_ground_truth_effects(panel_df)
    assert "overall" in effects
    assert "q1_to_q4_pooled" in effects
    assert "q5" in effects
    for key in ["overall", "q1_to_q4_pooled", "q5"]:
        for metric in ["realized_orders", "expected_orders", "realized_spend_winsor", "expected_spend_winsor"]:
            assert not np.isnan(effects[key][metric]), f"NaN found in {key} -> {metric}"


def test_simulate_policy_intervention_configurability_and_scenarios() -> None:
    """Tests that changing input effect parameters alters the realized ATT accordingly, and tests scenarios."""
    p_base = simulate_policy_intervention(
        churn_scenario="baseline",
        top_spend_churn_cumulative=0.20,
        random_seed=123,
        persist_to_db=False,
    )
    p_severe = simulate_policy_intervention(
        churn_scenario="severe",
        random_seed=123,
        persist_to_db=False,
    )

    eff_base = get_ground_truth_effects(p_base)
    eff_severe = get_ground_truth_effects(p_severe)

    # In Q5 (top spend), severe churn (20% weekly) produces significantly lower (more negative) ATT than baseline
    q5_base_spend = eff_base["q5"]["realized_spend_winsor"]
    q5_severe_spend = eff_severe["q5"]["realized_spend_winsor"]
    assert q5_severe_spend < q5_base_spend, (
        f"Severe churn should produce lower Q5 spend ATT ({q5_severe_spend}) than baseline ({q5_base_spend})"
    )


def test_simulate_policy_hidden_degradation() -> None:
    """Tests hidden_degradation scenario: cumulative churn 0.40 over post-period (~1.9% per week)."""
    p_hd = simulate_policy_intervention(
        churn_scenario="hidden_degradation",
        random_seed=42,
        persist_to_db=False,
    )
    eff_hd = get_ground_truth_effects(p_hd)

    # Non-churners in Q1-Q4 have positive lift, while Q5 has severe negative degradation
    q14_spend = eff_hd["q1_to_q4_pooled"]["realized_spend_winsor"]
    q5_spend = eff_hd["q5"]["realized_spend_winsor"]
    assert q14_spend > 0, f"Expected positive spend lift in Q1-Q4, got {q14_spend}"
    assert q5_spend < 0, f"Expected negative spend degradation in Q5, got {q5_spend}"
    assert eff_hd["q5"]["realized_orders"] < 0, "Expected negative orders in Q5 under hidden_degradation"
    assert eff_hd["q5"]["realized_active_90d"] < 0, "Expected negative 90-day retention in Q5"


def test_simulate_policy_confounded_assignment() -> None:
    """Tests confounded assignment mode: treatment correlated with pre-period spend growth."""
    p_conf = simulate_policy_intervention(
        assignment_mode="confounded",
        churn_scenario="baseline",
        random_seed=42,
        persist_to_db=False,
    )
    assert "spend_growth" in p_conf.columns
    # Check that treated customers have strictly higher mean spend growth than control customers
    treated_growth = p_conf[p_conf["is_treated"] == 1]["spend_growth"].mean()
    control_growth = p_conf[p_conf["is_treated"] == 0]["spend_growth"].mean()
    assert treated_growth > control_growth, (
        f"Expected treated growth ({treated_growth:.4f}) > control growth ({control_growth:.4f}) in confounded mode"
    )
