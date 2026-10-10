"""Unit tests for randomized policy simulation engine (src/causal/simulate_policy.py)."""

import duckdb
import numpy as np
import pytest
import pandas as pd

from causal.simulate_policy import simulate_policy_intervention
from clv_causal.config import DB_PATH, INTERVENTION_DATE


def test_simulate_policy_intervention_realized_effects() -> None:
    """Tests that realized simulated effects match configured parameters within tolerance."""
    treatment_share = 0.40
    order_rate_change = 0.15
    aov_change = 0.10
    top_clv_churn_hazard = 0.20
    seed = 42

    panel_df = simulate_policy_intervention(
        db_path=DB_PATH,
        intervention_date=INTERVENTION_DATE,
        treatment_share=treatment_share,
        order_rate_change=order_rate_change,
        aov_change=aov_change,
        top_clv_churn_hazard=top_clv_churn_hazard,
        random_seed=seed,
    )

    # 1. Check panel structure and column presence
    assert len(panel_df) > 0
    expected_cols = [
        "customer_id",
        "week_date",
        "is_treated",
        "is_post_period",
        "is_top_clv_quintile",
        "counterfactual_weekly_orders",
        "counterfactual_weekly_spend",
        "observed_weekly_orders",
        "observed_weekly_spend",
        "true_effect_orders",
        "true_effect_spend",
    ]
    for col in expected_cols:
        assert col in panel_df.columns

    # 2. Check realized treatment assignment share within 3% tolerance
    cust_treatment = panel_df.groupby("customer_id")["is_treated"].first()
    realized_share = float(cust_treatment.mean())
    assert abs(realized_share - treatment_share) < 0.03, f"Realized share {realized_share} outside tolerance of {treatment_share}"

    # 3. Check pre-period and control non-interference (true effect must be 0)
    pre_or_control = (panel_df["is_treated"] == 0) | (panel_df["is_post_period"] == 0)
    assert (panel_df.loc[pre_or_control, "true_effect_spend"] == 0).all()
    assert (panel_df.loc[pre_or_control, "true_effect_orders"] == 0).all()

    # 4. Check active treated post-t0 multiplicative effect
    active_post_treated = (
        (panel_df["is_treated"] == 1)
        & (panel_df["is_post_period"] == 1)
        & (panel_df["is_churned_top_clv"] == 0)
        & (panel_df["counterfactual_weekly_spend"] > 0)
    )

    if active_post_treated.sum() > 0:
        obs_orders = panel_df.loc[active_post_treated, "observed_weekly_orders"].values
        cf_orders = panel_df.loc[active_post_treated, "counterfactual_weekly_orders"].values
        order_ratios = obs_orders / cf_orders
        expected_order_mult = 1.0 + order_rate_change
        assert np.allclose(order_ratios, expected_order_mult, rtol=1e-2)

        obs_spend = panel_df.loc[active_post_treated, "observed_weekly_spend"].values
        cf_spend = panel_df.loc[active_post_treated, "counterfactual_weekly_spend"].values
        spend_ratios = obs_spend / cf_spend
        expected_spend_mult = (1.0 + order_rate_change) * (1.0 + aov_change)
        assert np.allclose(spend_ratios, expected_spend_mult, rtol=1e-2)

    # 5. Check churned top CLV zeroing post-t0
    churned_post_treated = (
        (panel_df["is_treated"] == 1)
        & (panel_df["is_post_period"] == 1)
        & (panel_df["is_churned_top_clv"] == 1)
    )
    if churned_post_treated.sum() > 0:
        assert (panel_df.loc[churned_post_treated, "observed_weekly_orders"] == 0).all()
        assert (panel_df.loc[churned_post_treated, "observed_weekly_spend"] == 0).all()

    # 6. Check DuckDB persistence
    conn = duckdb.connect(DB_PATH)
    duck_count = conn.execute("SELECT COUNT(*) FROM causal_experiment_data").fetchone()[0]
    conn.close()
    assert duck_count == len(panel_df)


def test_simulate_policy_intervention_configurability() -> None:
    """Tests that changing input effect parameters alters the realized ATT accordingly."""
    p1 = simulate_policy_intervention(order_rate_change=0.10, aov_change=0.05, random_seed=123)
    p2 = simulate_policy_intervention(order_rate_change=0.30, aov_change=0.20, random_seed=123)

    att1 = p1[(p1["is_treated"] == 1) & (p1["is_post_period"] == 1)]["true_effect_spend"].mean()
    att2 = p2[(p2["is_treated"] == 1) & (p2["is_post_period"] == 1)]["true_effect_spend"].mean()

    assert att2 > att1, f"Expected higher ATT for larger parameter settings: {att2} vs {att1}"
