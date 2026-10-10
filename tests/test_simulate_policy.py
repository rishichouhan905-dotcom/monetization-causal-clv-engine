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
        "true_effect_orders",
        "true_effect_spend",
        "counterfactual_weekly_spend_winsor",
        "observed_weekly_spend_winsor",
        "true_effect_spend_winsor",
        "counterfactual_active_last_90d",
        "observed_active_last_90d",
        "true_effect_active_last_90d",
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

    # 5. Check pre-period and control non-interference (true effect must be 0)
    pre_or_control = (panel_df["is_treated"] == 0) | (panel_df["is_post_period"] == 0)
    assert (panel_df.loc[pre_or_control, "true_effect_orders"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "true_effect_spend"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "true_effect_spend_winsor"] == 0.0).all()
    assert (panel_df.loc[pre_or_control, "true_effect_active_last_90d"] == 0.0).all()

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

    # 10. Check ground truth ATT effects
    effects = get_ground_truth_effects(panel_df)
    assert "weekly_orders" in effects
    assert "weekly_spend_raw" in effects
    assert "weekly_spend_winsor" in effects
    assert "active_last_90d" in effects
    assert not np.isnan(effects["weekly_orders"])
    assert not np.isnan(effects["weekly_spend_winsor"])
    assert not np.isnan(effects["active_last_90d"])


def test_simulate_policy_intervention_configurability() -> None:
    """Tests that changing input effect parameters alters the realized ATT accordingly."""
    p1 = simulate_policy_intervention(order_rate_change=0.05, aov_change=0.05, random_seed=123)
    p2 = simulate_policy_intervention(order_rate_change=0.25, aov_change=0.25, random_seed=123)

    eff1 = get_ground_truth_effects(p1)
    eff2 = get_ground_truth_effects(p2)

    # In lower quintiles without churn (Q1-Q4), effect must be strictly larger with higher parameters
    tr_post1_q1 = p1[(p1["is_treated"] == 1) & (p1["is_post_period"] == 1) & (p1["spend_quintile"] == 1)]
    tr_post2_q1 = p2[(p2["is_treated"] == 1) & (p2["is_post_period"] == 1) & (p2["spend_quintile"] == 1)]

    assert tr_post2_q1["true_effect_orders"].mean() > tr_post1_q1["true_effect_orders"].mean()
    assert tr_post2_q1["true_effect_spend"].mean() > tr_post1_q1["true_effect_spend"].mean()
