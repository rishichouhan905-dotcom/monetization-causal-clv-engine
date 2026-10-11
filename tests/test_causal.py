"""Unit tests for causal impact modeling engine."""

import pytest
import pandas as pd
from causal.simulate_policy import simulate_policy_intervention
from causal.twfe_did import estimate_twfe_did
from causal.synthetic_control import estimate_synthetic_control
from clv_causal.config import DB_PATH, INTERVENTION_DATE


def test_causal_engine_full() -> None:
    """Tests causal policy simulation, TWFE DiD, and Synthetic Control estimation."""
    panel_df: pd.DataFrame = simulate_policy_intervention(db_path=DB_PATH)
    assert len(panel_df) > 0
    assert "observed_weekly_spend" in panel_df.columns
    assert "is_treated" in panel_df.columns
    assert "is_post_period" in panel_df.columns

    did_res = estimate_twfe_did(panel_df=panel_df, outcome_col="observed_weekly_orders")
    assert "estimated_att" in did_res
    assert "std_err" in did_res
    assert "ci_lower" in did_res
    assert "ci_upper" in did_res

    # Assert 95% CI covers true effect on weekly_orders
    tr_post = (panel_df["is_treated"] == 1) & (panel_df["is_post_period"] == 1)
    true_att_orders = float(panel_df.loc[tr_post, "realized_effect_orders"].mean())
    assert did_res["ci_lower"] <= true_att_orders <= did_res["ci_upper"], (
        f"DiD 95% CI [{did_res['ci_lower']:.5f}, {did_res['ci_upper']:.5f}] does not cover true effect ({true_att_orders:.5f})"
    )

    sc_res = estimate_synthetic_control(panel_df=panel_df, n_donors=50)
    assert "donor_weights_df" in sc_res
    assert "estimated_att" in sc_res

    weights = sc_res["donor_weights_df"]["weight"].values
    if len(weights) > 0:
        assert (weights >= 0.0).all()
        assert abs(weights.sum() - 1.0) < 1e-3
