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

    did_res = estimate_twfe_did(panel_df=panel_df)
    assert "estimated_att" in did_res
    assert "std_err" in did_res

    sc_res = estimate_synthetic_control(panel_df=panel_df)
    assert "donor_weights_df" in sc_res
    assert "estimated_att" in sc_res

    weights = sc_res["donor_weights_df"]["weight"].values
    if len(weights) > 0:
        assert (weights >= 0.0).all()
        assert abs(weights.sum() - 1.0) < 1e-3
