import pytest
from clv_causal.causal_engine.intervention_simulator import simulate_causal_intervention
from clv_causal.causal_engine.did_estimator import DifferenceInDifferencesEstimator
from clv_causal.causal_engine.synthetic_control import SyntheticControlEstimator

def test_causal_engine_full():
    panel_df = simulate_causal_intervention(intervention_date="2010-06-01", policy_lift_pct=0.15)
    assert len(panel_df) > 0
    assert 'observed_weekly_spend' in panel_df.columns
    assert 'is_treated_cohort' in panel_df.columns
    assert 'is_post_period' in panel_df.columns
    
    did = DifferenceInDifferencesEstimator()
    did_res = did.estimate_att(panel_df)
    assert 'att' in did_res
    assert 'std_err' in did_res
    assert 'parallel_trends_valid' in did_res
    assert 'event_study_df' in did_res
    assert len(did_res['event_study_df']) > 0
    
    sc = SyntheticControlEstimator()
    sc_res = sc.fit_predict(panel_df)
    assert 'weights_df' in sc_res
    assert 'att_synth' in sc_res
    assert 'perm_pval' in sc_res
    
    # Verify synthetic control weights non-negative and sum to 1.0
    weights = sc_res['weights_df']['weight'].values
    assert (weights >= 0.0).all()
    assert abs(weights.sum() - 1.0) < 1e-3
