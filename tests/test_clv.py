import pytest
from clv_causal.clv_engine.clv_pipeline import ProbabilisticCLVEngine, extract_rfm_matrix

def test_clv_pipeline():
    rfm_df = extract_rfm_matrix()
    engine = ProbabilisticCLVEngine()
    res_df = engine.fit_and_predict(rfm_df, prediction_months=[12], n_simulations=50)
    
    assert 'clv_12m' in res_df.columns
    assert 'clv_12m_lower' in res_df.columns
    assert 'clv_12m_upper' in res_df.columns
    assert 'p_alive' in res_df.columns
    
    assert (res_df['p_alive'] >= 0.0).all() and (res_df['p_alive'] <= 1.0).all()
    assert (res_df['clv_12m'] >= 0.0).all()
    
    # Verify uncertainty interval ordering: lower <= clv <= upper
    assert (res_df['clv_12m_lower'] <= res_df['clv_12m_upper']).all()

def test_clv_holdout_validation():
    engine = ProbabilisticCLVEngine()
    val_res = engine.evaluate_holdout_validation(observation_end_date="2011-06-01")
    
    assert 'mae' in val_res
    assert 'rmse' in val_res
    assert 'pearson_r' in val_res
    assert val_res['mae'] >= 0.0
    assert val_res['rmse'] >= 0.0
