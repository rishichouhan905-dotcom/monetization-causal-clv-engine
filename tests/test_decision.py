"""Unit tests for business decision and strategy engine."""

import pytest
from causal.decision_engine import BusinessDecisionEngine


def test_decision_framework() -> None:
    """Tests 4-quadrant customer segmentation and financial scenario analysis."""
    engine = BusinessDecisionEngine()
    res = engine.evaluate_decision_framework()

    assert "segment_summary" in res
    assert "decision_scenarios" in res

    seg_df = res["segment_summary"]
    assert len(seg_df) > 0
    assert "strategy_segment" in seg_df.columns
    assert "customer_count" in seg_df.columns
    assert "total_12m_clv" in seg_df.columns

    scenario_df = res["decision_scenarios"]
    assert len(scenario_df) > 0
    assert "ROI_Percent" in scenario_df.columns
