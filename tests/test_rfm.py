"""Unit tests for RFM matrix extraction."""

import pytest
from clv_causal.clv_engine.rfm_builder import extract_rfm_matrix


def test_rfm_extraction() -> None:
    """Tests extraction of RFM matrix from DuckDB warehouse."""
    rfm_df = extract_rfm_matrix()
    assert len(rfm_df) > 0, "RFM matrix is empty!"
    assert "frequency" in rfm_df.columns
    assert "recency" in rfm_df.columns
    assert "T" in rfm_df.columns
    assert "monetary_value" in rfm_df.columns

    assert (rfm_df["frequency"] >= 0).all()
    assert (rfm_df["recency"] >= 0).all()
    assert (rfm_df["T"] >= rfm_df["recency"]).all()
