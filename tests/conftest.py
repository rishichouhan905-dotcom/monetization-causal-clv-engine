"""Pytest fixtures and hooks for automatic skipping of fulldata tests."""

from pathlib import Path
import pytest
from clv_causal.config import DB_PATH


def pytest_runtest_setup(item):
    """Automatically skips tests marked with 'fulldata' when retail_warehouse.duckdb is absent."""
    if "fulldata" in item.keywords:
        db_file = Path(DB_PATH)
        if not db_file.exists() or db_file.stat().st_size < 1_000_000:
            pytest.skip("Skipping fulldata test: full retail_warehouse.duckdb is absent")
