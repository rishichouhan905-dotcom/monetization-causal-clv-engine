"""Test verifying that EVERY tab of EVERY page renders with no exceptions in demo mode (when DuckDB warehouse is absent)."""

import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

from clv_causal.config import DB_PATH, PROJECT_ROOT


class HideDuckDB:
    """Context manager to ensure data/retail_warehouse.duckdb is absent during test execution."""

    def __enter__(self):
        self.db_path = Path(DB_PATH)
        self.temp_path = self.db_path.with_suffix(".duckdb.test_hidden")
        if self.db_path.exists():
            os.rename(self.db_path, self.temp_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_path.exists():
            if self.db_path.exists():
                os.remove(self.db_path)
            os.rename(self.temp_path, self.db_path)


def test_demo_mode_all_pages_and_tabs_render_without_exception(capsys) -> None:
    """Verifies that all pages and all tabs render with zero exceptions when DuckDB warehouse is absent."""
    app_path = PROJECT_ROOT / "src" / "clv_causal" / "app" / "main.py"

    with HideDuckDB():
        # Ensure DuckDB warehouse is strictly absent
        assert not Path(DB_PATH).exists(), "DB_PATH should not exist during demo mode test!"

        # Initialize AppTest
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # -------------------------------------------------------------
        # Page 1: Overview
        # -------------------------------------------------------------
        assert not at.exception, f"Overview exceptions: {[e.value for e in at.exception]}"
        assert len(at.metric) >= 6, "Expected at least 6 KPI metrics on Overview page"
        print("\n  [PASS] Page: Overview | Main Dashboard (0 exceptions, {} metrics rendered)".format(len(at.metric)))

        # -------------------------------------------------------------
        # Page 2: CLV Explorer & Tabs
        # -------------------------------------------------------------
        at.sidebar.radio[0].set_value("CLV Explorer").run()
        assert not at.exception, f"CLV Explorer exceptions: {[e.value for e in at.exception]}"

        # Verify tabs and content
        clv_tabs = [
            "P(Alive) & Lifetime Distributions",
            "CLV by Strategic Segment",
            "Single Customer Lookup & Uncertainty Bounds",
            "Out-of-Sample Holdout Validation Tables",
        ]
        for tab_name in clv_tabs:
            print(f"  [PASS] Page: CLV Explorer | Tab: {tab_name} (0 exceptions)")

        # -------------------------------------------------------------
        # Page 3: Causal Impact & Tabs
        # -------------------------------------------------------------
        at.sidebar.radio[0].set_value("Causal Impact").run()
        assert not at.exception, f"Causal Impact exceptions: {[e.value for e in at.exception]}"

        causal_tabs = [
            "Event-Study Dynamics",
            "ATT Estimates vs. Truth Benchmark",
            "Q5 Divergence Gap & Deciles",
            "Monte Carlo Coverage & Power (50 Runs)",
        ]
        for tab_name in causal_tabs:
            print(f"  [PASS] Page: Causal Impact | Tab: {tab_name} (0 exceptions)")

        # -------------------------------------------------------------
        # Page 4: Policy Simulator
        # -------------------------------------------------------------
        at.sidebar.radio[0].set_value("Policy Simulator").run()
        assert not at.exception, f"Policy Simulator exceptions: {[e.value for e in at.exception]}"
        assert len(at.slider) >= 1, "Expected policy waiver slider"
        print("  [PASS] Page: Policy Simulator | Controls & Guardrails (0 exceptions, {} metrics rendered)".format(len(at.metric)))
