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


def test_causal_impact_pre_trend_wald_matches_csv_and_scipy() -> None:
    """Verifies that the pre-trend Wald statistic and p-value on Causal Impact match results/baseline_assignment.csv and scipy.stats.chi2.sf."""
    import re
    import pandas as pd
    from scipy.stats import chi2
    from streamlit.testing.v1 import AppTest

    # 1. Read ground-truth benchmark from CSV
    csv_path = PROJECT_ROOT / "results" / "baseline_assignment.csv"
    assert csv_path.exists(), f"Missing required file: {csv_path}"

    df_csv = pd.read_csv(csv_path)
    csv_data = dict(zip(df_csv["metric"], df_csv["value"]))

    assert "pre_trend_wald_chi2" in csv_data, "pre_trend_wald_chi2 missing from baseline_assignment.csv"
    assert "pre_trend_wald_df" in csv_data, "pre_trend_wald_df missing from baseline_assignment.csv"
    assert "pre_trend_wald_p" in csv_data, "pre_trend_wald_p missing from baseline_assignment.csv"

    csv_chi2 = float(csv_data["pre_trend_wald_chi2"])
    csv_df = int(float(csv_data["pre_trend_wald_df"]))
    csv_p = float(csv_data["pre_trend_wald_p"])

    assert abs(csv_chi2 - 4.1326) < 1e-3, f"Expected chi2 ~ 4.1326, got {csv_chi2}"
    assert abs(csv_p - 0.9025) < 1e-3, f"Expected p ~ 0.9025, got {csv_p}"

    # 2. Launch Streamlit headless app and navigate to Causal Impact
    app_path = PROJECT_ROOT / "src" / "clv_causal" / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("Causal Impact").run()
    assert not at.exception, f"Exceptions on Causal Impact page: {[e.value for e in at.exception]}"

    # 3. Test both scenarios: 'baseline' and 'hidden_degradation'
    for scenario in ["baseline", "hidden_degradation"]:
        at.selectbox[0].set_value(scenario).run()
        assert not at.exception, f"Exceptions on scenario '{scenario}': {[e.value for e in at.exception]}"

        # Find the callout containing the pre-trend Wald test
        wald_boxes = [info.value for info in at.info if "Pre-Trend Wald Test" in info.value]
        assert len(wald_boxes) >= 1, f"Pre-Trend Wald Test callout box missing for scenario '{scenario}'"
        wald_text = wald_boxes[0]

        # Extract Chi^2, df, and p from displayed callout text
        m_chi2 = re.search(r"(?:Chi\^?2|chi2)\s*=\s*([0-9.]+)", wald_text, re.IGNORECASE)
        m_df = re.search(r"df\s*=\s*([0-9]+)", wald_text)
        m_p = re.search(r"p\s*=\s*([0-9.]+)", wald_text)

        assert m_chi2, f"Could not parse Chi^2 statistic from: '{wald_text}'"
        assert m_df, f"Could not parse df from: '{wald_text}'"
        assert m_p, f"Could not parse p-value from: '{wald_text}'"

        page_chi2 = float(m_chi2.group(1))
        page_df = int(m_df.group(1))
        page_p = float(m_p.group(1))

        # Check 1: Fail if the page and the CSV differ
        assert abs(page_chi2 - csv_chi2) < 1e-4, (
            f"Scenario '{scenario}': Page chi2 ({page_chi2}) differs from CSV ({csv_chi2})"
        )
        assert page_df == csv_df, (
            f"Scenario '{scenario}': Page df ({page_df}) differs from CSV ({csv_df})"
        )
        assert abs(page_p - csv_p) < 1e-3, (
            f"Scenario '{scenario}': Page p-value ({page_p}) differs from CSV ({csv_p})"
        )

        # Check 2: Check scipy.stats.chi2.sf(chi2, df) equals displayed p within 1e-3
        theoretical_p = float(chi2.sf(page_chi2, page_df))
        assert abs(theoretical_p - page_p) < 1e-3, (
            f"Scenario '{scenario}': scipy.stats.chi2.sf({page_chi2}, {page_df}) = {theoretical_p:.5f} "
            f"does not match displayed p = {page_p:.5f} within 1e-3"
        )


def test_app_runs_twice_in_a_row_layout_stays_wide(monkeypatch) -> None:
    """Verifies that running the app twice in a row keeps the layout wide and page title correct."""
    from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
    from streamlit.testing.v1.local_script_runner import LocalScriptRunner

    captured_configs = []
    orig_run = LocalScriptRunner.run

    def hooked_run(self, *args, **kwargs):
        res = orig_run(self, *args, **kwargs)
        for m in self.forward_msgs():
            if m.HasField("page_config_changed"):
                captured_configs.append(m.page_config_changed)
        return res

    monkeypatch.setattr(LocalScriptRunner, "run", hooked_run)

    # Test the server entry point (app/main.py)
    app_path = PROJECT_ROOT / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)

    # Run 1
    at.run()
    assert not at.exception, f"Run 1 exceptions: {[e.value for e in at.exception]}"
    assert len(captured_configs) >= 1, "Page config was not emitted on run 1"
    assert captured_configs[0].layout == PageConfigProto.WIDE, (
        f"Run 1 layout is not WIDE: {captured_configs[0].layout}"
    )
    assert captured_configs[0].title == "Monetization Causal CLV Engine"
    assert not captured_configs[0].favicon, "Page icon should have no emoji"

    # Run 2
    at.run()
    assert not at.exception, f"Run 2 exceptions: {[e.value for e in at.exception]}"
    assert len(captured_configs) >= 2, (
        "Page config was not emitted on run 2 (wide layout lost on later runs)"
    )
    assert captured_configs[1].layout == PageConfigProto.WIDE, (
        f"Run 2 layout is not WIDE: {captured_configs[1].layout}"
    )
    assert captured_configs[1].title == "Monetization Causal CLV Engine"
    assert not captured_configs[1].favicon, "Page icon should have no emoji"


def test_set_page_config_only_called_in_main() -> None:
    """Verifies that st.set_page_config is called only once in the entire repository, inside main()."""
    app_src = PROJECT_ROOT / "src" / "clv_causal" / "app"
    matches = []
    for py_file in app_src.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), 1):
            if "set_page_config" in line and not line.strip().startswith("#"):
                matches.append((str(py_file.relative_to(PROJECT_ROOT)), idx, line.strip()))

    assert len(matches) == 1, f"Expected exactly 1 call to set_page_config, found: {matches}"
    file_rel, line_no, line_content = matches[0]
    assert "main.py" in file_rel, f"set_page_config should be in main.py, found in {file_rel}"

