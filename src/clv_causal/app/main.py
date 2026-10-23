"""Enterprise Probabilistic CLV & Causal Decision Platform.

Main Streamlit application entry point connecting DuckDB data warehouse,
BG/NBD and Gamma-Gamma probabilistic CLV distributions, quasi-experimental
TWFE and event-study causal inference, and executive policy simulations.
"""

from pathlib import Path
import sys

# Ensure src/ is on sys.path for self-contained execution and AppTest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from clv_causal.app.components.overview_tab import render_overview_tab
from clv_causal.app.components.clv_tab import render_clv_tab
from clv_causal.app.components.causal_tab import render_causal_tab
from clv_causal.app.components.scenario_tab import render_scenario_tab

def main() -> None:
    """Main application loop with sidebar navigation across the 4 platform pages."""
    st.set_page_config(
        page_title="Monetization Causal CLV Engine",
        layout="wide",
    )

    # Custom Styling with Responsive Metric Cards
    st.markdown(
        """
        <style>
        /* Responsive Metric Card Styling to Prevent Truncation */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            min-height: 105px;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            white-space: normal !important;
            line-height: 1.25;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700;
            color: #0F172A;
            white-space: normal !important;
            word-break: break-word !important;
            line-height: 1.25;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.82rem;
            white-space: normal !important;
        }
        /* Alert Box Styling */
        .stAlert {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Enterprise Decision Engine")
    st.sidebar.caption("Probabilistic CLV & Quasi-Experimental Causal Platform")

    pages = [
        "Overview",
        "CLV Explorer",
        "Causal Impact",
        "Policy Simulator",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**System Architecture**\n"
        "- **Warehouse**: DuckDB Star Schema (`retail_warehouse.duckdb`)\n"
        "- **CLV Models**: Continuous BG/NBD & Gamma-Gamma (MLE)\n"
        "- **Econometrics**: PanelOLS DiD & Event Study\n"
        "- **Policy Engine**: Asymmetric Quintile Simulation\n"
    )
    st.sidebar.caption("UCI Online Retail II Benchmark Engine")

    # Render selected page
    if selected_page == "Overview":
        render_overview_tab()
    elif selected_page == "CLV Explorer":
        render_clv_tab()
    elif selected_page == "Causal Impact":
        render_causal_tab()
    elif selected_page == "Policy Simulator":
        render_scenario_tab()


if __name__ == "__main__":
    main()
