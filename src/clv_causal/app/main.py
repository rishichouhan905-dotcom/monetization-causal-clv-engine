import streamlit as st
import os
import sys

from clv_causal.app.components.overview_tab import render_overview_tab
from clv_causal.app.components.clv_tab import render_clv_tab
from clv_causal.app.components.causal_tab import render_causal_tab
from clv_causal.app.components.scenario_tab import render_scenario_tab

st.set_page_config(
    page_title="Enterprise Probabilistic CLV & Causal Decision Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("🚀 Enterprise Decision Engine")
    st.sidebar.caption("Probabilistic CLV & Causal Impact Decision Platform")
    
    st.title("📈 Enterprise Probabilistic CLV & Causal Impact Decision Platform")
    st.markdown("### Connecting authentic transaction data, probabilistic lifetime distributions, quasi-experimental causal inference, and executive decision strategy.")
    st.markdown("---")
    
    tabs = st.tabs([
        "📊 Executive Overview",
        "🔮 Probabilistic CLV Studio",
        "🧪 Causal Impact & Quasi-Experimentation",
        "🎯 Strategy & Decision Studio"
    ])
    
    with tabs[0]:
        render_overview_tab()
        
    with tabs[1]:
        render_clv_tab()
        
    with tabs[2]:
        render_causal_tab()
        
    with tabs[3]:
        render_scenario_tab()
        
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tech Stack**: DuckDB Analytical Warehouse | Lifetimes (BG/NBD & Gamma-Gamma) | Statsmodels / SciPy Econometrics | Streamlit BI Decision App")

if __name__ == "__main__":
    main()
