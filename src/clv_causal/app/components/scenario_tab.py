"""Streamlit Strategy & Executive Decision Studio Component.

Renders 4-quadrant strategic customer matrix visualization and interactive
financial ROI simulator using simulator/margin.py.
"""

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clv_causal.config import DB_PATH, GROSS_MARGIN_PCT
from simulator.margin import calculate_margin_scenario


def render_scenario_tab() -> None:
    """Renders the Strategy & Executive Decision Studio tab in Streamlit."""
    st.markdown("## 🎯 Strategy & Executive Decision Studio")
    st.caption("Connecting probabilistic CLV predictions and causal impact lifts to 4-quadrant strategic customer segmentation and financial gross margin decisions.")

    conn = duckdb.connect(DB_PATH)
    clv_df: pd.DataFrame = conn.execute("SELECT * FROM pred_clv_summary").df()

    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    seg_summary: pd.DataFrame = (
        conn.execute("SELECT * FROM decision_segment_summary").df()
        if "decision_segment_summary" in tables
        else pd.DataFrame()
    )
    conn.close()

    tab_matrix, tab_sim = st.tabs([
        "🧩 4-Quadrant Strategic Customer Matrix",
        "🎛️ Financial Gross Margin & ROI Simulator",
    ])

    with tab_matrix:
        st.subheader("Strategic Customer Quadrant Segmentation Framework")
        st.caption("Categorizing customers into actionable strategic buckets based on 12-Month Probabilistic CLV and P(Alive) activity status.")

        if not seg_summary.empty:
            st.dataframe(
                seg_summary.style.format(
                    {
                        "avg_12m_clv": "${:,.2f}",
                        "total_12m_clv": "${:,.2f}",
                        "avg_p_alive": "{:.1%}",
                        "avg_expected_orders": "{:.2f}",
                        "pct_of_customers": "{:.1f}%",
                        "pct_of_portfolio_val": "{:.1f}%",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("---")

            c_l, c_r = st.columns(2)
            with c_l:
                fig_cust = px.pie(
                    seg_summary,
                    names="strategy_segment",
                    values="customer_count",
                    title="Customer Base Share by Strategic Quadrant",
                    color_discrete_sequence=["#10B981", "#EF4444", "#6366F1", "#9CA3AF"],
                )
                fig_cust.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cust, use_container_width=True)

            with c_r:
                fig_val = px.pie(
                    seg_summary,
                    names="strategy_segment",
                    values="total_12m_clv",
                    title="Portfolio Value Concentration by Strategic Quadrant ($)",
                    color_discrete_sequence=["#10B981", "#EF4444", "#6366F1", "#9CA3AF"],
                )
                fig_val.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_val, use_container_width=True)

    with tab_sim:
        st.subheader("Interactive Financial Gross Margin & Campaign Simulator")
        st.caption("Simulate intervention lifts. Zero slider values strictly yield zero financial delta ($0.00).")

        st.sidebar.markdown("### ⚙️ Financial Scenario Sliders")
        freq_lift_pct = (
            st.sidebar.slider(
                "Order Frequency Lift (%)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )
            / 100.0
        )
        aov_lift_pct = (
            st.sidebar.slider(
                "Average Order Value (AOV) Lift (%)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )
            / 100.0
        )
        p_alive_boost = (
            st.sidebar.slider(
                "Retention P(Alive) Boost (+)",
                min_value=0.00,
                max_value=0.30,
                value=0.00,
                step=0.01,
            )
        )
        gross_margin_pct = (
            st.sidebar.slider(
                "Gross Margin Percentage (%)",
                min_value=10,
                max_value=80,
                value=int(GROSS_MARGIN_PCT * 100),
                step=5,
            )
            / 100.0
        )
        campaign_cost_per_cust = float(
            st.sidebar.slider(
                "Campaign Marketing Cost per Customer ($)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0,
            )
        )

        # Run margin scenario using simulator/margin.py
        sim_res = calculate_margin_scenario(
            clv_df=clv_df,
            freq_lift_pct=freq_lift_pct,
            aov_lift_pct=aov_lift_pct,
            p_alive_lift=p_alive_boost,
            gross_margin_pct=gross_margin_pct,
            campaign_cost_per_cust=campaign_cost_per_cust,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline Gross Margin", f"${sim_res['total_baseline_margin']:,.2f}")
        col2.metric(
            "Simulated Gross Margin",
            f"${sim_res['total_scenario_margin']:,.2f}",
            delta=f"${sim_res['total_delta_margin']:+,.2f} Margin Delta",
        )
        col3.metric("Campaign Operational Cost", f"${sim_res['total_campaign_cost']:,.2f}")
        col4.metric(
            "Net Financial Impact",
            f"${sim_res['net_financial_impact']:+,.2f}",
            delta=f"{sim_res['roi_pct']:+.1f}% Campaign ROI" if sim_res['total_campaign_cost'] > 0 else "No Campaign Cost",
        )

        st.markdown("---")

        if abs(sim_res["total_delta_margin"]) < 1e-4:
            st.success("✅ **Zero-Slider Baseline State**: All lift sliders are set to 0%, resulting in **$0.00 exact margin delta**.")
        else:
            st.info(f"💡 **Simulated Intervention Impact**: Lifts yield an incremental **${sim_res['total_delta_margin']:,.2f} in Gross Margin Dollars**.")

        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Baseline vs Simulated Gross Margin ($)")
            comparison_df = pd.DataFrame({
                "Scenario": ["Baseline Portfolio Margin", "Simulated Scenario Margin"],
                "Gross Margin ($)": [sim_res["total_baseline_margin"], sim_res["total_scenario_margin"]],
            })
            fig_bar = px.bar(
                comparison_df,
                x="Scenario",
                y="Gross Margin ($)",
                color="Scenario",
                text="Gross Margin ($)",
                color_discrete_map={"Baseline Portfolio Margin": "#64748B", "Simulated Scenario Margin": "#10B981"},
                title="Portfolio Gross Margin Comparison ($)",
            )
            fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_bar.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_r:
            st.subheader("Payback Horizon Sensitivity (Months)")
            payback_months = np.linspace(1, 24, 24)
            monthly_margin_delta = sim_res["total_delta_margin"] / 12.0
            cum_margin = [monthly_margin_delta * m for m in payback_months]

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=payback_months,
                y=cum_margin,
                mode="lines+markers",
                name="Cumulative Incremental Gross Margin ($)",
                line=dict(color="#10B981", width=3),
            ))

            if sim_res["total_campaign_cost"] > 0:
                fig_line.add_hline(
                    y=sim_res["total_campaign_cost"],
                    line_dash="dash",
                    line_color="#EF4444",
                    annotation_text=f"Campaign Cost (${sim_res['total_campaign_cost']:,.0f})",
                )

            fig_line.update_layout(
                title="Cumulative Incremental Gross Margin Payback ($)",
                xaxis_title="Months Post-Intervention",
                yaxis_title="Gross Margin ($)",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_line, use_container_width=True)
