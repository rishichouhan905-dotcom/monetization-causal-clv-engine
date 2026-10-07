import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from clv_causal.config import DB_PATH

def render_scenario_tab():
    st.markdown("## 🎯 Strategy & Executive Decision Studio")
    st.caption("Connecting probabilistic CLV predictions and causal impact lifts to 4-quadrant strategic customer segmentation and financial ROI decisions.")
    
    conn = duckdb.connect(DB_PATH)
    clv_df = conn.execute("SELECT * FROM pred_clv_summary").df()
    
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    seg_summary = conn.execute("SELECT * FROM decision_segment_summary").df() if 'decision_segment_summary' in tables else None
    conn.close()
    
    tab_matrix, tab_sim = st.tabs([
        "🧩 4-Quadrant Strategic Customer Matrix",
        "🎛️ Financial ROI & Causal Scenario Simulator"
    ])
    
    with tab_matrix:
        st.subheader("Priority 9: Strategic Customer Quadrant Segmentation Framework")
        st.caption("Categorizing customers into actionable strategic buckets based on 12-Month Probabilistic CLV and P(Alive) activity status.")
        
        if seg_summary is not None and len(seg_summary) > 0:
            st.dataframe(seg_summary.style.format({
                'avg_12m_clv': '${:,.2f}',
                'total_12m_clv': '${:,.2f}',
                'avg_p_alive': '{:.1%}',
                'avg_expected_orders': '{:.2f}',
                'pct_of_customers': '{:.1f}%',
                'pct_of_portfolio_val': '{:.1f}%'
            }), use_container_width=True)
            
            st.markdown("---")
            
            c_l, c_r = st.columns(2)
            with c_l:
                fig_cust = px.pie(
                    seg_summary,
                    names='strategy_segment',
                    values='customer_count',
                    title="Customer Base Share by Strategic Quadrant",
                    color_discrete_sequence=['#10B981', '#EF4444', '#6366F1', '#9CA3AF']
                )
                fig_cust.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cust, use_container_width=True)
                
            with c_r:
                fig_val = px.pie(
                    seg_summary,
                    names='strategy_segment',
                    values='total_12m_clv',
                    title="Portfolio Value Concentration by Strategic Quadrant ($)",
                    color_discrete_sequence=['#10B981', '#EF4444', '#6366F1', '#9CA3AF']
                )
                fig_val.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_val, use_container_width=True)

    with tab_sim:
        st.subheader("Interactive Financial ROI & Causal Policy Stress-Test")
        st.caption("Simulate macro discount rates, retention campaign lifts, pricing shifts, and Customer Acquisition Cost (CAC) ROI payback horizons.")
        
        st.sidebar.markdown("### ⚙️ Financial Decision Sliders")
        discount_rate = st.sidebar.slider("Annual Discount Rate (d)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f")
        retention_boost = st.sidebar.slider("Retention Campaign P(Alive) Boost (%)", min_value=0, max_value=30, value=10, step=1)
        pricing_lift = st.sidebar.slider("Monetization Policy Order Value Lift ($)", min_value=-20.0, max_value=50.0, value=15.0, step=2.5)
        cac_per_customer = st.sidebar.slider("Customer Acquisition Cost (CAC) ($/customer)", min_value=10.0, max_value=200.0, value=45.0, step=5.0)
        gross_margin_pct = st.sidebar.slider("Gross Margin Percentage (%)", min_value=10, max_value=80, value=40, step=5) / 100.0
        
        # Baseline Metrics
        base_12m_clv = clv_df['clv_12m'].sum()
        total_cust = len(clv_df)
        base_cac_spend = total_cust * cac_per_customer
        
        # Scenario Simulation
        sim_df = clv_df.copy()
        sim_p_alive = np.minimum(1.0, sim_df['p_alive'] * (1.0 + retention_boost / 100.0))
        sim_aov = sim_df['expected_avg_order_value'] + pricing_lift
        sim_12m_clv = sim_df['expected_purchases_12m'] * sim_p_alive * sim_aov / (1.0 + discount_rate)
        
        sim_total_12m_clv = sim_12m_clv.sum()
        sim_gross_margin = sim_total_12m_clv * gross_margin_pct
        sim_net_value = sim_gross_margin - base_cac_spend
        clv_delta = sim_total_12m_clv - base_12m_clv
        net_roi = (sim_gross_margin / max(1.0, base_cac_spend)) * 100.0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Simulated 1Y Portfolio Revenue", f"${sim_total_12m_clv:,.2f}", delta=f"${clv_delta:+,.2f}")
        col2.metric("Total Acquisition Spend (CAC)", f"${base_cac_spend:,.2f}")
        col3.metric("Simulated Gross Margin", f"${sim_gross_margin:,.2f}")
        col4.metric("Net Margin LTV:CAC ROI", f"{net_roi:.1f}%")
        
        st.markdown("---")
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.subheader("Baseline vs Simulated 1-Year Portfolio Value")
            comparison_df = pd.DataFrame({
                "Scenario": ["Baseline Portfolio", "Simulated Scenario"],
                "Portfolio Value ($)": [base_12m_clv, sim_total_12m_clv]
            })
            fig_bar = px.bar(
                comparison_df,
                x='Scenario',
                y='Portfolio Value ($)',
                color='Scenario',
                text='Portfolio Value ($)',
                color_discrete_map={"Baseline Portfolio": "#64748B", "Simulated Scenario": "#6366F1"},
                title="Revenue Comparison ($)"
            )
            fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_r:
            st.subheader("CAC Payback Horizon Sensitivity Curve")
            payback_months = np.linspace(1, 24, 24)
            cumulative_margin = [(sim_gross_margin / 12.0) * m for m in payback_months]
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=payback_months,
                y=cumulative_margin,
                mode='lines+markers',
                name='Cumulative Margin ($)',
                line=dict(color='#10B981', width=3)
            ))
            fig_line.add_hline(y=base_cac_spend, line_dash='dash', line_color='#EF4444', annotation_text=f"Total CAC Spend (${base_cac_spend:,.0f})")
            
            fig_line.update_layout(
                title="Cumulative Margin vs Acquisition Spend ($)",
                xaxis_title="Months Post-Acquisition",
                yaxis_title="Total Dollar Value ($)",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)
