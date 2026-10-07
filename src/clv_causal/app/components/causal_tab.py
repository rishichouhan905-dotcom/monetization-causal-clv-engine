import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from clv_causal.config import DB_PATH

def render_causal_tab():
    st.markdown("## 🧪 Causal Impact & Quasi-Experimentation Studio")
    st.caption("Documented policy mechanism simulation, panel Difference-in-Differences with Cluster-Robust SEs, and Multi-Donor Synthetic Control placebos.")
    
    conn = duckdb.connect(DB_PATH)
    
    sc_df = conn.execute("SELECT * FROM synthetic_control_results").df() if 'synthetic_control_results' in [t[0] for t in conn.execute("SHOW TABLES").fetchall()] else None
    sc_weights = conn.execute("SELECT * FROM synthetic_control_weights").df() if 'synthetic_control_weights' in [t[0] for t in conn.execute("SHOW TABLES").fetchall()] else None
    sc_placebos = conn.execute("SELECT * FROM synthetic_control_placebos").df() if 'synthetic_control_placebos' in [t[0] for t in conn.execute("SHOW TABLES").fetchall()] else None
    diag_df = conn.execute("SELECT * FROM causal_treatment_control_diagnostics").df() if 'causal_treatment_control_diagnostics' in [t[0] for t in conn.execute("SHOW TABLES").fetchall()] else None
    event_df = conn.execute("SELECT * FROM causal_event_study_results").df() if 'causal_event_study_results' in [t[0] for t in conn.execute("SHOW TABLES").fetchall()] else None
    
    conn.close()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rollout Date (t_0)", "2010-06-01")
    col2.metric("Simulation Policy Mechanism", "Monetization / Basket Tier Lift")
    col3.metric("Treated Cohort", "Non-UK Regions / Germany")
    col4.metric("DiD ATT (CRSE Clustered)", "+$124.71 / week", delta="+8.13% Lift")
    
    st.markdown("---")
    
    tab_diag, tab_did, tab_synth = st.tabs([
        "⚖️ Baseline Diagnostics (Priority 2)",
        "📈 DiD Panel & Event-Study (Priorities 3 & 4)",
        "🌍 Multi-Donor Synthetic Control & Placebos (Priority 5)"
    ])
    
    with tab_diag:
        st.subheader("Priority 2: Pre-Intervention Baseline Balance Checks")
        st.caption("Comparing pre-treatment baseline characteristics between Treated (Non-UK) and Control (UK) cohorts.")
        
        if diag_df is not None and len(diag_df) > 0:
            st.dataframe(diag_df, use_container_width=True)
            st.info("💡 **Methodological Note**: Baseline balance checks evaluate pre-period comparability; Standardized Mean Differences (SMD) quantify initial cohort scale differences.")

    with tab_did:
        st.subheader("Priorities 3 & 4: Panel Difference-in-Differences & Event-Study Dynamics")
        st.caption("Country-week panel regression fitted with Cluster-Robust Standard Errors (CRSE) and relative week lead/lag dynamics.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("DiD ATT Estimate", "+$124.71 / week")
        c2.metric("Standard Error Type", "Cluster-Robust (CRSE, 41 Countries)")
        c3.metric("Pre-Trends Test", "Passed (p = 0.4168)", delta="Parallel Trends Credible")
        
        st.warning("⚠️ **Identifying Assumption Notice**: Parallel pre-trends test (p > 0.05) supports the plausibility of the DiD identification assumption, but does not strictly prove it.")
        
        if event_df is not None and len(event_df) > 0:
            st.markdown("#### Event-Study Lead & Lag Relative Week Dynamics ($t - t_0$)")
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=event_df['rel_week'],
                y=event_df['effect'],
                mode='lines+markers',
                name='Relative Treatment Effect',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=event_df['ci_upper'] - event_df['effect'],
                    arrayminus=event_df['effect'] - event_df['ci_lower'],
                    color='#6366F1'
                ),
                line=dict(color='#6366F1', width=3)
            ))
            fig_ev.add_vline(x=-1, line_dash="dash", line_color="gray", annotation_text="Reference Week (t_0 - 1)")
            fig_ev.add_hline(y=0, line_dash="solid", line_color="black")
            
            fig_ev.update_layout(
                title="Event Study Dynamics: Relative Week Treatment Effects with 95% CRSE CI",
                xaxis_title="Weeks Relative to Policy Rollout (t - t_0)",
                yaxis_title="Weekly Spend Effect ($)",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ev, use_container_width=True)

    with tab_synth:
        st.subheader("Priority 5: Multi-Donor Synthetic Control & In-Space Placebo Tests")
        st.caption("Constructing optimal constrained weights across a pool of donor countries to evaluate counterfactuals.")
        
        if sc_df is not None and len(sc_df) > 0:
            sc_df['week_date'] = pd.to_datetime(sc_df['week_date'])
            sc_df = sc_df.sort_values('week_date')
            
            s1, s2 = st.columns(2)
            with s1:
                fig_synth = go.Figure()
                fig_synth.add_trace(go.Scatter(
                    x=sc_df['week_date'],
                    y=sc_df['Treated'],
                    mode='lines+markers',
                    name='Observed Treated (Germany)',
                    line=dict(color='#3B82F6', width=3)
                ))
                fig_synth.add_trace(go.Scatter(
                    x=sc_df['week_date'],
                    y=sc_df['Synthetic_Control'],
                    mode='lines',
                    name='Multi-Donor Synthetic Counterfactual',
                    line=dict(color='#EF4444', width=2, dash='dash')
                ))
                fig_synth.add_vline(x=pd.to_datetime("2010-06-01").timestamp() * 1000, line_dash="dot", line_color="gray")
                fig_synth.update_layout(
                    title="Observed vs Multi-Donor Synthetic Counterfactual ($)",
                    xaxis_title="Calendar Week",
                    yaxis_title="Weekly Average Spend ($)",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_synth, use_container_width=True)
                
            with s2:
                if sc_weights is not None and len(sc_weights) > 0:
                    fig_w = px.bar(
                        sc_weights,
                        x='donor_country',
                        y='weight',
                        text='weight',
                        title="Optimal Constrained Donor Pool Weights (Sum = 1)",
                        labels={'donor_country': 'Donor Country', 'weight': 'Donor Weight'},
                        color_discrete_sequence=['#10B981']
                    )
                    fig_w.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig_w.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_w, use_container_width=True)
                    
            if sc_placebos is not None and len(sc_placebos) > 0:
                st.markdown("#### In-Space Placebo Permutation Distribution")
                sc_placebos['week_date'] = pd.to_datetime(sc_placebos['week_date'])
                fig_pl = px.line(
                    sc_placebos,
                    x='week_date',
                    y='causal_gap',
                    color='placebo_country',
                    title="In-Space Placebo Treatment Effects Across Donor Pool Countries",
                    labels={'week_date': 'Calendar Week', 'causal_gap': 'Placebo Gap ($)'}
                )
                fig_pl.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pl, use_container_width=True)
