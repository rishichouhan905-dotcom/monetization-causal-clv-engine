"""Streamlit Policy Simulator Component.

Simulates the financial and customer retention impact of the Q5 fee-waiver intervention policy,
reusing simulate_q5_policy from decision_engine.py with low/central/high uncertainty bands,
active share guardrail monitors, and extrapolation warnings.
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from causal.decision_engine import simulate_q5_policy
from clv_causal.config import DB_PATH, GROSS_MARGIN_PCT, PROJECT_ROOT


def render_scenario_tab() -> None:
    """Renders the Policy Simulator page in Streamlit."""
    st.markdown("## 🎛️ Policy Simulator: Q5 Fee Waiver & Margin Strategy")

    # 1. Mandatory Causal Policy Banner
    st.warning(
        "**Real transactions (UCI Online Retail II) with a SIMULATED policy**\n\n"
        "The evaluated policy is a pricing and bundling change that lifts engagement and purchase frequency for most customers "
        "(Quintiles 1–4) but alienates high-frequency top spenders (Quintile 5)."
    )

    st.caption(
        "Calibrate fee exemption waivers for Quintile 5 (top spenders) to mitigate spend compression and preserve active share."
    )

    # 2. Controls and Sliders
    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        st.markdown("### Policy Controls")

        selected_scenario = st.selectbox(
            "Causal Policy Scenario",
            ["hidden_degradation", "baseline"],
            index=0,
            help="Select the stress condition. Hidden Degradation: 40% Q5 churn (~1.9%/week). Baseline: 20% Q5 churn.",
        )

        waived_share_pct = st.slider(
            "Q5 Fee Waiver Share (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Fraction of top-spend (Quintile 5) customers granted fee exemptions.",
        )
        waived_share = waived_share_pct / 100.0

        horizon_weeks = st.slider(
            "Evaluation Horizon (Weeks)",
            min_value=1,
            max_value=52,
            value=27,
            step=1,
            help="Simulation horizon in weeks. Capped at 52 weeks. Horizons beyond 27 weeks are an extrapolation of the estimated effect.",
        )

        gross_margin_input = st.slider(
            "Gross Margin (%)",
            min_value=10,
            max_value=80,
            value=int(GROSS_MARGIN_PCT * 100),
            step=5,
            help="Product gross margin percentage applied to incremental customer spend (from config).",
        )
        gross_margin_pct = gross_margin_input / 100.0

    # 3. Horizon Extrapolation Notice
    if horizon_weeks > 27:
        st.warning(
            f"**Extrapolation Notice**: Horizon is set to **{horizon_weeks} weeks**. "
            "Projections beyond 27 weeks are an **extrapolation of the estimated treatment effect**."
        )
    else:
        st.success(
            f"**Empirically Validated Window**: Evaluation horizon ({horizon_weeks} weeks) is within the "
            "empirically observed 27-week post-intervention study window."
        )

    # 4. Run Policy Simulator via decision_engine.py
    sim_res = simulate_q5_policy(
        scenario=selected_scenario,
        waived_share=waived_share,
        gross_margin_pct=gross_margin_pct,
        weeks=horizon_weeks,
    )

    net_central = sim_res["net_margin_change_central"]
    net_low = sim_res["net_margin_change_low"]
    net_high = sim_res["net_margin_change_high"]
    active_share_effect = sim_res["q5_retention_effect"]
    active_share_pp = active_share_effect * 100.0
    n_q5 = sim_res["n_q5"]

    with col_c2:
        st.markdown("### Simulated Policy Outcomes")

        m1, m2, m3 = st.columns(3)
        # Central estimate with real good/bad delta arrow
        m1.metric(
            "Net Margin Change (Central)",
            f"${net_central:+,.2f}",
            delta=f"${net_central:+,.2f} vs no waiver",
            help="Central estimate evaluated at -ATT_Q5_spend. Labeled as: based on the estimated effect.",
        )
        # 95% CI Uncertainty Band without decorative arrow to avoid truncation
        m2.metric(
            "95% CI Uncertainty Band",
            f"${net_low:+,.0f} to ${net_high:+,.0f}",
            help=f"Low evaluated at -ci_upper (${sim_res['ci_upper_spend']:+.2f}); High evaluated at -ci_lower (${sim_res['ci_lower_spend']:+.2f}).",
        )
        # Active share effect in pp with real good/bad arrow
        m3.metric(
            "Q5 Active Share Effect",
            f"{active_share_pp:+.2f} pp",
            delta=f"{active_share_pp:+.2f} pp vs no policy",
            delta_color="normal" if active_share_pp >= 0 else "inverse",
            help="Formula: (1 - waived_share) * ATT_Q5_active90. Represents change in 90-day active customer share in percentage points.",
        )

        st.caption(
            "**Central estimate status**: based on the estimated effect | "
            f"Target Cohort: **{n_q5:,} treated top spenders (Quintile 5)** | "
            f"Estimated Q5 Spend ATT: **\\${sim_res['att_q5_spend']:+.2f}/wk** "
            f"(95% CI: `[\\${sim_res['ci_lower_spend']:+.2f}, \\${sim_res['ci_upper_spend']:+.2f}]`)"
        )

    st.markdown("---")

    # 5. Uncertainty Message when the band includes zero
    band_spans_zero = bool(min(net_low, net_high) <= 0.0 <= max(net_low, net_high))
    if band_spans_zero and waived_share > 0:
        st.warning(
            f"**Uncertainty Notice**: The 95% confidence interval for net margin change includes zero "
            f"(\\${min(net_low, net_high):+,.0f} to \\${max(net_low, net_high):+,.0f}). "
            "Because the confidence band encompasses zero, there is statistical uncertainty whether the fee waiver policy "
            "yields a net financial gain or loss under estimation error."
        )

    # 6. Guardrail Lights on the Active Share
    st.markdown("### Active Share Guardrail Monitor")
    st.caption("Real-time monitoring of customer relationship stability and active share among top spenders (Quintile 5).")

    g_col1, g_col2, g_col3 = st.columns([2, 1, 1])

    if active_share_pp >= -0.5:
        # Green Guardrail
        g_col1.success(
            "**GREEN GUARDRAIL — Active Share Preserved**\n\n"
            f"Projected active share change is **{active_share_pp:+.2f} pp** (>= -0.5 pp). "
            "High active share retained among top spenders. Risk of customer alienation is effectively controlled."
        )
        g_col2.metric("Q5 active share change vs no policy", f"{active_share_pp:+.2f} pp")
        g_col3.metric("Alienation Risk Level", "Minimal (Protected)")
    elif active_share_pp >= -2.0:
        # Amber Guardrail
        g_col1.warning(
            "**AMBER GUARDRAIL — Moderate Active Share Impact**\n\n"
            f"Projected active share change is **{active_share_pp:+.2f} pp** (between -0.5 pp and -2.0 pp). "
            "Moderate active share erosion observed. Monitor top accounts closely and consider expanding waiver eligibility."
        )
        g_col2.metric("Q5 active share change vs no policy", f"{active_share_pp:+.2f} pp")
        g_col3.metric("Alienation Risk Level", "Moderate Caution")
    else:
        # Red Guardrail
        g_col1.error(
            "**RED GUARDRAIL — High Active Share Degradation**\n\n"
            f"Projected active share change is **{active_share_pp:+.2f} pp** (< -2.0 pp). "
            "Severe active share erosion detected among top spenders! We recommend increasing the fee waiver share toward 100%."
        )
        g_col2.metric("Q5 active share change vs no policy", f"{active_share_pp:+.2f} pp")
        g_col3.metric("Alienation Risk Level", "Critical Degradation")

    st.markdown("---")

    # 7. Interactive Visualizations: Band Chart & Sensitivity Curve
    col_chart_l, col_chart_r = st.columns(2)

    with col_chart_l:
        st.subheader("Net Margin Change ($): Low / Central / High Band")

        band_df = pd.DataFrame({
            "Estimate Tier": [
                "Low Estimate (Lower CI Bound)",
                "Central Estimate (based on estimated effect)",
                "High Estimate (Upper CI Bound)",
            ],
            "Net Margin Change ($)": [net_low, net_central, net_high],
        })

        fig_band = px.bar(
            band_df,
            x="Estimate Tier",
            y="Net Margin Change ($)",
            text="Net Margin Change ($)",
            color="Estimate Tier",
            color_discrete_sequence=["#EF4444", "#4F46E5", "#10B981"],
            title=f"Net Financial Impact at {waived_share_pct}% Waiver ({horizon_weeks} Weeks)",
        )
        fig_band.update_traces(texttemplate="$%{text:+,.0f}", textposition="outside")
        fig_band.add_hline(y=0, line_dash="solid", line_color="#64748B")
        fig_band.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_band, use_container_width=True)

    with col_chart_r:
        st.subheader("Sensitivity Across Waiver Share (0% to 100%)")

        sim_curve_shares = np.linspace(0.0, 1.0, 21)
        sim_curve_results = [
            simulate_q5_policy(
                scenario=selected_scenario,
                waived_share=s,
                gross_margin_pct=gross_margin_pct,
                weeks=horizon_weeks,
            )
            for s in sim_curve_shares
        ]

        curve_df = pd.DataFrame({
            "Waiver Share (%)": [s * 100.0 for s in sim_curve_shares],
            "Net Margin Central ($)": [r["net_margin_change_central"] for r in sim_curve_results],
            "Net Margin Low ($)": [r["net_margin_change_low"] for r in sim_curve_results],
            "Net Margin High ($)": [r["net_margin_change_high"] for r in sim_curve_results],
            "Active Share Impact (pp)": [r["q5_retention_effect"] * 100.0 for r in sim_curve_results],
        })

        fig_curve = go.Figure()
        # High bound
        fig_curve.add_trace(go.Scatter(
            x=curve_df["Waiver Share (%)"],
            y=curve_df["Net Margin High ($)"],
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1, dash="dash"),
            name="High Bound (Upper CI)",
        ))
        # Low bound
        fig_curve.add_trace(go.Scatter(
            x=curve_df["Waiver Share (%)"],
            y=curve_df["Net Margin Low ($)"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(79, 70, 229, 0.12)",
            line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dash"),
            name="Low Bound (Lower CI)",
        ))
        # Central line
        fig_curve.add_trace(go.Scatter(
            x=curve_df["Waiver Share (%)"],
            y=curve_df["Net Margin Central ($)"],
            mode="lines+markers",
            line=dict(color="#4F46E5", width=3),
            name="Central Estimate (based on estimated effect)",
        ))
        fig_curve.add_hline(y=0, line_dash="solid", line_color="#64748B")
        fig_curve.update_layout(
            title=f"Net Margin Frontier by Waiver Share ({selected_scenario.capitalize()})",
            xaxis_title="Q5 Waived Share (%)",
            yaxis_title="Net Margin Change ($)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("---")

    # 8. Econometric Specification Documentation
    st.markdown(
        """
        > **Econometric Formula Specification**:
        > - **Net Margin Change** $= \\text{gross\\_margin} \\times \\text{waived\\_share} \\times N_{\\text{Q5}} \\times \\text{weeks} \\times (-\\text{ATT}_{\\text{Q5\\_spend}})$
        >   * Evaluated for **Central** ($-\\text{ATT}$), **Low** ($-\\text{CI}_{\\text{upper}}$), and **High** ($-\\text{CI}_{\\text{lower}}$).
        > - **Q5 Active Share Effect** $= (1 - \\text{waived\\_share}) \\times \\text{ATT}_{\\text{Q5\\_active90}}$ (measured in percentage points).
        > - *Note: No separate CLV term is added to prevent double-counting customer churn and spend degradation.*
        """
    )

    st.markdown("---")

    # 9. Page Footer naming exact source tables
    st.caption(
        "**Source Tables**: `causal_experiment_data` | "
        f"**Files**: `results/att_vs_truth_{selected_scenario}.csv`, `config.yaml`"
    )
