"""Streamlit Policy Simulator Component.

Simulates the financial and customer retention impact of the Q5 fee-waiver intervention policy,
reusing simulate_q5_policy from decision_engine.py with low/central/high uncertainty bands,
active share guardrail monitors (both point estimate and CI worst case with config thresholds),
and horizon validation notices.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from causal.decision_engine import simulate_q5_policy
from clv_causal.app.formatters import fmt_ci, fmt_pp, fmt_usd
from clv_causal.config import (
    GROSS_MARGIN_PCT,
    GUARDRAIL_AMBER_THRESHOLD_PP,
    GUARDRAIL_RED_THRESHOLD_PP,
)


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
            help="Simulation horizon in weeks. Horizons beyond 27 weeks are an extrapolation of the estimated effect.",
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

    # 3. Horizon Box: renamed 'Horizon within observed 27-week window'
    if horizon_weeks <= 27:
        st.success(
            f"**Horizon within observed 27-week window** (evaluation horizon: {horizon_weeks} weeks)"
        )
    else:
        st.warning(
            f"**Extrapolation Notice**: Horizon is set to {horizon_weeks} weeks. "
            "Projections beyond 27 weeks are an **extrapolation of the estimated treatment effect**."
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
    n_q5 = sim_res["n_q5"]

    # Active share metrics (without waiver, with waiver, and improvement)
    att_act = sim_res["att_q5_active90"]
    ci_lower_act = sim_res.get("ci_lower_active90", att_act - 0.0366)
    active_without_waiver_pp = att_act * 100.0
    active_with_waiver_pp = (1.0 - waived_share) * att_act * 100.0
    active_improvement_pp = active_with_waiver_pp - active_without_waiver_pp

    # Active share display string requested: e.g. '−3.72 pp without waiver, −1.86 pp with waiver (improvement +1.86 pp)'
    active_share_delta_str = (
        f"{fmt_pp(active_without_waiver_pp)} without waiver, "
        f"{fmt_pp(active_with_waiver_pp)} with waiver "
        f"(improvement {fmt_pp(active_improvement_pp, plus_sign=True)})"
    )

    with col_c2:
        st.markdown("### Simulated Policy Outcomes")

        m1, m2, m3 = st.columns(3)
        # Central estimate with real good/bad delta arrow
        m1.metric(
            "Net Margin Change (Central)",
            fmt_usd(net_central, plus_sign=True),
            delta=f"{fmt_usd(net_central, plus_sign=True)} vs no waiver",
            help="Central estimate evaluated at -ATT_Q5_spend. Based on the estimated treatment effect.",
        )
        # 95% CI Uncertainty Band without decorative arrow to prevent truncation
        m2.metric(
            "95% CI Uncertainty Band",
            f"{fmt_usd(net_low, decimals=0)} to {fmt_usd(net_high, decimals=0)}",
            help=f"Low evaluated at -ci_upper ({fmt_usd(sim_res['ci_upper_spend'], plus_sign=True)}); High evaluated at -ci_lower ({fmt_usd(sim_res['ci_lower_spend'], plus_sign=True)}).",
        )
        # Q5 Active share effect with neutral colors
        m3.metric(
            "Q5 Active Share Effect",
            fmt_pp(active_with_waiver_pp),
            delta=active_share_delta_str,
            delta_color="off",  # Neutral colours as requested
            help="Change in 90-day active customer share with fee waiver vs without waiver.",
        )

        spend_ci_str = fmt_ci(sim_res['ci_lower_spend'], sim_res['ci_upper_spend'])
        st.caption(
            "**Central estimate status**: based on the estimated effect | "
            f"Target Cohort: **{n_q5:,} treated top spenders (Quintile 5)** | "
            f"Estimated Q5 Spend ATT: **{fmt_usd(sim_res['att_q5_spend'], plus_sign=True)} / week** "
            f"(95% CI: {spend_ci_str})"
        )

    st.markdown("---")

    # 5. Uncertainty Message when the band includes zero
    band_spans_zero = bool(min(net_low, net_high) <= 0.0 <= max(net_low, net_high))
    if band_spans_zero and waived_share > 0:
        low_usd = fmt_usd(min(net_low, net_high), decimals=0)
        high_usd = fmt_usd(max(net_low, net_high), decimals=0)
        st.warning(
            f"**Uncertainty Notice**: The 95% confidence interval for net margin change includes zero "
            f"({low_usd} to {high_usd}). "
            "Because the confidence band encompasses zero, there is statistical uncertainty whether the fee waiver policy "
            "yields a net financial gain or loss under estimation error."
        )

    # 6. Guardrail Status computed from BOTH Point Estimate and CI Worst Case with thresholds from config
    st.markdown("### Active Share Guardrail Monitor")
    st.caption(
        f"Real-time monitoring of customer relationship stability and active share among top spenders (Quintile 5). "
        f"**Thresholds from config**: Green >= {GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp, "
        f"Amber [{GUARDRAIL_RED_THRESHOLD_PP:+.1f} pp, {GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp), "
        f"Red < {GUARDRAIL_RED_THRESHOLD_PP:+.1f} pp."
    )

    # Point estimate and CI worst case calculations
    point_act_pp = active_with_waiver_pp
    worst_act_pp = (1.0 - waived_share) * ci_lower_act * 100.0

    def _eval_guardrail(val_pp: float) -> tuple:
        if val_pp >= GUARDRAIL_AMBER_THRESHOLD_PP:
            return ("GREEN", "Normal / Protected", "Preserved (Minimal Risk)")
        elif val_pp >= GUARDRAIL_RED_THRESHOLD_PP:
            return ("AMBER", "Moderate Caution", "Moderate Erosion (Monitor Closely)")
        else:
            return ("RED", "Critical Degradation", "Severe Alienation (Action Required)")

    status_pt_label, status_pt_risk, status_pt_desc = _eval_guardrail(point_act_pp)
    status_w_label, status_w_risk, status_w_desc = _eval_guardrail(worst_act_pp)

    g_col1, g_col2, g_col3 = st.columns([2, 1, 1])

    with g_col1:
        if status_pt_label == "GREEN" and status_w_label == "GREEN":
            st.success(
                f"**DUAL GREEN GUARDRAIL — Active Share Fully Protected**\n\n"
                f"Both the point estimate ({fmt_pp(point_act_pp)}) and the 95% CI worst-case bound ({fmt_pp(worst_act_pp)}) "
                f"meet the safety threshold (>= {GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp from config)."
            )
        elif status_pt_label == "GREEN" and status_w_label != "GREEN":
            st.warning(
                f"**POINT GREEN / CI WORST-CASE {status_w_label}**\n\n"
                f"The point estimate is protected at **{fmt_pp(point_act_pp)}** (>= {GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp), "
                f"but the 95% CI worst-case scenario reaches **{fmt_pp(worst_act_pp)}** ({status_w_desc}). "
                f"Config safety threshold ({GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp) warrants active monitoring."
            )
        elif status_pt_label == "AMBER":
            st.warning(
                f"**AMBER GUARDRAIL — Moderate Active Share Impact**\n\n"
                f"Point estimate: **{fmt_pp(point_act_pp)}** | 95% CI worst-case: **{fmt_pp(worst_act_pp)}**. "
                f"Falls within the Amber zone ({GUARDRAIL_RED_THRESHOLD_PP:+.1f} pp to {GUARDRAIL_AMBER_THRESHOLD_PP:+.1f} pp from config). "
                "Consider increasing waiver share to insulate top accounts."
            )
        else:
            st.error(
                f"**RED GUARDRAIL — Severe Active Share Degradation**\n\n"
                f"Point estimate: **{fmt_pp(point_act_pp)}** (< {GUARDRAIL_RED_THRESHOLD_PP:+.1f} pp threshold from config). "
                f"Worst-case 95% CI reaches **{fmt_pp(worst_act_pp)}**. Immediate fee waiver expansion recommended."
            )

    with g_col2:
        st.metric(
            "Point Estimate Status",
            status_pt_label,
            delta=f"{fmt_pp(point_act_pp)} ({status_pt_risk})",
            delta_color="off",
            help=f"Evaluated from point ATT: {fmt_pp(point_act_pp)}. Threshold from config.",
        )

    with g_col3:
        st.metric(
            "CI Worst-Case Status",
            status_w_label,
            delta=f"{fmt_pp(worst_act_pp)} ({status_w_risk})",
            delta_color="off",
            help=f"Evaluated from 95% CI lower bound: {fmt_pp(worst_act_pp)}. Threshold from config.",
        )

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
        > - **Net Margin Change** = gross_margin × waived_share × N_Q5 × weeks × (−ATT_Q5_spend)
        >   * Evaluated for **Central** (−ATT), **Low** (−CI_upper), and **High** (−CI_lower).
        > - **Q5 Active Share Effect** = (1 − waived_share) × ATT_Q5_active90 (measured in percentage points).
        > - *Note: No separate CLV term is added to prevent double-counting customer churn and spend degradation.*
        """
    )

    st.markdown("---")

    # 9. Page Footer naming exact source tables
    st.caption(
        "**Source Tables**: `causal_experiment_data` | "
        f"**Files**: `results/att_vs_truth_{selected_scenario}.csv`, `config.yaml`"
    )
