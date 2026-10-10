"""Streamlit Probabilistic CLV Studio Component.

Renders lifetime value distributions, P(Alive) heatmaps, time-based holdout model validation deciles,
and individual customer forecasts for 12M and 24M with empirical bootstrap confidence intervals.
"""

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clv_causal.config import CALIBRATION_END_DATE, DB_PATH, ESTIMATION_METHOD, HOLDOUT_END_DATE


def render_clv_tab() -> None:
    """Renders the Probabilistic Customer Lifetime Value (CLV) Studio tab in Streamlit."""
    st.markdown("## 🔮 Probabilistic Customer Lifetime Value (CLV) Studio")
    st.caption(f"BG/NBD & Gamma-Gamma probabilistic lifetime models fitted via **{ESTIMATION_METHOD}**.")

    conn = duckdb.connect(DB_PATH)
    clv_df: pd.DataFrame = conn.execute("SELECT * FROM pred_clv_summary").df()

    val_metrics: pd.DataFrame = pd.DataFrame()
    decile_df: pd.DataFrame = pd.DataFrame()
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    if "clv_holdout_metrics" in tables:
        val_metrics = conn.execute("SELECT * FROM clv_holdout_metrics").df()
    if "clv_holdout_deciles" in tables:
        decile_df = conn.execute("SELECT * FROM clv_holdout_deciles").df()
    conn.close()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mean 12-Month CLV", f"${clv_df['clv_12m'].mean():,.2f}")
    col_b.metric("Mean Expected Repeat Orders (12M)", f"{clv_df['expected_purchases_12m'].mean():.2f}")
    col_c.metric("Active Ratio P(Alive) > 80%", f"{(clv_df['p_alive'] > 0.8).mean()*100:.1f}%")
    if not val_metrics.empty:
        r_val = val_metrics["pearson_r"].iloc[0]
        mae_val = val_metrics["mae_purchases"].iloc[0] if "mae_purchases" in val_metrics.columns else val_metrics["mae"].iloc[0]
        col_d.metric("Holdout Validation MAE", f"{mae_val:.4f} orders", delta=f"r = {r_val:.4f}" if not np.isnan(r_val) else "Holdout Verified")
    else:
        col_d.metric("Holdout Validation", "Evaluated")

    st.markdown("---")

    tab_dist, tab_val, tab_lookup = st.tabs([
        "📊 Portfolio CLV Distributions",
        "🎯 Holdout Model Validation Diagnostics",
        "🔍 Customer Level Lookup & Uncertainty Bounds",
    ])

    with tab_dist:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("12-Month Projected CLV Distribution ($)")
            fig_hist = px.histogram(
                clv_df[clv_df["clv_12m"] < clv_df["clv_12m"].quantile(0.98)],
                x="clv_12m",
                nbins=50,
                title="Customer 1-Year Lifetime Value Frequency",
                labels={"clv_12m": "Predicted 12-Month CLV ($)"},
                color_discrete_sequence=["#10B981"],
            )
            fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Customer Recency vs Frequency P(Alive) Heatmap")
            heatmap_data = clv_df.groupby(["frequency", pd.cut(clv_df["recency"], bins=6)], observed=False)["p_alive"].mean().unstack()
            heatmap_data.columns = [f"{int(c.left)}-{int(c.right)} days" for c in heatmap_data.columns if hasattr(c, "left")]

            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="Recency (Days)", y="Frequency (Repeat Purchases)", color="P(Alive)"),
                title="Probability of Being Active Matrix",
                color_continuous_scale="RdYlGn",
            )
            fig_heat.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab_val:
        st.subheader("🎯 Time-Based Out-of-Sample Holdout Model Validation")
        st.caption(f"Fitting period: `{CALIBRATION_END_DATE}` (First 12 Months) | Evaluation period: `{CALIBRATION_END_DATE}` to `{HOLDOUT_END_DATE}` ({ESTIMATION_METHOD}).")

        if not val_metrics.empty:
            m_row = val_metrics.iloc[0]
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Estimation Method", ESTIMATION_METHOD)
            v2.metric("Purchase MAE (BG/NBD vs Naive)", f"{m_row.get('mae_purchases', m_row.get('mae', 0.0)):.4f}", delta=f"Naive: {m_row.get('naive_mae_purchases', 0.0):.4f}")
            v3.metric("Purchase RMSE", f"{m_row.get('rmse_purchases', m_row.get('rmse', 0.0)):.4f}")
            v4.metric("Aggregate Volume Error", f"{m_row.get('volume_error_pct', 0.0):+.2f}%")

            corr_v = m_row.get("freq_monetary_corr", 0.0)
            st.info(f"**Model Independence Check**: Frequency vs Repeat Monetary Pearson correlation is **r = {corr_v:.4f}** ($p = {m_row.get('freq_monetary_pval', 0.0):.4e}$), validating Gamma-Gamma conditional independence.")

        if not decile_df.empty:
            st.markdown("#### Customer Decile Calibration (Predicted vs Actual Holdout Purchases)")
            fig_decile = go.Figure()
            fig_decile.add_trace(go.Bar(
                x=decile_df["decile_rank"],
                y=decile_df["avg_actual_purchases"],
                name="Actual Holdout Purchases",
                marker_color="#3B82F6",
            ))
            fig_decile.add_trace(go.Bar(
                x=decile_df["decile_rank"],
                y=decile_df["avg_predicted_purchases"],
                name="BG/NBD Predicted Purchases",
                marker_color="#10B981",
            ))
            fig_decile.update_layout(
                barmode="group",
                title="Actual vs Predicted Purchases by Customer Decile (D01 = Lowest, D10 = Highest)",
                xaxis_title="Customer Decile Rank",
                yaxis_title="Average Holdout Purchases per Customer",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_decile, use_container_width=True)

            st.subheader("Decile Diagnostics Detail Table")
            st.dataframe(decile_df, use_container_width=True)

    with tab_lookup:
        st.subheader("🔍 Customer Level Forecast & 95% Bootstrap Confidence Intervals")
        cust_list = sorted(clv_df["customer_id"].unique().tolist())
        selected_cust = st.selectbox("Select Customer ID:", cust_list, index=0)

        c_row = clv_df[clv_df["customer_id"] == selected_cust].iloc[0]

        st.markdown(f"### Customer ID: `{int(c_row['customer_id'])}`")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Historical Repeat Orders", f"{int(c_row['frequency'])}")
        col_m2.metric("Historical Total Spend", f"${c_row['total_monetary']:,.2f}")
        col_m3.metric("Probability Active P(Alive)", f"{c_row['p_alive']*100:.1f}%")
        col_m4.metric("Expected Avg Order Value", f"${c_row['expected_avg_order_value']:,.2f}")

        st.markdown(f"#### Lifetime Forecast & Empirical Bootstrap 95% Confidence Intervals ({ESTIMATION_METHOD})")

        horizon_df = pd.DataFrame({
            "Horizon": ["12 Months", "24 Months"],
            "Expected CLV ($)": [c_row["clv_12m"], c_row["clv_24m"]],
            "Lower Bound (95% Bootstrap CI)": [c_row["clv_12m_lower"], c_row["clv_24m_lower"]],
            "Upper Bound (95% Bootstrap CI)": [c_row["clv_12m_upper"], c_row["clv_24m_upper"]],
            "Expected Purchases": [c_row["expected_purchases_12m"], c_row["expected_purchases_24m"]],
        })
        st.dataframe(horizon_df.style.format({
            "Expected CLV ($)": "${:,.2f}",
            "Lower Bound (95% Bootstrap CI)": "${:,.2f}",
            "Upper Bound (95% Bootstrap CI)": "${:,.2f}",
            "Expected Purchases": "{:,.2f}",
        }), use_container_width=True)
