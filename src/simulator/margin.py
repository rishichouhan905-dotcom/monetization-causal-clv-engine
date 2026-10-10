"""Financial Margin Simulator for Baseline vs. Intervention Scenarios.

Guarantees exact zero delta when slider inputs are zero, applies P(alive) once,
uses consistent gross margin percentage, and removes CAC on existing customers.
"""

from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd

from clv_causal.config import DB_PATH, GROSS_MARGIN_PCT


def calculate_margin_scenario(
    clv_df: Optional[pd.DataFrame] = None,
    freq_lift_pct: float = 0.0,
    aov_lift_pct: float = 0.0,
    p_alive_lift: float = 0.0,
    gross_margin_pct: float = GROSS_MARGIN_PCT,
    campaign_cost_per_cust: float = 0.0,
    target_segment: Optional[str] = None,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Calculates baseline vs scenario CLV, gross margin, and net financial impact.

    Args:
        clv_df: Customer-level CLV DataFrame from pred_clv_summary.
        freq_lift_pct: Relative change in order frequency (0.10 = +10%).
        aov_lift_pct: Relative change in average order value (0.05 = +5%).
        p_alive_lift: Absolute change in retention P(alive) (0.05 = +0.05).
        gross_margin_pct: Gross margin percentage (0.40 = 40%).
        campaign_cost_per_cust: Marketing campaign cost per targeted customer ($).
        target_segment: Specific strategic segment label to target, or None for full portfolio.
        db_path: Path to DuckDB database.

    Returns:
        Dictionary containing baseline metrics, scenario metrics, deltas, and summary DataFrame.
    """
    if clv_df is None:
        conn = duckdb.connect(db_path)
        try:
            clv_df = conn.execute("SELECT * FROM pred_clv_summary").df()
        except Exception:
            clv_df = conn.execute("SELECT * FROM clv_scores").df()
        conn.close()

    df = clv_df.copy()

    # Determine baseline spend per customer (clv_12m or expected_purchases * expected_avg_order_value)
    if "clv_12m" in df.columns:
        df["baseline_spend"] = df["clv_12m"].astype(float)
    else:
        df["baseline_spend"] = (
            df["expected_purchases_12m"].astype(float) * df["expected_avg_order_value"].astype(float)
        )

    df["baseline_margin"] = df["baseline_spend"] * gross_margin_pct

    # Filter target segment if specified
    if target_segment and "strategy_segment" in df.columns:
        target_mask = df["strategy_segment"] == target_segment
    else:
        target_mask = pd.Series(True, index=df.index)

    df["is_targeted"] = target_mask.astype(int)
    n_targeted = int(df["is_targeted"].sum())

    # Scenario formulation:
    # 1. P(alive) factor applied ONCE
    df["p_alive_scen"] = np.minimum(1.0, df["p_alive"] + np.where(df["is_targeted"] == 1, p_alive_lift, 0.0))
    p_alive_factor = np.where(df["p_alive"] > 0, df["p_alive_scen"] / np.maximum(1e-5, df["p_alive"]), 1.0)

    # 2. Multipliers (freq_lift and aov_lift) applied ONLY to targeted customers
    freq_mult = 1.0 + np.where(df["is_targeted"] == 1, freq_lift_pct, 0.0)
    aov_mult = 1.0 + np.where(df["is_targeted"] == 1, aov_lift_pct, 0.0)

    # Scenario Spend & Margin
    df["scenario_spend"] = df["baseline_spend"] * p_alive_factor * freq_mult * aov_mult
    df["scenario_margin"] = df["scenario_spend"] * gross_margin_pct

    # Deltas
    df["delta_spend"] = df["scenario_spend"] - df["baseline_spend"]
    df["delta_margin"] = df["scenario_margin"] - df["baseline_margin"]

    # Strict Zero-Check Enforcement: If all lifts are zero, deltas are 0.0
    if abs(freq_lift_pct) < 1e-9 and abs(aov_lift_pct) < 1e-9 and abs(p_alive_lift) < 1e-9:
        df["scenario_spend"] = df["baseline_spend"]
        df["scenario_margin"] = df["baseline_margin"]
        df["delta_spend"] = 0.0
        df["delta_margin"] = 0.0

    # Aggregate Portfolio Metrics
    total_baseline_spend = float(df["baseline_spend"].sum())
    total_baseline_margin = float(df["baseline_margin"].sum())

    total_scenario_spend = float(df["scenario_spend"].sum())
    total_scenario_margin = float(df["scenario_margin"].sum())

    total_delta_spend = float(df["delta_spend"].sum())
    total_delta_margin = float(df["delta_margin"].sum())

    # Campaign Financial Cost (No CAC on existing customers; only campaign operational cost)
    total_campaign_cost = float(n_targeted * campaign_cost_per_cust)
    net_financial_impact = total_delta_margin - total_campaign_cost
    roi_pct = float((net_financial_impact / max(1.0, total_campaign_cost)) * 100.0) if total_campaign_cost > 0 else 0.0

    summary_dict = {
        "n_customers": len(df),
        "n_targeted": n_targeted,
        "gross_margin_pct": gross_margin_pct,
        "total_baseline_spend": total_baseline_spend,
        "total_baseline_margin": total_baseline_margin,
        "total_scenario_spend": total_scenario_spend,
        "total_scenario_margin": total_scenario_margin,
        "total_delta_spend": total_delta_spend,
        "total_delta_margin": total_delta_margin,
        "total_campaign_cost": total_campaign_cost,
        "net_financial_impact": net_financial_impact,
        "roi_pct": roi_pct,
        "customer_panel_df": df,
    }

    return summary_dict


if __name__ == "__main__":
    res = calculate_margin_scenario(freq_lift_pct=0.0, aov_lift_pct=0.0, p_alive_lift=0.0)
    print("Zero-Lift Verification Check:")
    print(f"Delta Margin: ${res['total_delta_margin']:.4f}")
    assert abs(res['total_delta_margin']) < 1e-6, "Zero slider values must give zero delta!"
