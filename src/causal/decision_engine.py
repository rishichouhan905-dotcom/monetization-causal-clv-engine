"""Business Decision Layer & Strategy Studio Engine.

Transforms probabilistic CLV distributions and causal effect estimates (ATT)
into action-oriented customer segmentation, financial margin models, and campaign ROI scenarios.
"""

from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd

from clv_causal.config import (
    CAC_PER_CUSTOMER,
    CAMPAIGN_COST_PER_TARGET,
    DB_PATH,
    GROSS_MARGIN_PCT,
)


class BusinessDecisionEngine:
    """Business decision layer evaluating 4-quadrant strategic segmentation and campaign ROI scenarios."""

    def __init__(
        self,
        default_cac: float = CAC_PER_CUSTOMER,
        default_margin: float = GROSS_MARGIN_PCT,
        default_campaign_cost: float = CAMPAIGN_COST_PER_TARGET,
    ) -> None:
        """Initializes BusinessDecisionEngine with default financial parameters.

        Args:
            default_cac: Default Customer Acquisition Cost ($/customer).
            default_margin: Default gross margin percentage (0.0 to 1.0).
            default_campaign_cost: Default win-back campaign cost per targeted customer ($).
        """
        self.default_cac: float = default_cac
        self.default_margin: float = default_margin
        self.default_campaign_cost: float = default_campaign_cost

    def evaluate_decision_framework(
        self,
        cac_per_customer: float = CAC_PER_CUSTOMER,
        gross_margin_pct: float = GROSS_MARGIN_PCT,
        campaign_cost_per_target: float = CAMPAIGN_COST_PER_TARGET,
    ) -> Dict[str, Any]:
        """Evaluates 4-quadrant customer segmentation matrix and financial scenario ROI.

        Args:
            cac_per_customer: Customer Acquisition Cost ($/customer).
            gross_margin_pct: Gross margin percentage (0.0 to 1.0).
            campaign_cost_per_target: Win-back marketing campaign cost per targeted customer ($).

        Returns:
            Dictionary containing segment_summary DataFrame and decision_scenarios DataFrame.
        """
        conn = duckdb.connect(DB_PATH)
        clv_df = conn.execute("SELECT * FROM pred_clv_summary").df()

        clv_70th = clv_df["clv_12m"].quantile(0.70)
        p_alive_cutoff = 0.75

        conditions = [
            (clv_df["clv_12m"] >= clv_70th) & (clv_df["p_alive"] >= p_alive_cutoff),
            (clv_df["clv_12m"] >= clv_70th) & (clv_df["p_alive"] < p_alive_cutoff),
            (clv_df["clv_12m"] < clv_70th) & (clv_df["p_alive"] >= p_alive_cutoff),
            (clv_df["clv_12m"] < clv_70th) & (clv_df["p_alive"] < p_alive_cutoff),
        ]

        choices = [
            "1. VIP Retention & Loyalty (High CLV / High P(Alive))",
            "2. Win-Back Priority (High CLV / Low P(Alive))",
            "3. Cross-Sell / Upsell (Lower CLV / High P(Alive))",
            "4. Low-Cost Nurturing (Lower CLV / Low P(Alive))",
        ]

        clv_df["strategy_segment"] = np.select(
            conditions, choices, default="4. Low-Cost Nurturing (Lower CLV / Low P(Alive))"
        )

        segment_summary = (
            clv_df.groupby("strategy_segment")
            .agg(
                customer_count=("customer_id", "count"),
                avg_12m_clv=("clv_12m", "mean"),
                total_12m_clv=("clv_12m", "sum"),
                avg_p_alive=("p_alive", "mean"),
                avg_expected_orders=("expected_purchases_12m", "mean"),
            )
            .reset_index()
        )

        segment_summary["pct_of_customers"] = np.round((segment_summary["customer_count"] / len(clv_df)) * 100.0, 1)
        segment_summary["pct_of_portfolio_val"] = np.round((segment_summary["total_12m_clv"] / clv_df["clv_12m"].sum()) * 100.0, 1)

        att_val = 18.50
        did_tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        if "causal_experiment_data" in did_tables:
            att_res = conn.execute(
                "SELECT AVG(observed_weekly_spend - counterfactual_weekly_spend) FROM causal_experiment_data WHERE is_post_period=1 AND is_treated=1"
            ).fetchone()
            if att_res and att_res[0] is not None:
                att_val = float(att_res[0])

        winback_seg = (
            segment_summary[segment_summary["strategy_segment"].str.contains("Win-Back")].iloc[0]
            if len(segment_summary[segment_summary["strategy_segment"].str.contains("Win-Back")]) > 0
            else segment_summary.iloc[0]
        )

        target_custs = winback_seg["customer_count"]
        campaign_cost = target_custs * campaign_cost_per_target

        incremental_gross_revenue = target_custs * 0.15 * (winback_seg["avg_12m_clv"] + att_val * 4.0)
        incremental_gross_margin = incremental_gross_revenue * gross_margin_pct
        net_financial_impact = incremental_gross_margin - campaign_cost
        roi_pct = (net_financial_impact / max(1.0, campaign_cost)) * 100.0

        decision_scenarios = pd.DataFrame(
            [
                {
                    "Campaign_Strategy": "High-Value Customer Win-Back",
                    "Target_Cohort": "High CLV + Low P(Alive)",
                    "Targeted_Customers": int(target_custs),
                    "Campaign_Cost": float(campaign_cost),
                    "Est_Incremental_Gross_Revenue": float(incremental_gross_revenue),
                    "Gross_Margin_Dollar": float(incremental_gross_margin),
                    "Net_Financial_Impact": float(net_financial_impact),
                    "ROI_Percent": float(roi_pct),
                }
            ]
        )

        conn.execute("CREATE OR REPLACE TABLE decision_segment_summary AS SELECT * FROM segment_summary")
        conn.execute("CREATE OR REPLACE TABLE decision_scenarios AS SELECT * FROM decision_scenarios")
        conn.close()

        return {
            "segment_summary": segment_summary,
            "decision_scenarios": decision_scenarios,
        }


def run_decision_engine() -> Dict[str, Any]:
    """Orchestrates business decision evaluation pipeline."""
    engine = BusinessDecisionEngine()
    return engine.evaluate_decision_framework()


def simulate_q5_policy(
    scenario: str = "hidden_degradation",
    waived_share: float = 0.50,
    gross_margin_pct: float = GROSS_MARGIN_PCT,
    weeks: int = 27,
    n_q5: Optional[int] = None,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Evaluates Q5 fee waiver policy intervention financial and retention effects.

    Formula:
        net margin change = gross_margin * waived_share * N_Q5 * weeks * (-ATT_Q5_spend)
        evaluated for central (point ATT), low (-ci_upper), and high (-ci_lower) bounds.
        Q5 retention effect = (1 - waived_share) * ATT_Q5_active90.

    Note:
        Does NOT add a separate CLV term (it double counts churn).
        Horizon is capped at 52 weeks; weeks > 27 are flagged as an extrapolation of the estimated effect.
    """
    from clv_causal.config import PROJECT_ROOT

    # Cap horizon at 52 weeks
    weeks = min(52, max(1, int(weeks)))
    is_extrapolation = bool(weeks > 27)

    # Determine N_Q5
    if n_q5 is None:
        try:
            conn = duckdb.connect(db_path, read_only=True)
            n_q5 = conn.execute(
                "SELECT COUNT(DISTINCT customer_id) FROM causal_experiment_data WHERE spend_quintile = 5 AND is_treated = 1"
            ).fetchone()[0]
            conn.close()
        except Exception:
            n_q5 = 395

    # Load scenario benchmark results
    scen_csv = PROJECT_ROOT / "results" / f"att_vs_truth_{scenario}.csv"
    if not scen_csv.exists():
        scen_csv = PROJECT_ROOT / "results" / "att_vs_truth.csv"

    att_spend = 0.0
    ci_lower_spend = 0.0
    ci_upper_spend = 0.0
    att_active90 = 0.0

    if scen_csv.exists():
        df = pd.read_csv(scen_csv)
        spend_row = df[(df["Cohort"] == "Quintile 5 (Top Spend)") & (df["Outcome_Metric"] == "weekly_spend_winsor") & (df["Estimator_Model"] == "TWFE DiD (PanelOLS)")]
        if len(spend_row) > 0:
            att_spend = float(spend_row.iloc[0]["Estimated_ATT"])
            ci_lower_spend = float(spend_row.iloc[0]["ci_lower"])
            ci_upper_spend = float(spend_row.iloc[0]["ci_upper"])

        active_row = df[(df["Cohort"] == "Quintile 5 (Top Spend)") & (df["Outcome_Metric"] == "active_last_90d") & (df["Estimator_Model"] == "TWFE DiD (PanelOLS)")]
        if len(active_row) > 0:
            att_active90 = float(active_row.iloc[0]["Estimated_ATT"])

    # Net margin change calculations:
    # Central = gross_margin * waived_share * N_Q5 * weeks * (-att_spend)
    # Low = gross_margin * waived_share * N_Q5 * weeks * (-ci_upper_spend)
    # High = gross_margin * waived_share * N_Q5 * weeks * (-ci_lower_spend)
    factor = gross_margin_pct * waived_share * n_q5 * weeks
    net_margin_central = factor * (-att_spend)
    net_margin_low = factor * (-ci_upper_spend)
    net_margin_high = factor * (-ci_lower_spend)

    # Q5 retention effect = (1 - waived_share) * ATT_Q5_active90
    q5_retention_effect = (1.0 - waived_share) * att_active90

    return {
        "scenario": scenario,
        "waived_share": waived_share,
        "gross_margin_pct": gross_margin_pct,
        "weeks": weeks,
        "n_q5": n_q5,
        "is_extrapolation": is_extrapolation,
        "extrapolation_label": "Extrapolation of estimated effect (> 27 weeks)" if is_extrapolation else "Empirically validated window (<= 27 weeks)",
        "att_q5_spend": att_spend,
        "ci_lower_spend": ci_lower_spend,
        "ci_upper_spend": ci_upper_spend,
        "att_q5_active90": att_active90,
        "net_margin_change_central": net_margin_central,
        "net_margin_change_low": net_margin_low,
        "net_margin_change_high": net_margin_high,
        "q5_retention_effect": q5_retention_effect,
    }


if __name__ == "__main__":
    res = run_decision_engine()
    print("--- 4-Quadrant Strategic Customer Segmentation ---")
    print(res["segment_summary"].to_string(index=False))
    print("\n--- Q5 Policy Simulation (Hidden Degradation) ---")
    q5_sim = simulate_q5_policy(scenario="hidden_degradation", waived_share=0.50, weeks=27)
    print(q5_sim)
