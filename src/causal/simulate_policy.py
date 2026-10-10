"""Policy Intervention Simulator with Random Assignment and Configurable Multiplicative Effects.

Simulates a business policy intervention by randomly assigning customers to treatment
with a configurable share (default 40%) and applying multiplicative order rate, AOV,
and top-CLV churn hazard effects at policy date t_0.
"""

from typing import Dict, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd

from clv_causal.config import (
    AOV_CHANGE,
    DB_PATH,
    INTERVENTION_DATE,
    ORDER_RATE_CHANGE,
    RANDOM_SEED,
    TOP_CLV_CHURN_HAZARD,
    TREATMENT_SHARE,
)


def simulate_policy_intervention(
    db_path: str = DB_PATH,
    intervention_date: str = INTERVENTION_DATE,
    treatment_share: float = TREATMENT_SHARE,
    order_rate_change: float = ORDER_RATE_CHANGE,
    aov_change: float = AOV_CHANGE,
    top_clv_churn_hazard: float = TOP_CLV_CHURN_HAZARD,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulates policy rollout on randomly assigned customers with known multiplicative effects.

    Args:
        db_path: Absolute path to DuckDB warehouse.
        intervention_date: ISO date string for policy rollout date t_0 (e.g. '2010-06-01').
        treatment_share: Fraction of customers randomly assigned to treatment (e.g. 0.40).
        order_rate_change: Relative multiplicative change in order frequency (e.g. 0.15 = +15%).
        aov_change: Relative multiplicative change in average order value (e.g. 0.10 = +10%).
        top_clv_churn_hazard: Increased probability of churn post-t0 for top CLV quintile customers.
        random_seed: Integer seed for reproducible random assignment and churn draws.

    Returns:
        DataFrame customer-by-week panel with true counterfactual outcomes and true effects.
    """
    print(f"\n=================== Causal Policy Intervention Simulation ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Configured Treatment Share:             {treatment_share * 100:.1f}%")
    print(f"Configured Order Rate Change:           {order_rate_change * 100:+.1f}%")
    print(f"Configured AOV Change:                  {aov_change * 100:+.1f}%")
    print(f"Configured Top CLV Churn Hazard:        {top_clv_churn_hazard * 100:.1f}%")
    print(f"Random Seed:                            {random_seed}")

    conn = duckdb.connect(db_path)

    # 1. Fetch CLV scores to identify top CLV quintile customers
    try:
        clv_df = conn.execute("SELECT customer_id, clv_12m FROM clv_scores").df()
    except Exception:
        clv_df = conn.execute("""
            SELECT customer_id, SUM(line_total) AS clv_12m
            FROM fact_orders
            GROUP BY customer_id
        """).df()

    if len(clv_df) > 0 and "clv_12m" in clv_df.columns:
        clv_cutoff = float(clv_df["clv_12m"].quantile(0.80))
        top_clv_customers = set(clv_df[clv_df["clv_12m"] >= clv_cutoff]["customer_id"])
    else:
        top_clv_customers = set()

    # 2. Extract weekly transaction panel per customer
    tx_panel: pd.DataFrame = conn.execute("""
        SELECT
            fo.customer_id,
            dc.country,
            DATE_TRUNC('week', fo.invoice_date) AS week_date,
            COUNT(DISTINCT fo.invoice_no) AS counterfactual_weekly_orders,
            SUM(fo.line_total) AS counterfactual_weekly_spend
        FROM fact_orders fo
        JOIN dim_customers dc ON fo.customer_id = dc.customer_id
        GROUP BY fo.customer_id, dc.country, DATE_TRUNC('week', fo.invoice_date)
    """).df()

    tx_panel["week_date"] = pd.to_datetime(tx_panel["week_date"])
    t0_date = pd.to_datetime(intervention_date)
    tx_panel["is_post_period"] = np.where(tx_panel["week_date"] >= t0_date, 1, 0)

    # 3. Seeded Random Treatment Assignment (40% treatment share)
    unique_customers = np.sort(tx_panel["customer_id"].unique())
    n_customers = len(unique_customers)

    rng = np.random.default_rng(random_seed)
    rand_draws = rng.random(size=n_customers)
    treated_customer_ids = set(unique_customers[rand_draws < treatment_share])

    # 4. Seeded Churn Draws for Top CLV Quintile Treated Customers
    top_clv_treated_ids = sorted(list(treated_customer_ids.intersection(top_clv_customers)))
    churn_draws = rng.random(size=len(top_clv_treated_ids))
    churned_top_clv_ids = set([cid for cid, draw in zip(top_clv_treated_ids, churn_draws) if draw < top_clv_churn_hazard])

    # 5. Populate Customer Level Flags
    tx_panel["is_treated"] = tx_panel["customer_id"].isin(treated_customer_ids).astype(int)
    tx_panel["is_treated_cohort"] = tx_panel["is_treated"]  # Backward compatibility flag
    tx_panel["is_top_clv_quintile"] = tx_panel["customer_id"].isin(top_clv_customers).astype(int)
    tx_panel["is_churned_top_clv"] = tx_panel["customer_id"].isin(churned_top_clv_ids).astype(int)

    # 6. Apply Multiplicative Policy Effects
    is_post_treated = (tx_panel["is_treated"] == 1) & (tx_panel["is_post_period"] == 1)
    is_churned_post = is_post_treated & (tx_panel["is_churned_top_clv"] == 1)

    # Base multipliers
    order_multiplier = 1.0 + order_rate_change
    aov_multiplier = 1.0 + aov_change

    # Default observed equals counterfactual
    tx_panel["observed_weekly_orders"] = tx_panel["counterfactual_weekly_orders"].astype(float)
    tx_panel["observed_weekly_spend"] = tx_panel["counterfactual_weekly_spend"].astype(float)

    # Apply policy lift for non-churned treated post-t0 customer-weeks
    active_treated_post = is_post_treated & (~is_churned_post)
    tx_panel.loc[active_treated_post, "observed_weekly_orders"] = (
        tx_panel.loc[active_treated_post, "counterfactual_weekly_orders"] * order_multiplier
    )
    tx_panel.loc[active_treated_post, "observed_weekly_spend"] = (
        tx_panel.loc[active_treated_post, "counterfactual_weekly_spend"] * order_multiplier * aov_multiplier
    )

    # Apply zeroing for churned top CLV treated post-t0 customer-weeks
    tx_panel.loc[is_churned_post, "observed_weekly_orders"] = 0.0
    tx_panel.loc[is_churned_post, "observed_weekly_spend"] = 0.0

    # 7. Store True Counterfactual Effects
    tx_panel["true_effect_orders"] = tx_panel["observed_weekly_orders"] - tx_panel["counterfactual_weekly_orders"]
    tx_panel["true_effect_spend"] = tx_panel["observed_weekly_spend"] - tx_panel["counterfactual_weekly_spend"]

    # Round outputs cleanly
    tx_panel["observed_weekly_orders"] = np.round(tx_panel["observed_weekly_orders"], 2)
    tx_panel["observed_weekly_spend"] = np.round(tx_panel["observed_weekly_spend"], 2)
    tx_panel["true_effect_orders"] = np.round(tx_panel["true_effect_orders"], 2)
    tx_panel["true_effect_spend"] = np.round(tx_panel["true_effect_spend"], 2)

    # 8. Compute and Print Ground-Truth ATT
    post_treated_sub = tx_panel[is_post_treated]
    gt_att_spend = float(post_treated_sub["true_effect_spend"].mean()) if len(post_treated_sub) > 0 else 0.0
    gt_att_orders = float(post_treated_sub["true_effect_orders"].mean()) if len(post_treated_sub) > 0 else 0.0

    print(f"\n--- Ground-Truth Average Treatment Effect on Treated (ATT post-t_0) ---")
    print(f"Ground-Truth ATT (Weekly Spend / Treated Customer-Week):  ${gt_att_spend:+.4f}")
    print(f"Ground-Truth ATT (Weekly Orders / Treated Customer-Week): {gt_att_orders:+.4f}")
    print(f"Total Post-t0 Treated Customer-Weeks:                     {len(post_treated_sub):,}")
    print(f"Actual Realized Treatment Share:                          {len(treated_customer_ids) / max(1, n_customers) * 100:.2f}%")
    print("===============================================================================\n")

    # 9. Compute Baseline Diagnostics & Save to DuckDB
    pre_panel = tx_panel[tx_panel["is_post_period"] == 0].copy()
    tr_pre = pre_panel[pre_panel["is_treated"] == 1]
    co_pre = pre_panel[pre_panel["is_treated"] == 0]

    tr_cust_cnt = tr_pre["customer_id"].nunique()
    co_cust_cnt = co_pre["customer_id"].nunique()

    tr_avg_rev = float(tr_pre.groupby("customer_id")["observed_weekly_spend"].mean().mean()) if tr_cust_cnt > 0 else 0.0
    co_avg_rev = float(co_pre.groupby("customer_id")["observed_weekly_spend"].mean().mean()) if co_cust_cnt > 0 else 0.0

    tr_std_rev = float(tr_pre.groupby("customer_id")["observed_weekly_spend"].mean().std()) if tr_cust_cnt > 0 else 0.0
    co_std_rev = float(co_pre.groupby("customer_id")["observed_weekly_spend"].mean().std()) if co_cust_cnt > 0 else 0.0

    smd_rev = (
        (tr_avg_rev - co_avg_rev) / np.sqrt((tr_std_rev**2 + co_std_rev**2) / 2.0)
        if (tr_std_rev**2 + co_std_rev**2) > 0
        else 0.0
    )

    diag_df = pd.DataFrame([
        {
            "Metric": "Active Customer Count",
            "Treated_Cohort": f"{tr_cust_cnt:,}",
            "Control_Cohort": f"{co_cust_cnt:,}",
            "Standardized_Difference_SMD": "N/A",
        },
        {
            "Metric": "Pre-Period Weekly Spend / Customer ($)",
            "Treated_Cohort": f"${tr_avg_rev:.2f}",
            "Control_Cohort": f"${co_avg_rev:.2f}",
            "Standardized_Difference_SMD": f"{smd_rev:.4f}",
        },
    ])

    conn.execute("CREATE OR REPLACE TABLE causal_experiment_data AS SELECT * FROM tx_panel")
    conn.execute("CREATE OR REPLACE TABLE causal_treatment_control_diagnostics AS SELECT * FROM diag_df")
    conn.close()

    return tx_panel


if __name__ == "__main__":
    simulate_policy_intervention()
