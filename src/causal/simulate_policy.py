"""Balanced Panel Policy Intervention Simulator with Stratified Treatment Assignment.

Builds a balanced customer-by-week panel including zero-purchase weeks from each eligible
customer's first-order week to the end of data (2011-12-09).
Eligible customers are strictly those whose first order occurred before policy rollout t_0.

Treatment assignment:
- Stratified random assignment of the configured share (default 40%) within each
  observed pre-period spend quintile, seeded reproducibly.

Intervention logic post-t_0 for treated customers:
1. In a zero-purchase week: adds a new order with probability (order_rate_change * pre_weekly_order_rate),
   valued at customer's pre-period median order value, scaled by (1 + aov_change).
2. In a positive-purchase week: scales order values by (1 + aov_change).
3. For treated customers in the top quintile of OBSERVED pre-period spend:
   draws a churn week from the configured hazard (top_clv_churn_hazard) and removes all orders after it.

Outcomes generated (counterfactual, observed, and true effect):
1. `weekly_orders`
2. `weekly_spend` (and `weekly_spend_winsor` winsorized at 99th percentile)
3. `active_last_90d` (indicator of any purchase in past 13 weeks / 90 days)
"""

from typing import Dict, List, Optional, Tuple
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


def extract_customer_pre_period_metrics(
    conn: duckdb.DuckDBPyConnection,
    t0_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Queries DuckDB for eligible customers and calculates pre-period metrics.

    Args:
        conn: DuckDB database connection.
        t0_date: Policy intervention rollout timestamp t_0.

    Returns:
        Tuple of (pre_summary_df, weekly_tx_df).
    """
    t0_str = t0_date.strftime("%Y-%m-%d")

    # 1. Fetch weekly aggregate transactions per customer
    weekly_tx = conn.execute(f"""
        SELECT
            customer_id,
            DATE_TRUNC('week', invoice_date) AS week_date,
            COUNT(DISTINCT invoice_no) AS cf_orders,
            SUM(line_total) AS cf_spend
        FROM fact_orders
        GROUP BY customer_id, DATE_TRUNC('week', invoice_date)
    """).df()
    weekly_tx["week_date"] = pd.to_datetime(weekly_tx["week_date"])

    # 2. Identify first order week per customer
    first_weeks = weekly_tx.groupby("customer_id")["week_date"].min().reset_index()
    first_weeks.columns = ["customer_id", "first_week"]

    # Filter strictly to customers whose first order occurred before t0
    eligible_first_weeks = first_weeks[first_weeks["first_week"] < t0_date].copy()
    eligible_ids = set(eligible_first_weeks["customer_id"])
    weekly_tx = weekly_tx[weekly_tx["customer_id"].isin(eligible_ids)].copy()

    # 3. Pre-period invoices to calculate spend, order count, and median basket size
    pre_invoices = conn.execute(f"""
        SELECT customer_id, invoice_no, invoice_date, SUM(line_total) AS order_val
        FROM fact_orders
        WHERE invoice_date < '{t0_str}'
        GROUP BY customer_id, invoice_no, invoice_date
    """).df()
    pre_invoices = pre_invoices[pre_invoices["customer_id"].isin(eligible_ids)].copy()

    pre_summary = (
        pre_invoices.groupby("customer_id")
        .agg(
            pre_spend=("order_val", "sum"),
            pre_orders=("invoice_no", "count"),
            median_order_val=("order_val", "median"),
        )
        .reset_index()
    )

    pre_summary = pre_summary.merge(eligible_first_weeks, on="customer_id")

    # Pre-period week count: from first_week to the week before t0
    w_t0 = t0_date - pd.to_timedelta(t0_date.weekday(), unit="D")
    pre_summary["pre_weeks"] = np.maximum(
        1, ((w_t0 - pre_summary["first_week"]).dt.days // 7)
    )
    pre_summary["pre_order_rate"] = (
        pre_summary["pre_orders"] / pre_summary["pre_weeks"]
    )

    return pre_summary, weekly_tx


def build_balanced_panel(
    pre_summary: pd.DataFrame,
    weekly_tx: pd.DataFrame,
    last_week: pd.Timestamp,
) -> pd.DataFrame:
    """Builds a balanced panel from each customer's first order week to last_week."""
    records = []
    for _, row in pre_summary[["customer_id", "first_week"]].iterrows():
        c_id = row["customer_id"]
        fw = row["first_week"]
        w_range = pd.date_range(start=fw, end=last_week, freq="W-MON")
        records.append(pd.DataFrame({"customer_id": c_id, "week_date": w_range}))

    panel = pd.concat(records, ignore_index=True)
    panel = panel.merge(weekly_tx, on=["customer_id", "week_date"], how="left")
    panel["cf_orders"] = panel["cf_orders"].fillna(0).astype(int)
    panel["cf_spend"] = panel["cf_spend"].fillna(0.0)
    return panel


def simulate_policy_intervention(
    db_path: str = DB_PATH,
    intervention_date: str = INTERVENTION_DATE,
    treatment_share: float = TREATMENT_SHARE,
    order_rate_change: float = ORDER_RATE_CHANGE,
    aov_change: float = AOV_CHANGE,
    top_clv_churn_hazard: float = TOP_CLV_CHURN_HAZARD,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulates policy rollout on balanced customer-by-week panel with known multiplicative effects.

    Args:
        db_path: Path to DuckDB warehouse.
        intervention_date: ISO date string for policy rollout date t_0 (read from config.yaml).
        treatment_share: Fraction of customers randomly assigned to treatment within each quintile.
        order_rate_change: Relative multiplicative order rate lift applied to zero weeks.
        aov_change: Relative multiplicative AOV scale applied to orders.
        top_clv_churn_hazard: Discrete hazard rate for top spend quintile churn.
        random_seed: Seed for reproducible assignment and draws.

    Returns:
        Balanced panel DataFrame with counterfactual, observed, and true unit-level causal effects.
    """
    print("\n=================== Causal Policy Intervention Simulation ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Configured Treatment Share:             {treatment_share * 100:.1f}% (stratified by pre-spend)")
    print(f"Configured Order Rate Change:           {order_rate_change * 100:+.1f}%")
    print(f"Configured AOV Change:                  {aov_change * 100:+.1f}%")
    print(f"Configured Top-Spend Churn Hazard:      {top_clv_churn_hazard * 100:.1f}%")
    print(f"Random Seed:                            {random_seed}")

    conn = duckdb.connect(db_path)
    t0_date = pd.to_datetime(intervention_date)

    # 1. Extract customer pre-period metrics and historical weekly transactions
    pre_summary, weekly_tx = extract_customer_pre_period_metrics(conn, t0_date)
    last_week = weekly_tx["week_date"].max()
    n_eligible = len(pre_summary)
    print(f"Eligible Customers (first order < t0):  {n_eligible:,}")

    # 2. Stratified Treatment Assignment by observed pre-period spend quintile
    pre_summary["spend_quintile"] = pd.qcut(
        pre_summary["pre_spend"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    rng = np.random.default_rng(random_seed)
    treated_set = set()
    for q in [1, 2, 3, 4, 5]:
        c_ids = pre_summary[pre_summary["spend_quintile"] == q]["customer_id"].values.copy()
        n_q = len(c_ids)
        n_tr = int(round(n_q * treatment_share))
        treated_q = rng.permutation(c_ids)[:n_tr]
        treated_set.update(treated_q)

    pre_summary["is_treated"] = pre_summary["customer_id"].isin(treated_set).astype(int)
    n_treated = pre_summary["is_treated"].sum()
    print(f"Assigned Treated Customers:             {n_treated:,} ({n_treated / n_eligible * 100:.2f}%)")

    # 3. Draw churn week for treated customers in top quintile of OBSERVED pre-period spend
    top_treated_mask = (pre_summary["spend_quintile"] == 5) & (pre_summary["is_treated"] == 1)
    n_top_treated = int(top_treated_mask.sum())
    churn_steps = rng.geometric(p=top_clv_churn_hazard, size=n_top_treated)
    churn_map = dict(zip(pre_summary.loc[top_treated_mask, "customer_id"], churn_steps))
    print(f"Treated Top-Spend Customers (Q5):       {n_top_treated:,} (churn hazard={top_clv_churn_hazard:.2f})")

    # 4. Construct Balanced Customer-by-Week Panel
    panel = build_balanced_panel(pre_summary, weekly_tx, last_week)
    print(f"Constructed Balanced Panel Rows:        {len(panel):,}")

    # Merge customer metadata into panel
    panel = panel.merge(
        pre_summary[
            [
                "customer_id",
                "is_treated",
                "spend_quintile",
                "median_order_val",
                "pre_order_rate",
            ]
        ],
        on="customer_id",
        how="left",
    )

    # 5. Define Post-Period Weeks and Steps
    post_weeks = sorted(panel[panel["week_date"] >= t0_date]["week_date"].unique())
    post_week_to_idx = {w: i + 1 for i, w in enumerate(post_weeks)}
    panel["post_step"] = panel["week_date"].map(post_week_to_idx)
    panel["is_post_period"] = panel["post_step"].notna().astype(int)

    # Identify churn weeks
    panel["churn_step"] = panel["customer_id"].map(churn_map)
    panel["is_churned"] = (
        (panel["is_post_period"] == 1)
        & panel["churn_step"].notna()
        & (panel["post_step"] > panel["churn_step"])
    )

    # 6. Apply Intervention Simulation Logic
    panel["observed_weekly_orders"] = panel["cf_orders"].astype(float)
    panel["observed_weekly_spend"] = panel["cf_spend"].astype(float)

    tr_post = (panel["is_treated"] == 1) & (panel["is_post_period"] == 1)

    # Churned treated customers: orders removed after churn step
    churned_mask = tr_post & panel["is_churned"]
    panel.loc[churned_mask, "observed_weekly_orders"] = 0.0
    panel.loc[churned_mask, "observed_weekly_spend"] = 0.0

    # Active treated post-t0 customers
    active_tr_post = tr_post & (~panel["is_churned"])

    # Action 1: In a zero-purchase week, add a new order with probability (order_rate_change * pre_weekly_order_rate)
    zero_mask = active_tr_post & (panel["cf_orders"] == 0)
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        add_probs = np.clip(
            panel.loc[zero_mask, "pre_order_rate"].values * order_rate_change,
            0.0,
            1.0,
        )
        add_order = rng.random(size=n_zero) < add_probs
        med_vals = panel.loc[zero_mask, "median_order_val"].values

        panel.loc[zero_mask, "observed_weekly_orders"] = np.where(add_order, 1.0, 0.0)
        panel.loc[zero_mask, "observed_weekly_spend"] = np.where(
            add_order, med_vals * (1.0 + aov_change), 0.0
        )

    # Action 2: In a positive-purchase week, scale order values by (1 + aov_change)
    nonzero_mask = active_tr_post & (panel["cf_orders"] > 0)
    panel.loc[nonzero_mask, "observed_weekly_spend"] = (
        panel.loc[nonzero_mask, "cf_spend"] * (1.0 + aov_change)
    )

    # 7. Store Standard Column Aliases & True Effects
    panel["counterfactual_weekly_orders"] = panel["cf_orders"].astype(float)
    panel["counterfactual_weekly_spend"] = panel["cf_spend"].astype(float)

    panel["true_effect_orders"] = (
        panel["observed_weekly_orders"] - panel["counterfactual_weekly_orders"]
    )
    panel["true_effect_spend"] = (
        panel["observed_weekly_spend"] - panel["counterfactual_weekly_spend"]
    )

    # Outcome 2: Winsorized spend at 99th percentile of positive counterfactual spend
    pos_spend = panel.loc[panel["counterfactual_weekly_spend"] > 0, "counterfactual_weekly_spend"]
    p99_val = float(np.percentile(pos_spend, 99)) if len(pos_spend) > 0 else 5000.0

    panel["counterfactual_weekly_spend_winsor"] = np.clip(
        panel["counterfactual_weekly_spend"], 0.0, p99_val
    )
    panel["observed_weekly_spend_winsor"] = np.clip(
        panel["observed_weekly_spend"], 0.0, p99_val
    )
    panel["true_effect_spend_winsor"] = (
        panel["observed_weekly_spend_winsor"]
        - panel["counterfactual_weekly_spend_winsor"]
    )

    # Outcome 3: Active in the last 90 days (13 weeks rolling window)
    panel = panel.sort_values(by=["customer_id", "week_date"]).reset_index(drop=True)

    panel["counterfactual_active_last_90d"] = (
        panel.groupby("customer_id")["counterfactual_weekly_orders"]
        .rolling(13, min_periods=1)
        .sum()
        .reset_index(drop=True)
        > 0
    ).astype(float)

    panel["observed_active_last_90d"] = (
        panel.groupby("customer_id")["observed_weekly_orders"]
        .rolling(13, min_periods=1)
        .sum()
        .reset_index(drop=True)
        > 0
    ).astype(float)

    panel["true_effect_active_last_90d"] = (
        panel["observed_active_last_90d"] - panel["counterfactual_active_last_90d"]
    )

    # Compatibility alias columns for regression modeling
    panel["weekly_orders"] = panel["observed_weekly_orders"]
    panel["weekly_spend"] = panel["observed_weekly_spend"]
    panel["weekly_spend_winsor"] = panel["observed_weekly_spend_winsor"]
    panel["active_last_90d"] = panel["observed_active_last_90d"]

    # 8. Store Simulation Panel into DuckDB
    try:
        conn.execute("CREATE OR REPLACE TABLE causal_experiment_data AS SELECT * FROM panel")
        print("Persisted simulation panel to DuckDB table: causal_experiment_data")
    except Exception as e:
        print(f"Warning: Failed to persist table to DuckDB: {e}")
    finally:
        conn.close()

    # 9. Compute and Print Ground-Truth Effects
    print_ground_truth_effects(panel, p99_val)

    return panel


def get_ground_truth_effects(panel: pd.DataFrame) -> Dict[str, float]:
    """Computes population ground truth average treatment effect on the treated (ATT)."""
    tr_post = (panel["is_treated"] == 1) & (panel["is_post_period"] == 1)

    return {
        "weekly_orders": float(panel.loc[tr_post, "true_effect_orders"].mean()),
        "weekly_spend_raw": float(panel.loc[tr_post, "true_effect_spend"].mean()),
        "weekly_spend_winsor": float(panel.loc[tr_post, "true_effect_spend_winsor"].mean()),
        "active_last_90d": float(panel.loc[tr_post, "true_effect_active_last_90d"].mean()),
    }


def print_ground_truth_effects(panel: pd.DataFrame, p99_val: float) -> None:
    """Formats and prints ground truth ATT across all three outcomes."""
    effects = get_ground_truth_effects(panel)
    tr_post = (panel["is_treated"] == 1) & (panel["is_post_period"] == 1)

    print("\n" + "=" * 80)
    print("GROUND TRUTH AVERAGE TREATMENT EFFECTS ON THE TREATED (ATT)")
    print("=" * 80)
    print(f"1. Weekly Orders ATT:                  {effects['weekly_orders']:+.5f} orders / cust-week")
    print(f"2. Weekly Spend (Raw) ATT:             ${effects['weekly_spend_raw']:+.2f} / cust-week")
    print(f"3. Weekly Spend (Winsorized 99th) ATT: ${effects['weekly_spend_winsor']:+.2f} / cust-week (P99 = ${p99_val:,.2f})")
    print(f"4. Active in Last 90 Days ATT:         {effects['active_last_90d']:+.5f} (probability lift)")

    print("\n--- Quintile Breakdown of Realized ATT (Post-Treated Period) ---")
    for q in [1, 2, 3, 4, 5]:
        sub = panel[tr_post & (panel["spend_quintile"] == q)]
        att_ord = sub["true_effect_orders"].mean()
        att_sp_raw = sub["true_effect_spend"].mean()
        att_sp_win = sub["true_effect_spend_winsor"].mean()
        att_act = sub["true_effect_active_last_90d"].mean()
        print(f"  Quintile {q}: Orders={att_ord:+.5f} | Spend(Raw)=${att_sp_raw:+.2f} | Spend(Win99)=${att_sp_win:+.2f} | Active90d={att_act:+.5f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    simulate_policy_intervention()
