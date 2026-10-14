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
   draws a churn week from the configured weekly hazard (converted from cumulative churn or severe)
   and removes all orders after it.

Outcomes generated (counterfactual, observed, realized true effect, and expected true effect):
1. `weekly_orders`
2. `weekly_spend` (and `weekly_spend_winsor` winsorized at 99th percentile)
3. `active_last_90d` (indicator of any purchase in past 13 weeks / 90 days)
"""

from typing import Any, Dict, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd

from clv_causal.config import (
    AOV_CHANGE,
    CHURN_SCENARIO,
    DB_PATH,
    INTERVENTION_DATE,
    ORDER_RATE_CHANGE,
    RANDOM_SEED,
    TOP_SPEND_CHURN_CUMULATIVE,
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
    weekly_tx = conn.execute("""
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

    # Diagnostic: Raw order rate (including first order)
    pre_summary["raw_pre_order_rate"] = (
        pre_summary["pre_orders"] / pre_summary["pre_weeks"]
    )

    # Shrunken repeat order rate (Empirical Bayes shrinkage excluding first qualifying order):
    # repeat_orders = pre_orders - 1
    # lambda_hat = (repeat_orders + lambda_bar * beta) / (pre_weeks + beta) with beta = 4.0 weeks
    repeat_orders = np.maximum(0, pre_summary["pre_orders"] - 1)
    total_repeats = repeat_orders.sum()
    total_weeks = pre_summary["pre_weeks"].sum()
    global_repeat_rate = float(total_repeats / total_weeks) if total_weeks > 0 else 0.05
    beta_prior = 4.0
    alpha_prior = global_repeat_rate * beta_prior

    pre_summary["pre_order_rate"] = (
        (repeat_orders + alpha_prior) / (pre_summary["pre_weeks"] + beta_prior)
    )

    return pre_summary, weekly_tx


def build_balanced_panel(
    pre_summary: pd.DataFrame,
    weekly_tx: pd.DataFrame,
    last_week: pd.Timestamp,
) -> pd.DataFrame:
    """Builds a balanced panel from each customer's first order week to last_week."""
    c_ids = []
    dates = []
    for c_id, fw in zip(pre_summary["customer_id"].values, pre_summary["first_week"].values):
        w_range = pd.date_range(start=fw, end=last_week, freq="W-MON")
        c_ids.extend([c_id] * len(w_range))
        dates.extend(w_range)

    panel = pd.DataFrame({"customer_id": np.array(c_ids, dtype=np.int64), "week_date": dates})
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
    churn_scenario: str = CHURN_SCENARIO,
    top_spend_churn_cumulative: float = TOP_SPEND_CHURN_CUMULATIVE,
    top_clv_churn_hazard: Optional[float] = None,
    random_seed: int = RANDOM_SEED,
    assignment_mode: str = "stratified_random",
    persist_to_db: bool = True,
) -> pd.DataFrame:
    """Simulates policy rollout on balanced customer-by-week panel with known multiplicative effects.

    Args:
        db_path: Path to DuckDB warehouse.
        intervention_date: ISO date string for policy rollout date t_0 (read from config.yaml).
        treatment_share: Fraction of customers randomly assigned to treatment within each quintile.
        order_rate_change: Relative multiplicative order rate lift applied to zero weeks.
        aov_change: Relative multiplicative AOV scale applied to orders.
        churn_scenario: Named scenario ('baseline', 'hidden_degradation', or 'severe').
        top_spend_churn_cumulative: Cumulative churn probability over post-period (0.20 baseline, 0.40 hidden_degradation).
        top_clv_churn_hazard: Legacy alias for backwards compatibility.
        random_seed: Seed for reproducible assignment and draws.
        assignment_mode: Assignment mechanism ('stratified_random' or 'confounded').
        persist_to_db: Whether to overwrite DuckDB table causal_experiment_data (default True).

    Returns:
        Balanced panel DataFrame with counterfactual, observed, and true unit-level causal effects.
    """
    if top_clv_churn_hazard is not None and top_spend_churn_cumulative == 0.20:
        top_spend_churn_cumulative = top_clv_churn_hazard

    conn = duckdb.connect(db_path)
    t0_date = pd.to_datetime(intervention_date)
    t0_str = t0_date.strftime("%Y-%m-%d")

    # 1. Extract customer pre-period metrics and historical weekly transactions
    pre_summary, weekly_tx = extract_customer_pre_period_metrics(conn, t0_date)
    last_week = weekly_tx["week_date"].max()
    n_eligible = len(pre_summary)

    # 2. Spend Quintiles & Treatment Assignment
    pre_summary["spend_quintile"] = pd.qcut(
        pre_summary["pre_spend"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    rng = np.random.default_rng(random_seed)

    if assignment_mode.lower() == "confounded":
        # Confounded assignment mode: treatment probability rises with customer's pre-period spend growth
        tx = conn.execute(f"""
            SELECT customer_id, invoice_date, SUM(line_total) as order_val
            FROM fact_orders WHERE invoice_date < '{t0_str}'
            GROUP BY customer_id, invoice_no, invoice_date
        """).df()
        tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
        first_dates = tx.groupby("customer_id")["invoice_date"].min().reset_index().rename(columns={"invoice_date": "first_date"})
        tx = tx.merge(first_dates, on="customer_id")
        tx["mid_date"] = tx["first_date"] + (t0_date - tx["first_date"]) / 2.0
        early_spend = tx[tx["invoice_date"] < tx["mid_date"]].groupby("customer_id")["order_val"].sum()
        late_spend = tx[tx["invoice_date"] >= tx["mid_date"]].groupby("customer_id")["order_val"].sum()

        growth_df = pd.DataFrame({"customer_id": pre_summary["customer_id"]})
        growth_df["early"] = growth_df["customer_id"].map(early_spend).fillna(0.0)
        growth_df["late"] = growth_df["customer_id"].map(late_spend).fillna(0.0)
        # Explicit spend growth formula: normalized relative growth between 2nd half and 1st half of pre-period
        growth_df["spend_growth"] = (growth_df["late"] - growth_df["early"]) / (growth_df["late"] + growth_df["early"] + 10.0)
        growth_df["growth_rank"] = growth_df["spend_growth"].rank(pct=True)

        threshold = growth_df["growth_rank"].quantile(1.0 - treatment_share)
        treated_set = set(growth_df[growth_df["growth_rank"] >= threshold]["customer_id"])
        pre_summary["spend_growth"] = growth_df["spend_growth"].values
    else:
        # Stratified random assignment of configured share within each pre-period spend quintile
        treated_set = set()
        for q in [1, 2, 3, 4, 5]:
            c_ids = pre_summary[pre_summary["spend_quintile"] == q]["customer_id"].values.copy()
            n_q = len(c_ids)
            n_tr = int(round(n_q * treatment_share))
            treated_q = rng.permutation(c_ids)[:n_tr]
            treated_set.update(treated_q)

    pre_summary["is_treated"] = pre_summary["customer_id"].isin(treated_set).astype(int)
    n_treated = pre_summary["is_treated"].sum()

    # 3. Construct Balanced Customer-by-Week Panel
    panel = build_balanced_panel(pre_summary, weekly_tx, last_week)

    # Merge customer metadata into panel
    merge_cols = [
        "customer_id",
        "is_treated",
        "spend_quintile",
        "median_order_val",
        "pre_order_rate",
        "raw_pre_order_rate",
    ]
    if "spend_growth" in pre_summary.columns:
        merge_cols.append("spend_growth")

    panel = panel.merge(
        pre_summary[merge_cols],
        on="customer_id",
        how="left",
    )

    # 4. Define Post-Period Weeks and Steps
    post_weeks = sorted(panel[panel["week_date"] >= t0_date]["week_date"].unique())
    n_post_weeks = len(post_weeks)
    post_week_to_idx = {w: i + 1 for i, w in enumerate(post_weeks)}
    panel["post_step"] = panel["week_date"].map(post_week_to_idx)
    panel["is_post_period"] = panel["post_step"].notna().astype(int)

    # Convert cumulative churn probability to per-week hazard
    if churn_scenario.lower() == "severe":
        weekly_churn_hazard = 0.20
        eff_cum_churn = 1.0 - (1.0 - 0.20) ** n_post_weeks
    elif churn_scenario.lower() == "hidden_degradation":
        eff_cum_churn = 0.40 if top_spend_churn_cumulative == 0.20 else top_spend_churn_cumulative
        weekly_churn_hazard = 1.0 - (1.0 - eff_cum_churn) ** (1.0 / max(1, n_post_weeks))
    else:  # baseline
        eff_cum_churn = top_spend_churn_cumulative
        weekly_churn_hazard = 1.0 - (1.0 - eff_cum_churn) ** (1.0 / max(1, n_post_weeks))

    print("\n=================== Causal Policy Intervention Simulation ===================")
    print(f"Policy Date (t_0):                      {intervention_date}")
    print(f"Assignment Mode:                        {assignment_mode}")
    print(f"Configured Treatment Share:             {treatment_share * 100:.1f}%")
    print(f"Configured Order Rate Change:           {order_rate_change * 100:+.1f}%")
    print(f"Configured AOV Change:                  {aov_change * 100:+.1f}%")
    print(f"Churn Scenario:                         {churn_scenario}")
    print(f"Post-Period Weeks (n_post_weeks):       {n_post_weeks}")
    print(f"Top-Spend Cumulative Churn:             {eff_cum_churn * 100:.1f}%")
    print(f"Resulting Weekly Churn Hazard:          {weekly_churn_hazard:.5f} ({weekly_churn_hazard * 100:.3f}%/week)")
    print(f"Random Seed:                            {random_seed}")
    print(f"Eligible Customers (first order < t0):  {n_eligible:,}")
    print(f"Assigned Treated Customers:             {n_treated:,} ({n_treated / n_eligible * 100:.2f}%)")
    print(f"Constructed Balanced Panel Rows:        {len(panel):,}")

    # 5. Draw churn week for treated customers in top quintile of OBSERVED pre-period spend
    top_treated_mask = (pre_summary["spend_quintile"] == 5) & (pre_summary["is_treated"] == 1)
    n_top_treated = int(top_treated_mask.sum())
    churn_steps = rng.geometric(p=weekly_churn_hazard, size=n_top_treated)
    churn_map = dict(zip(pre_summary.loc[top_treated_mask, "customer_id"], churn_steps))
    print(f"Treated Top-Spend Customers (Q5):       {n_top_treated:,} (churn hazard={weekly_churn_hazard:.5f})")

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
    probs = np.zeros(len(panel))
    if n_zero > 0:
        add_probs = np.clip(
            panel.loc[zero_mask, "pre_order_rate"].values * order_rate_change,
            0.0,
            1.0,
        )
        probs[zero_mask] = add_probs
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

    # 7. Counterfactual, Realized True Effects, and Expected True Effects
    panel["counterfactual_weekly_orders"] = panel["cf_orders"].astype(float)
    panel["counterfactual_weekly_spend"] = panel["cf_spend"].astype(float)

    # Realized effect
    panel["realized_effect_orders"] = (
        panel["observed_weekly_orders"] - panel["counterfactual_weekly_orders"]
    )
    panel["realized_effect_spend"] = (
        panel["observed_weekly_spend"] - panel["counterfactual_weekly_spend"]
    )

    # Expected effect
    panel["expected_effect_orders"] = 0.0
    panel["expected_effect_spend"] = 0.0

    panel.loc[churned_mask, "expected_effect_orders"] = - panel.loc[churned_mask, "cf_orders"].astype(float)
    panel.loc[churned_mask, "expected_effect_spend"] = - panel.loc[churned_mask, "cf_spend"]

    panel.loc[zero_mask, "expected_effect_orders"] = probs[zero_mask]
    panel.loc[zero_mask, "expected_effect_spend"] = (
        probs[zero_mask] * panel.loc[zero_mask, "median_order_val"].values * (1.0 + aov_change)
    )

    panel.loc[nonzero_mask, "expected_effect_orders"] = 0.0
    panel.loc[nonzero_mask, "expected_effect_spend"] = (
        panel.loc[nonzero_mask, "cf_spend"] * aov_change
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
    panel["realized_effect_spend_winsor"] = (
        panel["observed_weekly_spend_winsor"]
        - panel["counterfactual_weekly_spend_winsor"]
    )

    # Expected effect for winsorized spend
    panel["expected_effect_spend_winsor"] = 0.0
    panel.loc[churned_mask, "expected_effect_spend_winsor"] = - panel.loc[churned_mask, "counterfactual_weekly_spend_winsor"]
    panel.loc[zero_mask, "expected_effect_spend_winsor"] = (
        probs[zero_mask] * np.clip(panel.loc[zero_mask, "median_order_val"].values * (1.0 + aov_change), 0.0, p99_val)
    )
    panel.loc[nonzero_mask, "expected_effect_spend_winsor"] = (
        np.clip(panel.loc[nonzero_mask, "cf_spend"] * (1.0 + aov_change), 0.0, p99_val)
        - panel.loc[nonzero_mask, "counterfactual_weekly_spend_winsor"]
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

    panel["realized_effect_active_last_90d"] = (
        panel["observed_active_last_90d"] - panel["counterfactual_active_last_90d"]
    )
    panel["expected_effect_active_last_90d"] = panel["realized_effect_active_last_90d"]

    # Legacy / convenience aliases
    panel["true_effect_orders"] = panel["realized_effect_orders"]
    panel["true_effect_spend"] = panel["realized_effect_spend"]
    panel["true_effect_spend_winsor"] = panel["realized_effect_spend_winsor"]
    panel["true_effect_active_last_90d"] = panel["realized_effect_active_last_90d"]

    panel["weekly_orders"] = panel["observed_weekly_orders"]
    panel["weekly_spend"] = panel["observed_weekly_spend"]
    panel["weekly_spend_winsor"] = panel["observed_weekly_spend_winsor"]
    panel["active_last_90d"] = panel["observed_active_last_90d"]

    # 8. Store Simulation Panel into DuckDB
    if persist_to_db:
        try:
            conn.execute("CREATE OR REPLACE TABLE causal_experiment_data AS SELECT * FROM panel")
            print("Persisted simulation panel to DuckDB table: causal_experiment_data")
        except Exception as e:
            print(f"Warning: Failed to persist table to DuckDB: {e}")
        finally:
            conn.close()
    else:
        conn.close()

    # 9. Compute and Print Ground-Truth Effects
    print_ground_truth_effects(panel, p99_val)

    return panel


def get_ground_truth_effects(panel: pd.DataFrame) -> Dict[str, Any]:
    """Computes population ground truth average treatment effect on the treated (ATT)."""
    tr_post = (panel["is_treated"] == 1) & (panel["is_post_period"] == 1)
    q14 = tr_post & (panel["spend_quintile"].isin([1, 2, 3, 4]))
    q5 = tr_post & (panel["spend_quintile"] == 5)

    return {
        "overall": {
            "realized_orders": float(panel.loc[tr_post, "realized_effect_orders"].mean()),
            "expected_orders": float(panel.loc[tr_post, "expected_effect_orders"].mean()),
            "realized_spend_winsor": float(panel.loc[tr_post, "realized_effect_spend_winsor"].mean()),
            "expected_spend_winsor": float(panel.loc[tr_post, "expected_effect_spend_winsor"].mean()),
            "realized_spend_raw": float(panel.loc[tr_post, "realized_effect_spend"].mean()),
            "expected_spend_raw": float(panel.loc[tr_post, "expected_effect_spend"].mean()),
            "realized_active_90d": float(panel.loc[tr_post, "realized_effect_active_last_90d"].mean()),
        },
        "q1_to_q4_pooled": {
            "realized_orders": float(panel.loc[q14, "realized_effect_orders"].mean()),
            "expected_orders": float(panel.loc[q14, "expected_effect_orders"].mean()),
            "realized_spend_winsor": float(panel.loc[q14, "realized_effect_spend_winsor"].mean()),
            "expected_spend_winsor": float(panel.loc[q14, "expected_effect_spend_winsor"].mean()),
            "realized_spend_raw": float(panel.loc[q14, "realized_effect_spend"].mean()),
            "expected_spend_raw": float(panel.loc[q14, "expected_effect_spend"].mean()),
            "realized_active_90d": float(panel.loc[q14, "realized_effect_active_last_90d"].mean()),
        },
        "q5": {
            "realized_orders": float(panel.loc[q5, "realized_effect_orders"].mean()),
            "expected_orders": float(panel.loc[q5, "expected_effect_orders"].mean()),
            "realized_spend_winsor": float(panel.loc[q5, "realized_effect_spend_winsor"].mean()),
            "expected_spend_winsor": float(panel.loc[q5, "expected_effect_spend_winsor"].mean()),
            "realized_spend_raw": float(panel.loc[q5, "realized_effect_spend"].mean()),
            "expected_spend_raw": float(panel.loc[q5, "expected_effect_spend"].mean()),
            "realized_active_90d": float(panel.loc[q5, "realized_effect_active_last_90d"].mean()),
        },
        # Flat legacy keys for compatibility
        "weekly_orders": float(panel.loc[tr_post, "realized_effect_orders"].mean()),
        "weekly_spend_raw": float(panel.loc[tr_post, "realized_effect_spend"].mean()),
        "weekly_spend_winsor": float(panel.loc[tr_post, "realized_effect_spend_winsor"].mean()),
        "active_last_90d": float(panel.loc[tr_post, "realized_effect_active_last_90d"].mean()),
    }


def print_ground_truth_effects(panel: pd.DataFrame, p99_val: float) -> None:
    """Formats and prints ground truth ATT across all three outcomes."""
    effects = get_ground_truth_effects(panel)
    tr_post = (panel["is_treated"] == 1) & (panel["is_post_period"] == 1)

    print("\n" + "=" * 88)
    print("GROUND TRUTH AVERAGE TREATMENT EFFECTS ON THE TREATED (ATT)")
    print("=" * 88)
    print(f"{'Cohort':<22} | {'Weekly Orders (Real/Exp)':<25} | {'Spend Winsor (Real/Exp)':<23} | {'Active 90d':<10}")
    print("-" * 88)

    ov = effects["overall"]
    q14 = effects["q1_to_q4_pooled"]
    q5 = effects["q5"]

    print(f"{'Overall (All Treated)':<22} | {ov['realized_orders']:+.5f} / {ov['expected_orders']:+.5f}   | ${ov['realized_spend_winsor']:+.2f} / ${ov['expected_spend_winsor']:+.2f}    | {ov['realized_active_90d']:+.5f}")
    print(f"{'Q1-Q4 Pooled':<22} | {q14['realized_orders']:+.5f} / {q14['expected_orders']:+.5f}   | ${q14['realized_spend_winsor']:+.2f} / ${q14['expected_spend_winsor']:+.2f}    | {q14['realized_active_90d']:+.5f}")
    print(f"{'Quintile 5 (Top Spend)':<22} | {q5['realized_orders']:+.5f} / {q5['expected_orders']:+.5f}   | ${q5['realized_spend_winsor']:+.2f} / ${q5['expected_spend_winsor']:+.2f}    | {q5['realized_active_90d']:+.5f}")

    print("\n--- Detailed Breakdown by Pre-Spend Quintile ---")
    print(f"{'Quintile':<10} | {'Treated':<8} | {'Orders (Real)':<14} | {'Orders (Exp)':<14} | {'Spend Win (Real)':<16} | {'Spend Win (Exp)':<16} | {'Active 90d':<10}")
    print("-" * 96)
    for q in [1, 2, 3, 4, 5]:
        sub = panel[tr_post & (panel["spend_quintile"] == q)]
        n_c = sub["customer_id"].nunique()
        r_o = sub["realized_effect_orders"].mean()
        e_o = sub["expected_effect_orders"].mean()
        r_w = sub["realized_effect_spend_winsor"].mean()
        e_w = sub["expected_effect_spend_winsor"].mean()
        r_a = sub["realized_effect_active_last_90d"].mean()
        print(f"Quintile {q:<2} | {n_c:<8} | {r_o:+.5f}      | {e_o:+.5f}      | ${r_w:+.2f}           | ${e_w:+.2f}           | {r_a:+.5f}")

    print("=" * 88 + "\n")


if __name__ == "__main__":
    simulate_policy_intervention()
