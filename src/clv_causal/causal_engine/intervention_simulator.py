import os
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

from clv_causal.config import DB_PATH

def simulate_causal_intervention(intervention_date="2010-06-01", policy_lift_pct=0.15):
    """
    Priority 1 & Priority 2 Redesign:
    Simulates a documented business policy rollout at t_0 (e.g. targeted free-shipping / premium loyalty pricing policy)
    applied to Non-UK customer cohorts while preserving UK customers as untreated baseline control.
    
    Rather than injecting a raw ATT outcome constant, the post-period transaction mechanism is modified:
    order quantities/basket values for treated customers in post-period receive a mechanism shift (+15% policy lift).
    
    Also generates pre-intervention treatment/control balance diagnostics (SMD, revenue trends, AOV, order frequency).
    """
    print(f"Step 4: Simulating Causal Policy Rollout at t_0 = {intervention_date}...")
    conn = duckdb.connect(DB_PATH)
    
    # Query transaction line items with customer and country attributes
    tx_df = conn.execute("""
        SELECT
            ft.transaction_id,
            ft.customer_id,
            ft.invoice_no,
            ft.invoice_date,
            ft.line_total,
            ft.quantity,
            ft.unit_price,
            dc.country
        FROM fact_transactions ft
        JOIN dim_customers dc ON ft.customer_id = dc.customer_id
    """).df()
    
    tx_df['invoice_date'] = pd.to_datetime(tx_df['invoice_date'])
    tx_df['week_date'] = tx_df['invoice_date'].dt.to_period('W').dt.start_time
    t0_date = pd.to_datetime(intervention_date)
    
    # Cohort assignment: Non-UK = Treated (1), UK = Control (0)
    tx_df['is_treated_cohort'] = np.where(tx_df['country'] != 'United Kingdom', 1, 0)
    tx_df['is_post_period'] = np.where(tx_df['week_date'] >= t0_date, 1, 0)
    
    # Mechanism transformation in post period for treated cohort
    # Simulate policy impact on basket line total via quantity/monetization policy mechanism
    np.random.seed(42)
    mechanism_multiplier = 1.0 + (tx_df['is_treated_cohort'] * tx_df['is_post_period'] * (policy_lift_pct + np.random.normal(0.0, 0.02, size=len(tx_df))))
    tx_df['simulated_line_total'] = np.round(tx_df['line_total'] * np.maximum(0.5, mechanism_multiplier), 2)
    
    # Aggregate to weekly panel per customer
    panel_df = tx_df.groupby(['customer_id', 'country', 'is_treated_cohort', 'week_date', 'is_post_period']).agg(
        weekly_spend=('simulated_line_total', 'sum'),
        counterfactual_weekly_spend=('line_total', 'sum'),
        weekly_orders=('invoice_no', 'nunique'),
        weekly_items=('quantity', 'sum')
    ).reset_index()
    
    panel_df.rename(columns={'weekly_spend': 'observed_weekly_spend'}, inplace=True)
    
    # Save panel to DuckDB table 'causal_experiment_data'
    conn.execute("CREATE OR REPLACE TABLE causal_experiment_data AS SELECT * FROM panel_df")
    
    # --- Priority 2: Pre-Intervention Treatment vs Control Balance Diagnostics ---
    print("Computing Pre-Intervention Baseline Diagnostics (Treatment vs Control)...")
    pre_panel = panel_df[panel_df['is_post_period'] == 0].copy()
    
    pre_diag = []
    cohorts = [(1, "Treated (Non-UK)"), (0, "Control (UK)")]
    
    metrics_summary = {}
    for code, label in cohorts:
        sub = pre_panel[pre_panel['is_treated_cohort'] == code]
        cust_cnt = sub['customer_id'].nunique()
        avg_weekly_rev = sub.groupby('customer_id')['observed_weekly_spend'].mean().mean() if cust_cnt > 0 else 0.0
        std_weekly_rev = sub.groupby('customer_id')['observed_weekly_spend'].mean().std() if cust_cnt > 0 else 0.0
        avg_orders_per_cust = sub.groupby('customer_id')['weekly_orders'].sum().mean() if cust_cnt > 0 else 0.0
        std_orders = sub.groupby('customer_id')['weekly_orders'].sum().std() if cust_cnt > 0 else 0.0
        avg_aov = sub['observed_weekly_spend'].sum() / max(1, sub['weekly_orders'].sum())
        
        metrics_summary[code] = {
            'label': label,
            'cust_cnt': cust_cnt,
            'avg_weekly_rev': avg_weekly_rev,
            'std_weekly_rev': std_weekly_rev,
            'avg_orders_per_cust': avg_orders_per_cust,
            'std_orders': std_orders,
            'aov': avg_aov
        }
    
    # Calculate Standardized Mean Difference (SMD) for baseline metrics
    tr = metrics_summary[1]
    co = metrics_summary[0]
    
    smd_rev = (tr['avg_weekly_rev'] - co['avg_weekly_rev']) / np.sqrt((tr['std_weekly_rev']**2 + co['std_weekly_rev']**2) / 2.0) if (tr['std_weekly_rev']**2 + co['std_weekly_rev']**2) > 0 else 0.0
    smd_orders = (tr['avg_orders_per_cust'] - co['avg_orders_per_cust']) / np.sqrt((tr['std_orders']**2 + co['std_orders']**2) / 2.0) if (tr['std_orders']**2 + co['std_orders']**2) > 0 else 0.0
    
    diag_df = pd.DataFrame([
        {
            'Metric': 'Active Customer Count',
            'Treated_Cohort': f"{tr['cust_cnt']:,}",
            'Control_Cohort': f"{co['cust_cnt']:,}",
            'Standardized_Difference_SMD': 'N/A'
        },
        {
            'Metric': 'Pre-Period Weekly Spend / Customer ($)',
            'Treated_Cohort': f"${tr['avg_weekly_rev']:.2f}",
            'Control_Cohort': f"${co['avg_weekly_rev']:.2f}",
            'Standardized_Difference_SMD': f"{smd_rev:.4f}"
        },
        {
            'Metric': 'Pre-Period Orders / Customer',
            'Treated_Cohort': f"{tr['avg_orders_per_cust']:.2f}",
            'Control_Cohort': f"{co['avg_orders_per_cust']:.2f}",
            'Standardized_Difference_SMD': f"{smd_orders:.4f}"
        },
        {
            'Metric': 'Pre-Period Average Order Value (AOV) ($)',
            'Treated_Cohort': f"${tr['aov']:.2f}",
            'Control_Cohort': f"${co['aov']:.2f}",
            'Standardized_Difference_SMD': 'N/A'
        }
    ])
    
    conn.execute("CREATE OR REPLACE TABLE causal_treatment_control_diagnostics AS SELECT * FROM diag_df")
    conn.close()
    
    print("\n--- Pre-Intervention Baseline Balance Check ---")
    print(diag_df.to_string(index=False))
    print("\nSimulated intervention dataset saved to DuckDB table 'causal_experiment_data'.")
    
    return panel_df

if __name__ == "__main__":
    simulate_causal_intervention()
