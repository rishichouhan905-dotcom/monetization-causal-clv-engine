import os
import duckdb
import pandas as pd
import numpy as np
from scipy import stats

from clv_causal.config import DB_PATH

def compute_ols_cluster_robust(X, y, cluster_ids):
    """
    Fits OLS linear regression and computes Cluster-Robust Standard Errors (CRSE)
    clustered by group/cluster_ids (Liang & Zeger / Arellano formula).
    """
    N, K = X.shape
    # Solve beta = (X^T X)^-1 X^T y
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.pinv(XtX)
    beta = np.dot(XtX_inv, np.dot(X.T, y))
    
    residuals = y - np.dot(X, beta)
    
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    
    # Meat matrix calculation
    meat = np.zeros((K, K))
    for g in unique_clusters:
        mask = (cluster_ids == g)
        X_g = X[mask]
        e_g = residuals[mask].reshape(-1, 1)
        score_g = np.dot(X_g.T, e_g)
        meat += np.dot(score_g, score_g.T)
        
    # Finite sample correction adjustment
    df_correction = (G / max(1, G - 1)) * ((N - 1) / max(1, N - K))
    vcov_crse = df_correction * np.dot(XtX_inv, np.dot(meat, XtX_inv))
    
    se = np.sqrt(np.maximum(0.0, np.diag(vcov_crse)))
    t_stats = np.where(se > 0, beta / se, 0.0)
    
    # p-values using Student's t distribution with G - 1 degrees of freedom
    dof = max(1, G - 1)
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=dof))
    
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se
    
    return beta, se, p_values, ci_lower, ci_upper, dof

class DifferenceInDifferencesEstimator:
    def __init__(self):
        self.results = None

    def estimate_att(self, panel_df=None):
        """
        Priority 3 & Priority 4 Upgrade:
        Estimates DiD ATT using country-week panel structure with Cluster-Robust Standard Errors (CRSE)
        clustered by country unit, accompanied by multi-window pre-trend tests and an event-study relative week decomposition.
        """
        if panel_df is None:
            conn = duckdb.connect(DB_PATH)
            panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
            conn.close()

        print("Aggregating country-week panel trajectory for Difference-in-Differences (DiD)...")
        panel_df['week_date'] = pd.to_datetime(panel_df['week_date'])
        
        # Collapse to Country-Week level preserving unit variance
        country_week = panel_df.groupby(['country', 'is_treated_cohort', 'week_date', 'is_post_period']).agg(
            mean_weekly_spend=('observed_weekly_spend', 'mean'),
            customer_count=('customer_id', 'nunique')
        ).reset_index()
        
        country_week['post_x_treated'] = country_week['is_treated_cohort'] * country_week['is_post_period']
        
        # Design matrix X: [Intercept, is_treated_cohort, is_post_period, post_x_treated]
        N = len(country_week)
        X = np.column_stack([
            np.ones(N),
            country_week['is_treated_cohort'].values,
            country_week['is_post_period'].values,
            country_week['post_x_treated'].values
        ])
        y = country_week['mean_weekly_spend'].values
        clusters = country_week['country'].values
        
        beta, se, pvals, ci_lower, ci_upper, dof = compute_ols_cluster_robust(X, y, clusters)
        
        att = beta[3]
        std_err = se[3]
        pval = pvals[3]
        ci_l = ci_lower[3]
        ci_u = ci_upper[3]
        
        # Calculate percentage lift relative to pre-period treated baseline spend
        pre_treated_mean = country_week[(country_week['is_treated_cohort'] == 1) & (country_week['is_post_period'] == 0)]['mean_weekly_spend'].mean()
        pct_lift = (att / pre_treated_mean * 100.0) if pre_treated_mean > 0 else 0.0
        
        print("\n=================== Difference-in-Differences (DiD) Results ===================")
        print(f"Average Treatment Effect on Treated (ATT): ${att:.2f} / week ({pct_lift:+.2f}% lift)")
        print(f"Cluster-Robust Standard Error (CRSE):    ${std_err:.2f} (clustered across {len(np.unique(clusters))} country units)")
        print(f"p-value (df={dof}):                      {pval:.4e}")
        print(f"95% Cluster-Robust Confidence Interval:   [${ci_l:.2f}, ${ci_u:.2f}]")
        print("===============================================================================\n")

        # --- Priority 4: Parallel Trends Diagnostics & Interaction Test ---
        pre_df = country_week[country_week['is_post_period'] == 0].copy()
        min_week = pre_df['week_date'].min()
        pre_df['time_trend'] = (pre_df['week_date'] - min_week).dt.days // 7
        pre_df['trend_x_treated'] = pre_df['time_trend'] * pre_df['is_treated_cohort']
        
        N_pre = len(pre_df)
        X_pre = np.column_stack([
            np.ones(N_pre),
            pre_df['is_treated_cohort'].values,
            pre_df['time_trend'].values,
            pre_df['trend_x_treated'].values
        ])
        y_pre = pre_df['mean_weekly_spend'].values
        clusters_pre = pre_df['country'].values
        
        beta_pre, se_pre, pvals_pre, ci_l_pre, ci_u_pre, _ = compute_ols_cluster_robust(X_pre, y_pre, clusters_pre)
        
        trend_att = beta_pre[3]
        trend_se = se_pre[3]
        trend_pval = pvals_pre[3]
        trend_valid = bool(trend_pval > 0.05)
        
        print(f"Pre-Period Parallel Trends Interaction Test: slope_diff=${trend_att:.4f}/week, p-value={trend_pval:.4f}")
        if trend_valid:
            print("[PASS] Pre-period parallel trends test (p > 0.05). Baseline pre-trends are statistically parallel.")
        else:
            print("[NOTE] Differential pre-trend detected (p <= 0.05).")
            
        # --- Event Study Analysis (Relative Week Leads & Lags) ---
        t0_date = pd.to_datetime("2010-06-01")
        country_week['rel_week'] = (country_week['week_date'] - t0_date).dt.days // 7
        
        # Restrict relative weeks to [-10, +10]
        event_df = country_week[(country_week['rel_week'] >= -10) & (country_week['rel_week'] <= 10)].copy()
        rel_weeks = sorted([w for w in event_df['rel_week'].unique() if w != -1]) # -1 is omitted reference week
        
        X_event_list = [np.ones(len(event_df)), event_df['is_treated_cohort'].values]
        event_cols = []
        for w in rel_weeks:
            is_w = (event_df['rel_week'] == w).astype(int).values
            X_event_list.append(is_w) # main week dummy
            inter = (event_df['is_treated_cohort'] * (event_df['rel_week'] == w)).astype(int).values
            X_event_list.append(inter) # interaction dummy
            event_cols.append(w)
            
        X_event = np.column_stack(X_event_list)
        y_event = event_df['mean_weekly_spend'].values
        clusters_event = event_df['country'].values
        
        beta_ev, se_ev, pvals_ev, ci_l_ev, ci_u_ev, _ = compute_ols_cluster_robust(X_event, y_event, clusters_event)
        
        event_results = []
        # Add omitted reference week -1
        event_results.append({'rel_week': -1, 'effect': 0.0, 'se': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'is_post': 0})
        
        for idx, w in enumerate(event_cols):
            param_idx = 2 + idx * 2 + 1
            is_post = 1 if w >= 0 else 0
            event_results.append({
                'rel_week': int(w),
                'effect': float(beta_ev[param_idx]),
                'se': float(se_ev[param_idx]),
                'ci_lower': float(ci_l_ev[param_idx]),
                'ci_upper': float(ci_u_ev[param_idx]),
                'is_post': is_post
            })
            
        event_study_df = pd.DataFrame(event_results).sort_values('rel_week').reset_index(drop=True)
        
        # Save event study table to DuckDB
        conn = duckdb.connect(DB_PATH)
        conn.execute("CREATE OR REPLACE TABLE causal_event_study_results AS SELECT * FROM event_study_df")
        conn.close()

        summary_dict = {
            "att": float(att),
            "pct_lift": float(pct_lift),
            "std_err": float(std_err),
            "pval": float(pval),
            "ci_lower": float(ci_l),
            "ci_upper": float(ci_u),
            "parallel_trends_pval": float(trend_pval),
            "parallel_trends_valid": trend_valid,
            "event_study_df": event_study_df
        }
        
        return summary_dict

def run_did_analysis():
    estimator = DifferenceInDifferencesEstimator()
    res = estimator.estimate_att()
    return res

if __name__ == "__main__":
    run_did_analysis()
