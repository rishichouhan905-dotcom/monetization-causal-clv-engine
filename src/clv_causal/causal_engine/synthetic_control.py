import os
import duckdb
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from clv_causal.config import DB_PATH

class SyntheticControlEstimator:
    """
    Priority 5 Upgrade: Multi-Donor Synthetic Control Estimator with Placebo Permutation Tests.
    Constructs optimal constrained non-negative weights W (summing to 1) across a pool of top donor countries
    to match pre-treatment outcome trajectories, then evaluates post-treatment counterfactuals and in-space placebos.
    """
    def __init__(self):
        self.weights = None
        self.donor_countries = None
        self.treated_country = None

    def fit_predict(self, panel_df=None, target_treated_country="Germany", top_n_donors=7):
        if panel_df is None:
            conn = duckdb.connect(DB_PATH)
            panel_df = conn.execute("SELECT * FROM causal_experiment_data").df()
            conn.close()

        panel_df['week_date'] = pd.to_datetime(panel_df['week_date'])
        
        # Rank countries by total spending volume
        country_totals = panel_df.groupby('country')['observed_weekly_spend'].sum().sort_values(ascending=False)
        top_countries = [c for c in country_totals.index if c != 'United Kingdom']
        
        if target_treated_country in top_countries:
            treated_unit = target_treated_country
        else:
            treated_unit = top_countries[0]
            
        donors = [c for c in top_countries if c != treated_unit][:top_n_donors]
        
        # Aggregate weekly spend per country, filling missing weeks with 0.0
        country_weekly = panel_df[panel_df['country'].isin([treated_unit] + donors)].groupby(
            ['country', 'week_date', 'is_post_period']
        )['observed_weekly_spend'].mean().unstack(level=0).fillna(0.0).reset_index()
        
        country_weekly['week_date'] = pd.to_datetime(country_weekly['week_date'])
        country_weekly = country_weekly.sort_values('week_date')

        self.treated_country = treated_unit
        self.donor_countries = donors

        pre_mask = country_weekly['is_post_period'] == 0
        post_mask = country_weekly['is_post_period'] == 1

        y_pre_treated = country_weekly.loc[pre_mask, treated_unit].values
        X_pre_donors = country_weekly.loc[pre_mask, donors].values # (n_pre, n_donors)

        N_donors = len(donors)
        init_weights = np.full(N_donors, 1.0 / N_donors)
        bounds = [(0.0, 1.0) for _ in range(N_donors)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        def loss_func(w):
            pred = np.dot(X_pre_donors, w)
            return np.mean((y_pre_treated - pred) ** 2)

        res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = res.x
        self.weights = w_opt

        # Predict counterfactual trajectory for full timeline
        X_full_donors = country_weekly[donors].values
        synthetic_series = np.dot(X_full_donors, w_opt)
        
        country_weekly['Treated'] = country_weekly[treated_unit]
        country_weekly['Synthetic_Control'] = synthetic_series
        country_weekly['Causal_Gap'] = country_weekly['Treated'] - country_weekly['Synthetic_Control']

        pre_rmse = float(np.sqrt(np.mean((country_weekly.loc[pre_mask, 'Treated'] - country_weekly.loc[pre_mask, 'Synthetic_Control']) ** 2)))
        post_rmse = float(np.sqrt(np.mean((country_weekly.loc[post_mask, 'Treated'] - country_weekly.loc[post_mask, 'Synthetic_Control']) ** 2)))
        att_synth = float(country_weekly.loc[post_mask, 'Causal_Gap'].mean())

        weights_df = pd.DataFrame({
            'donor_country': donors,
            'weight': np.round(w_opt, 4)
        }).sort_values('weight', ascending=False).reset_index(drop=True)

        print("\n=================== Multi-Donor Synthetic Control Results ===================")
        print(f"Target Treated Country: {treated_unit}")
        print(f"Donor Pool Size:        {N_donors} countries ({', '.join(donors)})")
        print(f"Pre-Treatment RMSE:     ${pre_rmse:.2f}")
        print(f"Post-Treatment ATT:     ${att_synth:.2f} / week")
        print("\nOptimal Donor Weights:")
        for idx, row in weights_df.iterrows():
            print(f"  - {row['donor_country']:<15}: {row['weight']*100:.1f}%")
        print("=============================================================================\n")

        # --- In-Space Placebo Permutation Tests ---
        print("Running In-Space Placebo Permutation Tests across Donor Pool...")
        placebo_results = []
        treated_rmse_ratio = post_rmse / max(1e-5, pre_rmse)
        
        for p_idx, p_country in enumerate(donors):
            p_donors = [c for c in donors if c != p_country] + [treated_unit]
            y_pre_p = country_weekly.loc[pre_mask, p_country].values
            X_pre_p = country_weekly.loc[pre_mask, p_donors].values
            
            init_w_p = np.full(len(p_donors), 1.0 / len(p_donors))
            b_p = [(0.0, 1.0) for _ in range(len(p_donors))]
            c_p = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            
            res_p = minimize(lambda w: np.mean((y_pre_p - np.dot(X_pre_p, w))**2), init_w_p, method='SLSQP', bounds=b_p, constraints=c_p)
            w_p = res_p.x
            
            synth_p = np.dot(country_weekly[p_donors].values, w_p)
            gap_p = country_weekly[p_country].values - synth_p
            
            pre_rmse_p = float(np.sqrt(np.mean(gap_p[pre_mask] ** 2)))
            post_rmse_p = float(np.sqrt(np.mean(gap_p[post_mask] ** 2)))
            ratio_p = post_rmse_p / max(1e-5, pre_rmse_p)
            
            for w_i, date_i, g_i in zip(country_weekly['week_date'], country_weekly['week_date'], gap_p):
                placebo_results.append({
                    'placebo_country': p_country,
                    'week_date': date_i,
                    'causal_gap': float(g_i),
                    'pre_rmse': pre_rmse_p,
                    'post_rmse': post_rmse_p,
                    'rmse_ratio': ratio_p
                })
                
        placebo_df = pd.DataFrame(placebo_results)
        
        # Calculate Permutation p-value
        placebo_ratios = placebo_df.groupby('placebo_country')['rmse_ratio'].first().values
        perm_pval = float((np.sum(placebo_ratios >= treated_rmse_ratio) + 1) / (len(placebo_ratios) + 1))
        print(f"Synthetic Control Placebo Permutation p-value: {perm_pval:.4f}")

        # Save synthetic control outputs to DuckDB
        conn = duckdb.connect(DB_PATH)
        conn.execute("CREATE OR REPLACE TABLE synthetic_control_results AS SELECT * FROM country_weekly")
        conn.execute("CREATE OR REPLACE TABLE synthetic_control_weights AS SELECT * FROM weights_df")
        conn.execute("CREATE OR REPLACE TABLE synthetic_control_placebos AS SELECT * FROM placebo_df")
        conn.close()

        return {
            "treated_country": treated_unit,
            "weights_df": weights_df,
            "pre_rmse": pre_rmse,
            "post_rmse": post_rmse,
            "att_synth": att_synth,
            "perm_pval": perm_pval,
            "trajectory_df": country_weekly,
            "placebo_df": placebo_df
        }

def run_synthetic_control():
    sc = SyntheticControlEstimator()
    return sc.fit_predict()

if __name__ == "__main__":
    run_synthetic_control()
