# System Results & Empirical Facts Reference 📊

> **Notice**: This document is generated 100% programmatically by `scripts/build_results_facts.py` directly from analytical data in `results/`, `app_data/`, and `holdout_report.md`. It contains zero hand-typed numbers and strict provenance validation, serving as the single source of empirical truth.

## 1. Dataset & Population Summary

| Metric | Value | Source / Entity |
| :--- | :--- | :--- |
| **Positive Order Lines** | 805,549 | `app_data/fact_orders.parquet` |
| **Return / Cancellation Lines** | 18,744 | `app_data/fact_returns.parquet` |
| **Total Transaction Records** | 824,293 | Orders + Returns |
| **Empirical Date Range** | 2009-12-01 to 2011-12-09 | Full 2-year timeline |
| **Total Customers in Warehouse** | 5,939 | `app_data/dim_customers.parquet` |
| **Scored CLV Customers** | 5,878 | `app_data/pred_clv_summary.parquet` |
| **Mean Model P(alive)** | 88.43% | Fitted BG/NBD dropout parameters |

## 2. Revenue Accounting Reconciliation

| Accounting Model / Table | Gross Revenue ($) | Returns Value ($) | Net Realized Revenue ($) | Reconciliation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Transaction Layer** (`fact_orders` & `fact_returns`) | $17,743,429.18 | $1,095,136.79 | $16,648,292.39 | Base Transaction Truth |
| **Customer Layer** (`dim_customers`) | $17,743,429.16 | $1,095,136.79 | $16,648,292.37 | Balanced Customer Aggregate |
| **Delta (Customer Gross - Returns - Net)** | — | — | **$0.0000** | Exact $0.00 Balance |
| **Delta (Transaction Net vs Customer Net)** | — | — | **$0.0180** | Exact match (delta < $0.02 rounding across 824k lines) |

## 3. Rolling-Origin Multi-Cutoff Validation Results

### 3.1 Multi-Cutoff Aggregate Performance

| Origin Cutoff | Calibration Customers ($N$) | Holdout Window | Holdout Days | Actual vs Predicted Purchases | Volume Error | Actual vs Predicted Revenue | Revenue Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-12-01** | **4,266** | 2010-12-01 $\to$ 2011-12-09 | 373 | 14,365 vs 21,107 | +46.93% | $7,426,035.15 vs $10,438,731.24 | +40.57% |
| **2011-03-01** | **4,537** | 2011-03-01 $\to$ 2011-12-09 | 283 | 11,787 vs 12,722 | +7.93% | $5,878,377.60 vs $6,493,844.08 | +10.47% |
| **2011-06-01** | **4,933** | 2011-06-01 $\to$ 2011-12-09 | 191 | 9,092 vs 8,597 | -5.44% | $4,581,997.30 vs $4,322,963.80 | -5.65% |

### 3.2 Subpopulation Breakdown (F == 0 One-Time vs F > 0 Repeat Buyers)

| Origin Cutoff | Calibration $N$ | Segment | Actual Purchases | Pred Purchases | Actual Revenue | Pred Revenue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-12-01** | 4,266 | **$F = 0$** (1,431 cust) | 1,316 | 2,964 (+125.2%) | $455,278.81 | $1,083,463.78 (+138.0%) |
| **2010-12-01** | 4,266 | **$F > 0$** (2,835 cust) | 13,049 | 18,143 (+39.0%) | $6,970,756.34 | $9,355,267.47 (+34.2%) |
| **2011-03-01** | 4,537 | **$F = 0$** (1,489 cust) | 1,023 | 1,376 (+34.5%) | $321,606.51 | $488,747.10 (+52.0%) |
| **2011-03-01** | 4,537 | **$F > 0$** (3,048 cust) | 10,764 | 11,345 (+5.4%) | $5,556,771.09 | $6,005,096.98 (+8.1%) |
| **2011-06-01** | 4,933 | **$F = 0$** (1,544 cust) | 777 | 798 (+2.7%) | $248,552.44 | $279,387.68 (+12.4%) |
| **2011-06-01** | 4,933 | **$F > 0$** (3,389 cust) | 8,315 | 7,799 (-6.2%) | $4,333,444.86 | $4,043,576.12 (-6.7%) |

### 3.3 Multi-Model Architecture Comparison (BG/NBD vs. MBG/NBD vs. Pareto/NBD)

| Cutoff Origin | Model Architecture | Params | Purchases MAE (All / F=0 / F>0) | Revenue MAE (All / F=0 / F>0) | Total Purchases (Act vs Pred) | Total Revenue (Act vs Pred) | Share $P(\text{alive}) > 0.8$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **2010-12-01** | **BG/NBD** | MLE | 2.92 / 1.75 / 3.52 | $1,441 / $670 / $1,830 | 14,365 vs 21,107 | $7,426,035.15 vs $10,438,731.24 | 97.3% |
| **2010-12-01** | **MBG/NBD** | MLE | 2.95 / 1.75 / 3.55 | $1,459 / $671 / $1,856 | 14,365 vs 21,178 | $7,426,035.15 vs $10,451,437.01 | 98.4% |
| **2010-12-01** | **Pareto/NBD** | MLE | 3.02 / 1.71 / 3.68 | $1,475 / $654 / $1,889 | 14,365 vs 21,829 | $7,426,035.15 vs $10,806,111.01 | 97.4% |
| **2011-03-01** | **BG/NBD** | MLE | 1.74 / 0.97 / 2.12 | $869 / $344 / $1,125 | 11,787 vs 12,722 | $5,878,377.60 vs $6,493,844.08 | 85.2% |
| **2011-03-01** | **MBG/NBD** | MLE | 1.76 / 0.96 / 2.14 | $879 / $340 / $1,143 | 11,787 vs 12,743 | $5,878,377.60 vs $6,488,788.85 | 88.2% |
| **2011-03-01** | **Pareto/NBD** | MLE | 1.75 / 0.95 / 2.15 | $877 / $335 / $1,141 | 11,787 vs 12,638 | $5,878,377.60 vs $6,458,513.10 | 77.1% |
| **2011-06-01** | **BG/NBD** | MLE | 1.21 / 0.68 / 1.45 | $598 / $229 / $766 | 9,092 vs 8,597 | $4,581,997.30 vs $4,322,963.80 | 77.6% |
| **2011-06-01** | **MBG/NBD** | MLE | 1.21 / 0.66 / 1.46 | $602 / $224 / $774 | 9,092 vs 8,585 | $4,581,997.30 vs $4,311,278.27 | 68.7% |
| **2011-06-01** | **Pareto/NBD** | MLE | 1.21 / 0.65 / 1.46 | $602 / $220 / $776 | 9,092 vs 8,422 | $4,581,997.30 vs $4,234,523.06 | 60.2% |

## 4. Model vs. Naive & Tenure Baselines Comparison

| Cohort | Metric | BG/NBD Model | Carry-Forward Baseline ($F \cdot days/365$) | Delta / Outperformance |
| :--- | :--- | :--- | :--- | :--- |
| **All Customers ($N=4,266$)** | **Purchases MAE** | **2.9239** | **2.2743** | **+0.6496** |
| **All Customers ($N=4,266$)** | **Revenue MAE** | **$1,440.63** | **$1,118.13** | **+$322.51** |
| **All Customers ($N=4,266$)** | **Spearman Rank Corr ($\rho$)** | **0.5673** | **0.5804** | **-0.0131** |
| **All Customers ($N=4,266$)** | **Top Decile Capture** | **43.78%** | **43.22%** | **+0.56%** |
| **$F == 0$ Customers ($N=1,431$)** | **Purchases MAE** | **1.7484** | **0.9196** | **+0.8287** |
| **$F == 0$ Customers ($N=1,431$)** | **Revenue MAE** | **$669.60** | **$318.15** | **+$351.44** |
| **$F == 0$ Customers ($N=1,431$)** | **Spearman Rank Corr ($\rho$)** | **0.2314** | **0.0000** | **+0.2314** |
| **$F == 0$ Customers ($N=1,431$)** | **Top Decile Capture** | **17.63%** | **10.56%** | **+7.07%** |
| **$F == 0$ Tenure Baseline ($-T$)** | **Spearman Rank Corr ($\rho$)** | **0.2314** | **0.0000** | **+0.2314** (Identical to BG/NBD) |
| **$F == 0$ Tenure Baseline ($-T$)** | **Top Decile Capture** | **17.63%** | **10.56%** | **+7.07%** (Identical to BG/NBD) |
| **$F > 0$ Customers ($N=2,835$)** | **Purchases MAE** | **3.5173** | **2.9581** | **+0.5592** |
| **$F > 0$ Customers ($N=2,835$)** | **Revenue MAE** | **$1,829.82** | **$1,521.92** | **+$307.90** |
| **$F > 0$ Customers ($N=2,835$)** | **Spearman Rank Corr ($\rho$)** | **0.5394** | **0.5488** | **-0.0094** |
| **$F > 0$ Customers ($N=2,835$)** | **Top Decile Capture** | **40.58%** | **39.31%** | **+1.26%** |

## 5. P(alive) Customer Decile Calibration Breakdown

| Decile Rank | Customer Count | Avg Actual Purchases | Avg Predicted Purchases | Avg Actual Spend ($) | Avg Predicted Spend ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **D01** | 427 | 0.585 | 0.827 | $242.24 | $292.42 |
| **D02** | 427 | 0.892 | 1.322 | $260.90 | $479.82 |
| **D03** | 426 | 1.103 | 1.952 | $410.87 | $806.95 |
| **D04** | 427 | 1.407 | 2.522 | $538.11 | $1,181.99 |
| **D05** | 426 | 1.678 | 3.083 | $659.98 | $1,267.05 |
| **D06** | 427 | 1.986 | 3.771 | $706.17 | $1,588.96 |
| **D07** | 426 | 2.547 | 4.577 | $1,208.86 | $1,962.52 |
| **D08** | 427 | 3.602 | 5.775 | $1,512.72 | $2,627.47 |
| **D09** | 426 | 5.136 | 8.016 | $2,270.17 | $3,723.57 |
| **D10** | 427 | 14.728 | 17.627 | $9,591.82 | $10,534.11 |

## 6. Causal Policy Treatment Effects (Two-Way Fixed Effects DiD)

### 6.1 Baseline Scenario (20% Top-Spend Churn)

| Cohort | Outcome Metric | True Realized ATT | TWFE Estimate | Standard Error | 95% Confidence Interval | p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (All Treated)** | `weekly_orders` | +0.0057 | +0.0087 | 0.0031 | [+0.0027, +0.0147] | 4.7707e-03 |
| **Overall (All Treated)** | `weekly_spend_winsor` | +4.4831 | +5.5032 | 1.7218 | [+2.1286, +8.8777] | 1.3922e-03 |
| **Overall (All Treated)** | `active_last_90d` | +0.0304 | +0.0370 | 0.0093 | [+0.0188, +0.0551] | 6.5098e-05 |
| **Q1-Q4 Pooled** | `weekly_orders` | +0.0067 | +0.0075 | 0.0020 | [+0.0036, +0.0114] | 1.5517e-04 |
| **Q1-Q4 Pooled** | `weekly_spend_winsor` | +2.9217 | +3.1067 | 0.7079 | [+1.7192, +4.4942] | 1.1419e-05 |
| **Q1-Q4 Pooled** | `active_last_90d` | +0.0388 | +0.0499 | 0.0108 | [+0.0288, +0.0710] | 3.4788e-06 |
| **Quintile 5 (Top Spend)** | `weekly_orders` | +0.0016 | +0.0131 | 0.0117 | [-0.0099, +0.0362] | 2.6283e-01 |
| **Quintile 5 (Top Spend)** | `weekly_spend_winsor` | +10.7208 | +13.5010 | 7.1812 | [-0.5740, +27.5761] | 6.0104e-02 |
| **Quintile 5 (Top Spend)** | `active_last_90d` | -0.0032 | -0.0025 | 0.0183 | [-0.0384, +0.0334] | 8.9061e-01 |

### 6.2 Hidden Degradation Scenario (40% Top-Spend Churn)

| Cohort | Outcome Metric | True Realized ATT | TWFE Estimate | Standard Error | 95% Confidence Interval | p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (All Treated)** | `weekly_orders` | +0.0012 | +0.0037 | 0.0031 | [-0.0025, +0.0098] | 2.4362e-01 |
| **Overall (All Treated)** | `weekly_spend_winsor` | +1.1324 | +1.6588 | 1.7191 | [-1.7107, +5.0282] | 3.3460e-01 |
| **Overall (All Treated)** | `active_last_90d` | +0.0197 | +0.0254 | 0.0092 | [+0.0073, +0.0435] | 5.8602e-03 |
| **Q1-Q4 Pooled** | `weekly_orders` | +0.0065 | +0.0073 | 0.0020 | [+0.0034, +0.0112] | 2.5816e-04 |
| **Q1-Q4 Pooled** | `weekly_spend_winsor` | +2.9483 | +3.0537 | 0.7167 | [+1.6490, +4.4584] | 2.0371e-05 |
| **Q1-Q4 Pooled** | `active_last_90d` | +0.0341 | +0.0452 | 0.0107 | [+0.0242, +0.0661] | 2.4221e-05 |
| **Quintile 5 (Top Spend)** | `weekly_orders` | -0.0197 | -0.0081 | 0.0120 | [-0.0316, +0.0155] | 5.0225e-01 |
| **Quintile 5 (Top Spend)** | `weekly_spend_winsor` | -6.1220 | -3.2609 | 7.1478 | [-17.2705, +10.7487] | 6.4824e-01 |
| **Quintile 5 (Top Spend)** | `active_last_90d` | -0.0379 | -0.0372 | 0.0187 | [-0.0738, -0.0005] | 4.6746e-02 |

## 7. Monte Carlo Simulation Summary (50 Replications per Scenario)

| Scenario | Cohort | Outcome | Replications | Mean True ATT | Mean Est ATT | Mean Bias | RMSE | 95% Coverage | 5% Rejection Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline** | Overall (All Treated) | `active_last_90d` | 50 | +0.0278 | +0.0275 | -0.0004 | 0.0081 | **100%** | 88% |
| **baseline** | Overall (All Treated) | `weekly_orders` | 50 | +0.0047 | +0.0043 | -0.0003 | 0.0029 | **98%** | 30% |
| **baseline** | Overall (All Treated) | `weekly_spend_winsor` | 50 | +3.6161 | +3.5768 | -0.0393 | 1.8396 | **96%** | 52% |
| **baseline** | Q1-Q4 Pooled | `active_last_90d` | 50 | +0.0344 | +0.0347 | +0.0003 | 0.0088 | **98%** | 98% |
| **baseline** | Q1-Q4 Pooled | `weekly_orders` | 50 | +0.0060 | +0.0056 | -0.0004 | 0.0019 | **94%** | 80% |
| **baseline** | Q1-Q4 Pooled | `weekly_spend_winsor` | 50 | +2.7883 | +2.7227 | -0.0656 | 0.6575 | **94%** | 96% |
| **baseline** | Quintile 5 (Top Spend) | `active_last_90d` | 50 | +0.0017 | +0.0026 | +0.0010 | 0.0178 | **92%** | 0% |
| **baseline** | Quintile 5 (Top Spend) | `weekly_orders` | 50 | -0.0007 | -0.0000 | +0.0007 | 0.0112 | **96%** | 4% |
| **baseline** | Quintile 5 (Top Spend) | `weekly_spend_winsor` | 50 | +6.9234 | +6.4937 | -0.4297 | 7.4915 | **96%** | 16% |
| **hidden_degradation** | Overall (All Treated) | `active_last_90d` | 50 | +0.0188 | +0.0164 | -0.0024 | 0.0092 | **96%** | 50% |
| **hidden_degradation** | Overall (All Treated) | `weekly_orders` | 50 | -0.0005 | -0.0015 | -0.0011 | 0.0031 | **94%** | 4% |
| **hidden_degradation** | Overall (All Treated) | `weekly_spend_winsor` | 50 | +0.7203 | +0.2734 | -0.4468 | 1.4790 | **100%** | 4% |
| **hidden_degradation** | Q1-Q4 Pooled | `active_last_90d` | 50 | +0.0339 | +0.0332 | -0.0007 | 0.0119 | **94%** | 82% |
| **hidden_degradation** | Q1-Q4 Pooled | `weekly_orders` | 50 | +0.0059 | +0.0055 | -0.0004 | 0.0022 | **94%** | 84% |
| **hidden_degradation** | Q1-Q4 Pooled | `weekly_spend_winsor` | 50 | +2.7352 | +2.4591 | -0.2760 | 0.7544 | **96%** | 96% |
| **hidden_degradation** | Quintile 5 (Top Spend) | `active_last_90d` | 50 | -0.0416 | -0.0407 | +0.0009 | 0.0160 | **96%** | 70% |
| **hidden_degradation** | Quintile 5 (Top Spend) | `weekly_orders` | 50 | -0.0261 | -0.0256 | +0.0006 | 0.0102 | **98%** | 48% |
| **hidden_degradation** | Quintile 5 (Top Spend) | `weekly_spend_winsor` | 50 | -7.3292 | -7.1614 | +0.1678 | 5.6550 | **100%** | 14% |

## 8. Confounded-Assignment Diagnostic & Pre-Trend Falsification (Real Panel)

To test the identification framework against parallel-trend violations, the engine executes an **extreme near-deterministic selection stress test** on the 390,939-row balanced customer panel. Under this stress test, treatment is deliberately assigned to units exhibiting the highest pre-period spend velocity, violating the unconfoundedness assumption ($Y(0), Y(1) \not\perp D \mid X$).

### Exact Assignment Rule:
1. **Normalized Pre-Period Spend Velocity**:
   $$\text{spend\_growth}_i = \frac{\text{late\_spend}_i - \text{early\_spend}_i}{\text{late\_spend}_i + \text{early\_spend}_i + 10.0}$$
   where $\text{early\_spend}_i$ is customer $i$'s total spend from first order to their pre-period midpoint ($t_{\text{mid}, i} = t_{\text{first}, i} + \frac{t_0 - t_{\text{first}, i}}{2}$), and $\text{late\_spend}_i$ is spend from $t_{\text{mid}, i}$ to $t_0$ (`2011-06-01`).
2. **Near-Deterministic Selection**:
   $$\text{threshold} = \text{quantile}_{1 - \text{share}}(\text{growth\_rank}_i) = 60^{\text{th}}\text{ percentile}$$
   $$\text{is\_treated}_i = \mathbb{I}(\text{growth\_rank}_i \ge \text{threshold})$$
   Customers in the top 40% of spend acceleration are assigned to treatment; flat and decelerating customers form the control donor pool.

### Empirical Comparison: Baseline Random Assignment vs. Selection Stress Test

Both evaluations are generated on the real empirical panel using the same covariance-based leads-and-lags estimator (`run_event_study` with degrees of freedom $\text{df} = K - 1 = 9$ across leads $k \in [-10, -2]$ relative to $t_0$):

| Identification Diagnostic / Estimator Metric | Baseline Stratified Random Assignment | Confounded Assignment (Selection Stress Test) | Statistical Mechanism / Impact |
| :--- | :---: | :---: | :--- |
| **Provenance: Source Script** | `scripts/run_confounded_benchmark.py` | `scripts/run_confounded_benchmark.py` | Verified real-panel benchmark scripts |
| **Provenance: Balanced Panel Rows** | 390,939 rows | 390,939 rows | Exact match to empirical warehouse panel |
| **Provenance: Code Version** | `79f4e2e35d` | `79f4e2e35d` | Git commit tracking |
| **Pre-Trend Joint Wald Test Statistic ($\chi^2$)** | $\chi^2 = 4.13$ (df=9) | **$\chi^2 = 60.63$** (df=9) | Consistent covariance-based Wald test |
| **Pre-Trend Wald Test $p$-value** | $p = 0.9025 \ge 0.05$ | **$p = 1.0118e-09 < 0.001$** | **Falsification triggered**: correctly rejects parallel trends |
| **True Ground-Truth ATT ($/cust/wk)** | $+4.4831 | $+4.7670 | Known counterfactual policy lift |
| **TWFE DiD Estimate ($/cust/wk)** | $+5.5032 (SE=$1.7218) | **$+14.0773** (SE=$1.9602) | Upward bias absorbing differential trajectory |
| **Single-Draw Error / TWFE Bias ($/cust/wk)** | **Single-draw error: $+1.0200** (Monte Carlo bias: **$-0.0393**, RMSE=$1.8396, **96% coverage** over 50 reps) | **Confounded bias: $+9.3103** | **236.8x higher** than Monte Carlo bias |
| **Synthetic Control ATT Estimate ($/cust/wk)** | $+6.7308 | **$+50.6556** | Donor weights extrapolate upward to match treated growth |
| **Synthetic Control Pre-RMSPE ($)** | $5.2480 | **$36.5875** | Pre-intervention donor fit breaks down (7.0x error) |
| **Treated Mean Spend Growth** | -0.5032 | -0.0169 | Selected accelerating spenders |
| **Control Mean Spend Growth** | -0.5095 | -0.9121 | Decelerating spenders concentrated in control |
| **Pre-Period Spend Growth Gap** | **+0.0063** (balanced across quintiles) | **+0.8952** | Extreme differential selection violating unconfoundedness |

## 9. Treatment Effect Heterogeneity across Pre-Intervention Spend Deciles

### 9.1 Baseline Scenario Deciles

| Decile | Customers | Treated | True Realized ATT ($) | Estimated DiD ATT ($) | Std Error | p-value | Bonferroni p-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D01** | 494 | 197 | +0.46 | +0.26 | 0.75 | 0.7244 | 1.0000 |
| **D02** | 493 | 198 | +1.05 | +1.16 | 0.87 | 0.1821 | 1.0000 |
| **D03** | 493 | 194 | +1.16 | +0.52 | 1.03 | 0.6088 | 1.0000 |
| **D04** | 493 | 199 | +1.63 | +0.75 | 1.50 | 0.6171 | 1.0000 |
| **D05** | 494 | 195 | +3.31 | +5.13 | 1.88 | 0.0065 | 0.0649 |
| **D06** | 493 | 201 | +3.70 | +1.57 | 2.08 | 0.4495 | 1.0000 |
| **D07** | 493 | 201 | +4.33 | +7.23 | 2.77 | 0.0090 | 0.0901 |
| **D08** | 493 | 194 | +7.80 | +7.43 | 2.89 | 0.0103 | 0.1029 |
| **D09** | 493 | 198 | +5.53 | +9.29 | 3.96 | 0.0188 | 0.1883 |
| **D10** | 494 | 196 | +15.96 | +17.16 | 13.67 | 0.2095 | 1.0000 |

### 9.2 Hidden Degradation Scenario Deciles (Q5 Divergence)

| Decile | Customers | Treated | True Realized ATT ($) | Estimated DiD ATT ($) | Std Error | p-value | Bonferroni p-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D01** | 494 | 197 | +0.57 | +0.39 | 0.75 | 0.6089 | 1.0000 |
| **D02** | 493 | 198 | +0.91 | +1.07 | 0.88 | 0.2259 | 1.0000 |
| **D03** | 493 | 194 | +1.22 | +0.47 | 1.01 | 0.6461 | 1.0000 |
| **D04** | 493 | 199 | +1.66 | +0.80 | 1.52 | 0.5974 | 1.0000 |
| **D05** | 494 | 195 | +3.07 | +4.95 | 1.91 | 0.0095 | 0.0946 |
| **D06** | 493 | 201 | +3.63 | +1.54 | 2.05 | 0.4536 | 1.0000 |
| **D07** | 493 | 201 | +5.92 | +8.52 | 2.84 | 0.0027 | 0.0273 |
| **D08** | 493 | 194 | +6.70 | +6.00 | 2.93 | 0.0404 | 0.4041 |
| **D09** | 493 | 198 | -0.62 | +3.49 | 3.95 | 0.3761 | 1.0000 |
| **D10** | 494 | 196 | -11.83 | -10.54 | 13.59 | 0.4382 | 1.0000 |

## 10. Automated Test Suite & CI Execution Summary

- **Fast CI Test Suite**: **13 passed tests** (unmarked, no warehouse required, executed in under 7 seconds).
- **Warehouse & Full-Data Suite**: **27 tests** (marked `@pytest.mark.slow` or `@pytest.mark.fulldata`).
- **Total Repository Tests**: **40 collected tests** across all units and integration suites.

### Fast CI Tests Running in GitHub Actions Workflow:
- `tests/test_causal_estimators.py::test_q5_policy_simulator_formula`
- `tests/test_causal_estimators.py::test_policy_simulator_zero_delta_and_monotonicity`
- `tests/test_decision.py::test_decision_framework`
- `tests/test_demo_mode_headless.py::test_demo_mode_all_pages_and_tabs_render_without_exception`
- `tests/test_demo_mode_headless.py::test_causal_impact_pre_trend_wald_matches_csv_and_scipy`
- `tests/test_fast_synthetic_causal_and_clv.py::test_synthetic_simulator_panel_structure_and_realized_effects`
- `tests/test_fast_synthetic_causal_and_clv.py::test_synthetic_twfe_solver_vs_panel_ols`
- `tests/test_fast_synthetic_causal_and_clv.py::test_synthetic_wald_pre_trend_random_vs_confounded`
- `tests/test_fast_synthetic_causal_and_clv.py::test_synthetic_quadrature_vs_closed_form`
- `tests/test_formatters.py::test_fmt_usd`
- `tests/test_formatters.py::test_fmt_pp`
- `tests/test_formatters.py::test_fmt_num`
- `tests/test_formatters.py::test_fmt_ci`

---
*End of Results Facts Reference.*