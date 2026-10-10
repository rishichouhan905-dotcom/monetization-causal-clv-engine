# Out-of-Sample Holdout Validation Report 🎯

**Estimation Method**: `Maximum Likelihood Estimation (MLE)`  
**Calibration Window**: `2009-12-01` to `2010-12-01`  
**Holdout Window**: `2010-12-01` to `2011-12-09` (373 days)  
**Frequency-Monetary Pearson Correlation**: `r = 0.0605` ($p = 1.2697e-03$)

---

## 📊 Summary Performance Comparison: BG/NBD vs. Naive Baselines

| Metric | BG/NBD Probabilistic Model | Naive 1 ($F/T \cdot days$) | Naive 2 ($F \cdot days/365$) | Naive 3 (Empirical $F=0$) | Outperformance vs Naive 1 | Outperformance vs Naive 2 | Outperformance vs Naive 3 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Actual Purchases** | **14,365** | **14,365** | **14,365** | **14,365** | — | — | — |
| **Total Predicted Purchases** | **21,107** (+46.93%) | **21,886** (+52.36%) | **14,482** (+0.81%) | **23,202** (+61.52%) | — | — | — |
| **Purchase MAE** (purchases/cust) | **2.9239** | **3.2996** | **2.2743** | **3.3688** | **+11.38%** | **-28.56%** | **+13.20%** |
| **Purchase RMSE** | **4.6012** | **6.9491** | **4.6943** | **6.9286** | **+33.79%** | **+1.98%** | **+33.59%** |
| **Total Actual Revenue** | **$7,426,035.15** | **$7,426,035.15** | **$7,426,035.15** | **$7,426,035.15** | — | — | — |
| **Total Predicted Revenue** | **$10,438,731.24** (+40.57%) | **$11,404,885.25** (+53.58%) | **$7,659,999.86** (+3.15%) | **$11,872,135.99** (+59.87%) | — | — | — |
| **Revenue MAE** ($/cust) | **$1,440.63** | **$1,640.21** | **$1,118.13** | **$1,670.59** | **+12.17%** | **-28.84%** | **+13.77%** |
| **Revenue RMSE** ($/cust) | **$3,843.78** | **$5,381.34** | **$4,410.16** | **$5,383.87** | **+28.57%** | **+12.84%** | **+28.61%** |
| **Pearson Correlation ($r$)** | **0.8354** | **0.7234** | **0.8234** | **0.7251** | — | — | — |

---

## 🏆 Ranking & Performance Metrics: BG/NBD vs. Carry-Forward Baseline vs. Tenure

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

---

## 📈 Customer Decile Calibration Breakdown

| Decile | Customer Count | Avg Actual Purchases | Avg Predicted Purchases (BG/NBD) | Avg Actual Spend ($) | Avg Predicted Spend ($) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D01** | 427 | 0.585 | 0.827 | $242.24 | $292.82 |
| **D02** | 427 | 0.892 | 1.322 | $260.90 | $479.42 |
| **D03** | 426 | 1.099 | 1.952 | $410.36 | $819.90 |
| **D04** | 427 | 1.412 | 2.522 | $538.63 | $1,169.07 |
| **D05** | 426 | 1.678 | 3.083 | $659.98 | $1,267.05 |
| **D06** | 427 | 1.986 | 3.771 | $706.17 | $1,588.96 |
| **D07** | 426 | 2.547 | 4.577 | $1,208.86 | $1,962.52 |
| **D08** | 427 | 3.602 | 5.775 | $1,512.72 | $2,627.47 |
| **D09** | 426 | 5.136 | 8.016 | $2,270.17 | $3,723.57 |
| **D10** | 427 | 14.728 | 17.627 | $9,591.82 | $10,534.11 |

---

## 🔄 Rolling-Origin Multi-Cutoff Evaluation

| Origin Cutoff | Calibration Customers ($N$) | Holdout Window | Holdout Days | Actual vs Predicted Purchases | Volume Error | Actual vs Predicted Revenue | Revenue Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-12-01** | **4,266** | 2010-12-01 $\to$ 2011-12-09 | 373 | 14,365 vs 21,107 | +46.93% | $7,426,035.15 vs $10,438,731.24 | +40.57% |
| **2011-03-01** | **4,537** | 2011-03-01 $\to$ 2011-12-09 | 283 | 11,787 vs 12,722 | +7.93% | $5,878,377.60 vs $6,493,844.08 | +10.47% |
| **2011-06-01** | **4,933** | 2011-06-01 $\to$ 2011-12-09 | 191 | 9,092 vs 8,597 | -5.44% | $4,581,997.30 vs $4,322,963.80 | -5.65% |

### Rolling-Origin Subpopulation Breakdown ($F = 0$ vs. $F > 0$)

| Origin Cutoff | Calibration $N$ | Segment | Actual Purchases | Pred Purchases | Actual Revenue | Pred Revenue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-12-01** | 4,266 | **$F = 0$** (1,431 cust) | 1,316 | 2,964 (+125.2%) | $455,278.81 | $1,083,463.78 (+138.0%) |
| **2010-12-01** | 4,266 | **$F > 0$** (2,835 cust) | 13,049 | 18,143 (+39.0%) | $6,970,756.34 | $9,355,267.47 (+34.2%) |
| **2011-03-01** | 4,537 | **$F = 0$** (1,489 cust) | 1,023 | 1,376 (+34.5%) | $321,606.51 | $488,747.10 (+52.0%) |
| **2011-03-01** | 4,537 | **$F > 0$** (3,048 cust) | 10,764 | 11,345 (+5.4%) | $5,556,771.09 | $6,005,096.98 (+8.1%) |
| **2011-06-01** | 4,933 | **$F = 0$** (1,544 cust) | 777 | 798 (+2.7%) | $248,552.44 | $279,387.68 (+12.4%) |
| **2011-06-01** | 4,933 | **$F > 0$** (3,389 cust) | 8,315 | 7,799 (-6.2%) | $4,333,444.86 | $4,043,576.12 (-6.7%) |

---

## 🎯 Recency-Rule vs. $P(\text{alive})$ Targeting Benchmark (Matched Selection Sizes)

| Origin Cutoff | Rule Inactivity Window | Matched Size ($K$) | Cohort Share | Recency Precision | Recency Recall | $P(\text{alive})$ Precision | $P(\text{alive})$ Recall | $\Delta$ Precision | $\Delta$ Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-12-01** | Inactive $\le 60$ days | **2,426** | 56.9% | 76.50% | 68.09% | 60.10% | 53.48% | **-16.41%** | **-14.60%** |
| **2010-12-01** | Inactive $\le 90$ days | **2,877** | 67.4% | 74.97% | 79.13% | 62.84% | 66.32% | **-12.13%** | **-12.80%** |
| **2010-12-01** | Inactive $\le 180$ days | **3,479** | 81.6% | 70.59% | 90.10% | 65.05% | 83.02% | **-5.55%** | **-7.08%** |
| **2010-12-01** | Inactive $\le 270$ days | **3,950** | 92.6% | 66.99% | 97.07% | 64.89% | 94.02% | **-2.10%** | **-3.04%** |
| **2011-03-01** | Inactive $\le 60$ days | **1,237** | 27.3% | 83.59% | 38.65% | 36.86% | 17.05% | **-46.73%** | **-21.61%** |
| **2011-03-01** | Inactive $\le 90$ days | **1,682** | 37.1% | 80.86% | 50.84% | 41.02% | 25.79% | **-39.83%** | **-25.05%** |
| **2011-03-01** | Inactive $\le 180$ days | **3,349** | 73.8% | 68.89% | 86.24% | 59.48% | 74.47% | **-9.41%** | **-11.78%** |
| **2011-03-01** | Inactive $\le 270$ days | **3,841** | 84.7% | 64.98% | 93.31% | 60.43% | 86.77% | **-4.56%** | **-6.54%** |
| **2011-06-01** | Inactive $\le 60$ days | **1,519** | 30.8% | 80.58% | 47.06% | 26.60% | 15.53% | **-53.98%** | **-31.53%** |
| **2011-06-01** | Inactive $\le 90$ days | **1,959** | 39.7% | 76.67% | 57.75% | 41.25% | 31.06% | **-35.43%** | **-26.68%** |
| **2011-06-01** | Inactive $\le 180$ days | **2,659** | 53.9% | 71.53% | 73.13% | 52.39% | 53.56% | **-19.14%** | **-19.57%** |
| **2011-06-01** | Inactive $\le 270$ days | **3,898** | 79.0% | 61.67% | 92.43% | 55.49% | 83.16% | **-6.18%** | **-9.27%** |

---

## 🔬 Comprehensive Multi-Model Architecture Comparison

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

---

## 💵 Documented Single Monetary Definition & Revenue Discrepancy Explanation

### 1. Root Cause of Previous Revenue Discrepancy
In earlier prototype reports, predicted revenue totals at the three rolling cutoffs were reported as:
- `2010-12-01`: **$9.41M**
- `2011-03-01`: **$5.96M**
- `2011-06-01`: **$3.99M**

Whereas subsequent model evaluations reported:
- `2010-12-01`: **$10.44M**
- `2011-03-01`: **$6.49M**
- `2011-06-01`: **$4.32M**

**Why Did They Differ While Purchases and Revenue MAE Were Identical?**
1. **The $9.41M / $5.96M / $3.99M Figures**: Used the raw, unadjusted calibration average order spend `monetary_value` ($E[Y] \cdot \text{monetary\_value}$) without passing repeat buyers through the Gamma-Gamma conditional spend model.
2. **The $10.44M / $6.49M / $4.32M Figures**: Used the unified expected order spend ($M_i$), where repeat buyers ($F > 0$) are regressed to the population mean via the Gamma-Gamma model ($E[M \mid F, M_{repeat}]$), and single buyers ($F = 0$) receive their observed baseline spend $M_0$.

**The Correct Definition**:
$$\text{Customer-Level Predicted Revenue}_i = \hat{Y}_i \times E[M_i \mid F_i, M_{repeat, i}]$$
Customer-level predicted revenue is strictly the product of expected purchases $\hat{Y}_i$ and expected order value $M_i$. Every table in this report, across all rolling-origin cutoffs and subpopulations, programmatically and strictly uses this unified definition.

### 2. CLV Specification
- **Gross Revenue vs. Net Margin**: CLV is modeled and predicted as **Discounted Gross Revenue**. To derive **Discounted Net Margin**, financial decision engines apply `GROSS_MARGIN_PCT = 0.40` (40% gross margin).
- **Discount Rate**: Evaluated at a monthly continuous discount rate of $d = 0.01$ (1% per month).

---

## ⚠️ Known Limitations & Operational Guidance

1. **One-Time Buyers ($F = 0$) Ranking and Calibration Dynamics**:
   - **Ranking Equivalence to Tenure**: For one-time buyers ($x = 0, t_x = 0$), the BG/NBD conditional expected purchases function simplifies to a strictly monotonic decreasing function of customer tenure $T$. Consequently, **BG/NBD ranks one-time buyers identically to tenure $T$ (newest first, $-T$)**, producing identical Spearman rank correlation ($\rho = 0.2314$) and top-decile capture ($17.63\%$).
   - **Lack of Point Forecast Calibration at Cutoff 1**: While BG/NBD provides meaningful ranking power, its point forecast for one-time buyers is **not calibrated at the first cutoff (`2010-12-01`)**, predicting 2,964 purchases ($2.07$ purchases/customer) against only 1,316 actual holdout purchases ($0.92$ purchases/customer), representing a **$+125.2\%$ forecast over-prediction**. Calibration improves at later origins (`2011-03-01`: $+34.5\%$, `2011-06-01`: $+2.7\%$).
2. **Targeting Rule Guidance: Recency Rules Outperform Raw $P(\text{alive})$**:
   - In BG/NBD, customers with $F = 0$ observed repeat purchases have never encountered a repeat transaction opportunity to drop out, leading lifetimes to set $P(\text{alive}) = 1.0$ identically for all 1,400+ single buyers.
   - Targeting customers strictly by highest $P(\text{alive})$ selects dormant single buyers who made a single order months ago ahead of active repeat buyers. As demonstrated in the matched-size targeting benchmark, simple recency rules (e.g. Inactive $\le 60$ or $\le 90$ days) provide substantially higher precision (e.g., $76.5\% - 83.6\%$ vs. $26.2\% - 60.1\%$) for customer reactivation and retention campaigns.
3. **Seasonality & Stationarity Assumptions**:
   - The BG/NBD model assumes a stationary Poisson transaction process with constant transaction rate $\lambda$. Retail transaction volume in Q4 (holiday peak) exhibits pronounced seasonality not captured by stationary parameters.

---

## 🛡️ Model Identification & Assumption Diagnostics

1. **Frequency-Monetary Independence**: The Pearson correlation between repeat frequency $F$ and repeat average spend $M_{repeat}$ is $r = 0.0605$ ($p = 1.2697e-03$). This low correlation supports the conditional independence assumption of the Gamma-Gamma sub-model.
2. **Strict Time-Based Holdout Split**: All models are fit strictly on calibration data prior to each cutoff and evaluated strictly out-of-sample on holdout transactions through `2011-12-09`.
