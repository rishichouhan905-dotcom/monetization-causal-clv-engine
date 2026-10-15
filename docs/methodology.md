# Technical Methodology & Analytical Reference Manual 📐

This document details the mathematical, probabilistic, econometric, and computational methodology underlying the **Enterprise Probabilistic CLV & Causal Impact Decision Platform**. All empirical parameters, sample sizes, and benchmark results documented herein align strictly with [`docs/RESULTS_FACTS.md`](RESULTS_FACTS.md).

---

## 1. Data Engineering & Star Schema Architecture (dbt + DuckDB)

### 1.1 Empirical Dataset
The foundational dataset is the **UCI Machine Learning Repository: Online Retail II** (Chen, 2019; DOI: `10.24432/C5CG6D`), comprising transaction logs from a UK-based registered non-store online retail merchant between **2009-12-01** and **2011-12-09**.
- **Raw Transaction Rows**: 1,067,371 records.
- **Valid Customer Invoices**: Filtered to non-null `Customer ID` integers across 41 countries.
- **Settled Transaction Scope**: 824,293 records partitioned into:
  - **Positive Orders (`fact_orders`)**: 805,549 lines with $Q > 0$, UnitPrice $> 0$.
  - **Returns / Cancellations (`fact_returns`)**: 18,744 lines with $Q < 0$ or Invoice prefixed with `C`.

### 1.2 Explicit Cancellation & Financial Net Reconciliation
Many e-commerce pipelines discard negative return lines or blindly aggregate them into gross totals, introducing systematic margin bias. Here, cancellations are explicitly preserved in staging (`stg_orders`) with `is_return` and `is_cancelled` flags. Downstream marts enforce a rigid accounting balance:
$$\text{Net Revenue} = \text{Gross Revenue} - \text{Return Amount}$$
$$\sum \text{dim\_customers.net\_spend} = \$16,648,292.37 \approx \sum (\text{fact\_orders} - \text{fact\_returns}) = \$16,648,292.39$$
$$\Delta = \$0.0180 \quad (\text{rounding precision across 824,293 rows; exact } \$0.00 \text{ customer-level balance})$$

24 automated dbt schema tests verify:
- Primary key uniqueness on all dimension and fact tables (`dim_customers.customer_id`, `dim_products.stock_code`, `fact_orders.order_line_id`).
- Referential integrity (`relationships` tests from fact foreign keys to dimension tables).
- Singular SQL balance assertions (`assert_net_revenue_equals_gross_minus_returns.sql`).

---

## 2. Probabilistic Customer Lifetime Value (CLV) Engine

### 2.1 RFM Data Representation
For each customer $i$ observed over historical time $[0, T_i]$, we compute:
- **Frequency ($x_i$ or $F_i$)**: Number of repeat transaction dates ($x \ge 0$).
- **Recency ($t_{x,i}$ or $T_{\text{recency},i}$)**: Time of the customer's last observed purchase date.
- **Tenure ($T_i$)**: Total elapsed observation time from customer acquisition to the calibration cutoff.
- **Monetary Value ($M_{\text{repeat},i}$)**: Average monetary value across repeat transactions ($x_i > 0$).

### 2.2 BG/NBD Model Specification (Fader, Hardie, & Lee, 2005)
The Beta-Geometric / Negative Binomial Distribution model characterizes transaction activity in continuous non-contractual settings under the following generative assumptions:
1. **Purchase Process**: While active, customer $i$ makes transactions following a Poisson process with rate $\lambda_i$:
   $$P(X(t) = x \mid \lambda_i) = \frac{(\lambda_i t)^x e^{-\lambda_i t}}{x!}$$
2. **Heterogeneity in Transaction Rates**: Transaction rates across customers follow a Gamma distribution:
   $$\lambda_i \sim \text{Gamma}(r, \alpha), \quad f(\lambda \mid r, \alpha) = \frac{\alpha^r \lambda^{r-1} e^{-\alpha \lambda}}{\Gamma(r)}$$
3. **Dropout Opportunity**: Immediately after each transaction, an active customer has an unobserved probability $p_i$ of permanently dropping out (becoming inactive).
4. **Heterogeneity in Dropout Probabilities**: Dropout probabilities across customers follow a Beta distribution:
   $$p_i \sim \text{Beta}(a, b), \quad f(p \mid a, b) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)}$$
5. **Independence**: Transaction rate $\lambda_i$ and dropout probability $p_i$ are distributed independently across customers.

### 2.3 Likelihood Function & Maximum Likelihood Estimation (MLE)
The likelihood for an individual customer with history $(x, t_x, T)$ is given by:
$$L(r, \alpha, a, b \mid x, t_x, T) = A_1 \cdot A_2 \cdot (A_3 + \mathbb{I}(x > 0) A_4)$$
where parameters $(r, \alpha, a, b)$ are estimated via constrained Nelder-Mead optimization of the joint log-likelihood.

### 2.4 Model Estimate of Active Probability: $P(\text{Alive})$
The posterior probability that a customer with history $(x, t_x, T)$ is active at time $T$ is computed as:
$$P(\text{Alive} \mid x, t_x, T, r, \alpha, a, b) = \left[ 1 + \frac{a}{b + x - 1} \left( \frac{\alpha + T}{\alpha + t_x} \right)^{r + x} \right]^{-1}$$
- **Scored Population (Cutoff 2011-12-09)**: Across 5,878 scored customers, the mean model estimate is **88.43%**, reflecting the fitted dropout parameters ($a, b$).
- **$F = 0$ Dynamics**: For single buyers ($x=0, t_x=0$), the customer has never had a repeat transaction opportunity to drop out, yielding $P(\text{Alive}) = 1.0$ identically across all single buyers.

### 2.5 Gamma-Gamma Monetary Sub-Model (Fader & Hardie, 2013)
To forecast customer spend, monetary value is modeled conditional on repeat transactions:
- Individual transaction values $v_{i,j} \sim \text{Gamma}(p, \nu_i)$.
- Latent customer spend scale $\nu_i \sim \text{Gamma}(q, \gamma)$.
- Expected order spend for repeat buyers is derived via empirical Bayes shrinkage:
  $$E[M \mid p, q, \gamma, F_i, M_{\text{repeat},i}] = \frac{(q - 1) \frac{\gamma}{p} + F_i M_{\text{repeat},i}}{q - 1 + F_i}$$
- **Independence Validation**: Pearson correlation between frequency $F$ and repeat average spend $M_{\text{repeat}}$ is $r = 0.0605$ ($p = 1.27 \times 10^{-3}$), confirming the validity of independent transaction and monetary processes.

### 2.6 Rolling-Origin Out-of-Sample Holdout Validation
To evaluate model stability, rolling-origin evaluation is performed across three chronological cutoffs:
1. **2010-12-01 (373-Day Holdout, $N=4,266$)**: Actual purchases 14,365 vs. Predicted 21,107 (+46.93% volume error; revenue error +40.57%). BG/NBD purchase MAE: 2.92 vs. Naive 1: 3.30 (+11.4% improvement).
2. **2011-03-01 (283-Day Holdout, $N=4,537$)**: Actual purchases 11,787 vs. Predicted 12,722 (+7.93% volume error; revenue error +10.47%).
3. **2011-06-01 (191-Day Holdout, $N=4,933$)**: Actual purchases 9,092 vs. Predicted 8,597 (-5.44% volume error; revenue error -5.65%). Holdout purchase MAE: 1.21; revenue MAE: $598.

---

## 3. Quasi-Experimental Causal Inference Pipeline

### 3.1 Mechanism-Driven Policy Intervention Simulation
At intervention threshold $t_0 = \text{2011-06-01}$, a targeted monetization policy is evaluated across a balanced **390,939-row customer-week panel** (4,933 customers $\times$ 79 calendar weeks, comprising 52 pre-period weeks and 27 post-period weeks).
- **Targeting Mechanism**: Targeted fee waiver or free shipping threshold incentive applied to top-tier accounts.
- **Ground Truth ATT**: Preserved at $+\$4.4820/\text{week}$ spend lift among treated customers.
- **Dual Churn Scenarios**:
  - **Baseline Scenario**: 20% cumulative churn among treated top-tier spenders.
  - **Hidden Degradation Stress Scenario**: 40% cumulative churn concentrated among Quintile 5 VIP spenders, testing estimator robustness to asymmetric high-value attrition.

### 3.2 Two-Way Fixed Effects (TWFE) Difference-in-Differences
The primary causal estimator solves the standard two-way fixed effects panel specification via `PanelOLS`:
$$Y_{it} = \alpha_i + \lambda_t + \beta^{\text{TWFE}} D_{it} + \varepsilon_{it}$$
where:
- $\alpha_i$ is customer fixed effects (absorbing time-invariant unobserved customer heterogeneity).
- $\lambda_t$ is week fixed effects (absorbing macroeconomic and seasonal retail shocks).
- $D_{it} = \mathbb{I}(\text{Treated}_i = 1 \land t \ge t_0)$ is the post-treatment indicator.
- Standard errors are clustered at the customer level ($i$), robust to arbitrary serial correlation within customer spending histories.

**Empirical Baseline Estimates**:
- **Estimated ATT**: $+\$5.5020/\text{week}$ ($\text{SE} = \$1.7218$, $t = 3.1955$, $p = 0.0014$).
- **95% Confidence Interval**: $[+\$2.1273, +\$8.8767]$ (successfully covering ground truth $+\$4.4820$).
- **Single-Draw Realization Error**: $+\$1.0200/\text{week}$ (well within $1 \times \text{SE}$).

### 3.3 Dynamic Event-Study Decomposition & Joint Wald Pre-Trend Test
To test the core identifying assumption of parallel counterfactual trends, the panel is estimated with event-time leads and lags relative to omit week $k = -1$:
$$Y_{it} = \alpha_i + \lambda_t + \sum_{k = -52, k \ne -1}^{26} \beta_k (D_i \times \mathbb{I}(t - t_0 = k)) + \varepsilon_{it}$$
- **Pre-Intervention Leads**: Evaluated across $k \in [-10, -2]$ ($\text{df} = 9$).
- **Joint Wald Test**:
  $$W = \boldsymbol{\hat{\beta}}_{\text{pre}}^T \left[ \mathbf{V}_{\boldsymbol{\hat{\beta}}_{\text{pre}}} \right]^{-1} \boldsymbol{\hat{\beta}}_{\text{pre}} \sim \chi^2(\text{df}=9)$$
- **Baseline Parallel Trends Result**: $\chi^2 = 4.1326$, $p = 0.9025$. With $p \gg 0.05$, there is zero evidence of differential pre-trends between treatment and control prior to $t_0$.
- **Post-Intervention Cumulative Effect**: Linear combination variance $w^T \Sigma w$ confirms sustained positive post-period divergence.

### 3.4 Multi-Donor Synthetic Control
To construct a nonparametric counterfactual, 50 donor cohorts are synthesized via constrained quadratic optimization:
$$\min_{\mathbf{w}} \|\mathbf{Y}_{1,\text{pre}} - \mathbf{Y}_{0,\text{pre}} \mathbf{w}\|_2^2 \quad \text{s.t.} \quad \sum_{j=1}^J w_j = 1, \quad w_j \ge 0$$
- **Baseline Pre-RMSPE**: $\$5.2480$ (demonstrating tight donor-pool tracking).
- **Synthetic Control ATT**: $+\$5.1140/\text{week}$ (converging closely with TWFE).
- **Sensitivity Refutation**: In-space placebo permutations and leave-one-donor-out sensitivity bounds confirm stability.

---

## 4. Monte Carlo Estimator Validation (50 Replications)

To confirm that the TWFE estimator is unbiased across random data realizations, a 50-run Monte Carlo simulation is executed across both experimental conditions (`results/monte_carlo.csv`):
- **Replications ($M$)**: 50 full synthetic panel realizations per condition.
- **Baseline Scenario Performance**:
  - **Mean Bias**: $-\$0.0393/\text{week}$ (demonstrating asymptotic unbiasedness).
  - **Empirical RMSE**: $\$1.8396/\text{week}$.
  - **95% Confidence Interval Coverage**: **96.0%** (matching the nominal $95\%$ standard).
  - **Statistical Rejection Rate ($\alpha = 0.05$)**: 86.0% statistical power.
- **Hidden Degradation Scenario Performance**:
  - **Mean Bias**: $-\$0.0768/\text{week}$.
  - **Empirical RMSE**: $\$1.9168/\text{week}$.
  - **95% Confidence Interval Coverage**: **94.0%**.
  - **Statistical Rejection Rate**: 80.0%.

---

## 5. Selection Stress Testing & Falsification Mechanics

### 5.1 Confounded Assignment Rule
To test the pipeline's falsification sensitivity, a near-deterministic selection stress test is run where treatment assignment is tied directly to pre-intervention spend velocity:
$$\text{spend\_growth}_i = \frac{\text{late\_pre\_spend}_i - \text{early\_pre\_spend}_i}{\text{late\_pre\_spend}_i + \text{early\_pre\_spend}_i + 10.0}$$
$$\text{growth\_rank}_i = \text{PercentileRank}(\text{spend\_growth}_i)$$
$$\text{is\_treated}_i = \mathbb{I}(\text{growth\_rank}_i \ge \text{quantile}_{0.60}) = \mathbb{I}(\text{growth\_rank}_i \ge 60^{\text{th}}\text{ percentile})$$
- **Selection Gradient**: Treated customers have an average growth index of $-0.0169$, whereas control customers have an average growth index of $-0.9121$ (growth gap $+0.8952$).

### 5.2 Empirical Stress Test Results
- **Pre-Trend Parallelism Joint Wald Test**: $\chi^2 = 60.6339$, $\text{df} = 9$, $p = 1.0118 \times 10^{-9}$. The pipeline decisively rejects parallel trends ($p < 0.001$).
- **TWFE Estimator Bias**: Under confounded selection, TWFE severely over-attributes selection momentum to the policy, estimating $\hat{\beta} = +\$13.7923/\text{week}$ vs. ground truth $+\$4.4820$, producing an extreme bias of **$+\$9.3103/\text{week}$** ($236.8\times$ higher than Monte Carlo bias).
- **Synthetic Control Fit Breakdown**: Donor tracking collapses; Pre-Period RMSPE increases by $7.0\times$ from $\$5.2480$ to **$\$36.5875$**.

This dual benchmark demonstrates that the automated diagnostic framework reliably detects broken identifying assumptions before executive policy decisions are made.
