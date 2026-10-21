# Methodology

This document explains what the project does, why each choice was made, which assumptions each result rests on, and where the evidence is weak. All numbers come from `docs/RESULTS_FACTS.md`, which is generated from files in `results/`, `app_data/` and `holdout_report.md`.

## 1. Scope and the central caveat

The transactions are real. The pricing policy is **simulated**. The simulator changes outcomes after a policy date by known amounts, so the project can measure how well each method recovers a truth that is known. A real pricing change would not offer that check. Nothing here estimates the effect of a real policy.

## 2. Data and warehouse

**Source.** UCI Online Retail II: 1,067,371 raw rows from a UK-based online gift-ware retailer, 2009-12-01 to 2011-12-09, many wholesale customers. Both Excel sheets are loaded.

**Cleaning rules.**
- Rows without a customer ID are excluded from customer-level analysis.
- Returns (invoices starting with C, negative quantity) are kept in a separate table, `fact_returns`, instead of being dropped.
- Zero-price lines are excluded from orders.

**Result.** 805,549 positive order lines and 18,744 return lines, 5,939 customers (5,878 scored; 61 appear only in returns). Gross revenue $17,743,429.18, returns $1,095,136.79, net $16,648,292.39. The customer dimension agrees within two cents.

**Warehouse.** dbt on DuckDB with staging, intermediate and mart layers (orders, returns, customers, products). Tests include unique and non-null keys, referential integrity and a revenue identity (gross minus returns equals net). A 5,000-row stratified sample lets CI run `dbt build` without the raw data.

## 3. CLV model

**Models.** BG/NBD for purchase counts and dropout, Gamma-Gamma for average order value, both fit by maximum likelihood with `lifetimes` (penalizer 0). CLV is discounted gross revenue at 1% per month over 12 and 24 months. Gross margin is applied only in the policy simulator.

**Why penalizer 0.** A positive penalizer pushed the fitted parameters to a+b below 1. In that region the closed-form expected-purchases formula for customers with no repeat orders returns NaN, and an earlier version silently converted that to zero. With penalizer 0, a+b is 5.71, 4.11 and 2.96 at the three cutoffs, so the standard formula is valid. A numerical integration fallback exists and raises an error instead of substituting values; it agrees with the closed form to about 1e-11 and with a large Monte Carlo simulation.

**Validation design.** Rolling origin: the model is fit on data up to a cutoff (2010-12-01, 2011-03-01, 2011-06-01) and judged on purchases and revenue until 2011-12-09. Results are reported for all customers and separately for one-time buyers (F = 0) and repeat buyers (F > 0). Customers are only scored if they appear in the calibration window.

**Baselines.** (a) Repeat purchases per day times holdout days; (b) carry-forward: last window's repeat count scaled to the holdout length; (c) tenure ranking for one-time buyers.

**Findings.**
- Over-forecast shrinks as more history accumulates: +46.9% (first cutoff), +7.9%, -5.4% for purchases.
- Carry-forward beats BG/NBD on purchases MAE at the first cutoff (2.27 vs 2.92).
- For one-time buyers, BG/NBD ranking equals tenure ranking (Spearman 0.231 for both).
- BG/NBD, MBG/NBD and Pareto/NBD produce nearly identical errors.
- Mean P(alive) is 88.4%, concentrated near 1. The fitted dropout parameters are the reason; no other cause was tested.

**Untested explanations.** Seasonality (the holiday peak falls inside the first holdout) and non-stationary customer behaviour are plausible causes of the early over-forecast. They were not tested and are labelled as hypotheses.

## 4. Simulated policy

| Setting | Value |
|---|---|
| Policy date t0 | 2011-06-01 (78 pre-weeks, 27 post-weeks) |
| Eligible customers | first order before t0: 4,933 |
| Treatment share | 40%, assigned at random within pre-period spend quintiles (seed 42) |
| Treated customers | 1,973 |
| Order-rate lift | +15% of each customer's pre-period weekly order rate (new orders appear in zero-purchase weeks) |
| Basket-size lift | +10% on existing orders |
| Top-quintile churn | cumulative probability over the 27 weeks: 20% (`baseline`), 40% (`hidden_degradation`), converted to a weekly hazard (about 0.82% and 1.87%) |
| Panel | balanced customer-by-week, 390,939 rows, from each customer's first-order week |

Quintiles use observed pre-period spend, never model scores fitted on later data. The pre-period order rate excludes the first order and is shrunk toward the mean, because counting the first order biases the rate by tenure (84% of the lowest quintile are one-order buyers). The simulator stores the counterfactual outcome, the observed outcome and the true effect for every row, so true ATTs are exact.

Outcomes: weekly orders, weekly spend winsorized at the 99th percentile, and "active in the last 90 days" (any order in the last 13 weeks).

## 5. Estimators and checks

**Two-way fixed effects DiD.** Customer and week fixed effects, standard errors clustered by customer. With one policy date and no staggered adoption, the usual TWFE weighting problems do not arise. The solver is cross-checked against `linearmodels.PanelOLS` (agreement to about 1e-14).

**Event study.** Relative weeks from -10 to +26, normalised to the average of all pre-period weeks. The post-period mean (+$5.11, SE $1.89) is computed as a linear combination using the full covariance matrix.

**Pre-trend test.** Joint Wald test of the pre-period coefficients using the full covariance matrix. The baseline gives chi2 = 4.13 (df = 9, p = 0.90). Under random assignment parallel trends hold by construction, so passing is expected; it cannot prove the assumption holds.

**Synthetic control.** 50 donor cohorts made from random control customers, non-negative weights summing to one, placebo-in-space p-values. Its pre-period RMSPE is about $5.25 per customer-week, comparable to the effect, so it cannot detect effects this small.

**Refutation and heterogeneity.** Placebo policy date, leave-one-donor-out, and effects by pre-period spend decile with Bonferroni-adjusted p-values (10 tests). In `baseline` no decile is significant after correction. In `hidden_degradation` one decile (D07) is; the top decile is not detected (true -$11.83, estimate -$10.54, SE $13.59).

**Monte Carlo.** 50 independent replications per scenario, redrawing assignment and outcomes. Reported per cohort and outcome: mean bias, RMSE, 95% interval coverage (92-100%), rejection rate at 5%. With 50 replications each coverage figure has about +/-3 points of simulation error.

**Confounded stress test.** Treatment probability rises with pre-period spend growth, defined as (late - early) / (late + early + 10) around each customer's pre-period midpoint. Treated customers average -0.017 growth against -0.912 for controls (gap 0.895), so assignment is close to deterministic. Result: pre-trend chi2 = 60.6 (p = 1.0e-9), TWFE +$14.08 against a true +$4.77 (bias +$9.31, +195%), synthetic control +$50.66 with RMSPE $36.59. This is an extreme scenario meant to show the diagnostics fire.

## 6. Policy simulator

Net margin change = gross margin x waived share x top-quintile customers x weeks x (-ATT of top-quintile spend). Low, central and high values use the confidence limits of that ATT. The Q5 active-share effect scales with the unwaived share. No separate CLV term is added, because the estimated spend and retention responses already include lost customers; adding one would double count. At zero waiver the change is exactly zero (tested). Gross margin (40%) and the guardrail thresholds (-0.5 pp amber, -2.0 pp red) are configuration assumptions.

## 7. Reproducibility

- `make all` builds everything from the raw file in about 3.6 minutes (data 53 s, dbt 21 s, CLV 15 s, causal 59 s, report 3 s).
- `make montecarlo` takes about 6 minutes.
- `docs/RESULTS_FACTS.md` is generated by `scripts/build_results_facts.py` from saved files and checks the provenance of each source.
- CI: ruff, 13 fast tests on synthetic data, and `dbt build` on the 5,000-row sample. A further 27 tests need the full data or run slowly (40 in total).

## 8. Known limitations

- Simulated policy; no real experiment, spillovers, price elasticity or competitor response.
- One retailer and two years of data.
- BG/NBD is no better than the carry-forward baseline on volume or overall ranking; its early over-forecast is not explained.
- Heavy-tailed spend makes spend effects hard to detect; the top-quintile spend effect has about 14% power in `hidden_degradation`.
- Random assignment guarantees parallel trends in the main design.
- The 40% margin, discount rate, churn levels and guardrail thresholds are assumptions.
- Extrapolating the effect beyond the observed 27 weeks assumes it persists.
