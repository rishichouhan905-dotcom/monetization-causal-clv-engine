# Changelog

All notable changes to the Enterprise Probabilistic CLV & Quasi-Experimental Causal Platform are documented in this file.

## [Unreleased] - 2026-10-14

### Reconciled
- **P(Alive) Metric Standardization**: Reconciled the mean active probability metric across the entire application and documentation. An earlier preliminary 0.854 figure originated from an intermediate rolling-origin evaluation cutoff calibration. The unified production figure across all 5,878 scored customers at the final warehouse scoring cutoff (2011-12-09) is **88.4%** (mean P(Alive) = 0.884), reflecting the fitted BG/NBD dropout parameters. All UI notes mentioning the legacy 0.854 intermediate figure have been removed from the application dashboard in favor of the single unified estimate.

### Added
- **Demo Mode Extracts**: Added full Parquet extractions to `app_data/` (`dim_customers`, `fact_orders`, `fact_returns`, `dim_products`, `pred_clv_summary`, `clv_scores`, `clv_holdout_deciles`, `clv_holdout_metrics`, `decision_segment_summary`, `decision_scenarios`, `causal_att_vs_truth`, `causal_clv_heterogeneity`, `causal_treatment_control_diagnostics`, `causal_experiment_data`) under 25 MB total, enabling offline rendering and demonstration without requiring the full DuckDB warehouse file.
- **Headless & AppTest Verification**: Added automated tests verifying that all pages and tabs render without exceptions when the DuckDB database file is absent.
- **Data Source & Attribution Notice**: Added formal `NOTICE` file specifying UCI Machine Learning Repository Online Retail II donor attribution, Creative Commons Attribution 4.0 International (CC BY 4.0) license, and DOI citation.
- **Automated CI Suite**: Added GitHub Actions workflow `.github/workflows/ci.yml` supporting sample-mode dbt build (`SAMPLE=1`) and fast unit testing with pytest markers (`not slow and not fulldata`).
- **Monte Carlo Target**: Added `make montecarlo` running 50 replications of the synthetic quasi-experiment with runtime reporting.
