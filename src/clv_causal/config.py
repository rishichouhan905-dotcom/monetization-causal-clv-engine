"""Centralized Configuration Module for Monetization Causal CLV Engine.

Loads settings from config.yaml located at the project root and exposes
paths, dates, model hyperparameters, and decision engine parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

# Base Paths
CONFIG_FILE = Path(__file__).resolve()
PACKAGE_DIR = CONFIG_FILE.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
YAML_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Loads configuration dictionary from config.yaml if present."""
    if YAML_CONFIG_PATH.exists():
        with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG_DICT: Dict[str, Any] = load_config()

# Data Directories & File Paths
paths_cfg = CONFIG_DICT.get("paths", {})
DATA_DIR: Path = PROJECT_ROOT / paths_cfg.get("data_dir", "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR: Path = PROJECT_ROOT / paths_cfg.get("raw_dir", "data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH: str = str(PROJECT_ROOT / paths_cfg.get("db_path", "data/retail_warehouse.duckdb"))
EXCEL_PATH: str = str(PROJECT_ROOT / paths_cfg.get("excel_path", "data/online_retail_II.xlsx"))
PARQUET_PATH: str = str(PROJECT_ROOT / paths_cfg.get("parquet_path", "data/raw/online_retail_II.parquet"))
CSV_PATH: str = str(PROJECT_ROOT / paths_cfg.get("csv_path", "data/online_retail_II.csv"))
UCI_URL: str = paths_cfg.get(
    "uci_url",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
)

# Key Dates
dates_cfg = CONFIG_DICT.get("dates", {})
START_DATE: str = dates_cfg.get("start_date", "2009-12-01")
END_DATE: str = dates_cfg.get("end_date", "2011-12-09")
CALIBRATION_END_DATE: str = dates_cfg.get("calibration_end_date", "2010-12-01")
HOLDOUT_END_DATE: str = dates_cfg.get("holdout_end_date", "2011-12-09")
OBSERVATION_END_DATE: str = CALIBRATION_END_DATE
INTERVENTION_DATE: str = dates_cfg.get("intervention_date", "2010-06-01")

# CLV Model Parameters
clv_cfg = CONFIG_DICT.get("clv_model", {})
ESTIMATION_METHOD: str = clv_cfg.get("estimation_method", "Maximum Likelihood Estimation (MLE)")
PENALIZER_COEF: float = float(clv_cfg.get("penalizer_coef", 0.0))
DISCOUNT_RATE: float = float(clv_cfg.get("discount_rate", 0.01))
PREDICTION_MONTHS: List[int] = clv_cfg.get("prediction_months", [12, 24])
N_BOOTSTRAPS: int = int(clv_cfg.get("n_bootstraps", 30))

# Causal Model Parameters
causal_cfg = CONFIG_DICT.get("causal_model", {})
TREATMENT_SHARE: float = float(causal_cfg.get("treatment_share", 0.40))
ORDER_RATE_CHANGE: float = float(causal_cfg.get("order_rate_change", 0.15))
AOV_CHANGE: float = float(causal_cfg.get("aov_change", 0.10))
TOP_CLV_CHURN_HAZARD: float = float(causal_cfg.get("top_clv_churn_hazard", 0.20))
RANDOM_SEED: int = int(causal_cfg.get("random_seed", 42))
POLICY_LIFT_PCT: float = float(causal_cfg.get("policy_lift_pct", 0.15))
TARGET_TREATED_COUNTRY: str = causal_cfg.get("target_treated_country", "Germany")
TOP_N_DONORS: int = int(causal_cfg.get("top_n_donors", 7))

# Decision Engine Parameters
decision_cfg = CONFIG_DICT.get("decision_engine", {})
CAC_PER_CUSTOMER: float = float(decision_cfg.get("cac_per_customer", 45.0))
GROSS_MARGIN_PCT: float = float(decision_cfg.get("gross_margin_pct", 0.40))
CAMPAIGN_COST_PER_TARGET: float = float(decision_cfg.get("campaign_cost_per_target", 15.0))
