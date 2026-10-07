import os
from pathlib import Path

# Base Paths
CONFIG_FILE = Path(__file__).resolve()
PACKAGE_DIR = CONFIG_FILE.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Data Directories & File Paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(DATA_DIR / "retail_warehouse.duckdb")
EXCEL_PATH = str(DATA_DIR / "online_retail_II.xlsx")
CSV_PATH = str(DATA_DIR / "online_retail_II.csv")

# Remote Dataset URL
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
