"""Dataset Ingestion and Download Script.

Ingests UCI Machine Learning Repository Online Retail II dataset into DuckDB.
NEVER silently falls back to synthetic data: if real UCI file cannot be fetched,
stops with a clear error instructing where to place online_retail_II.xlsx manually.
Loads BOTH sheets and saves to data/raw/online_retail_II.parquet.
Synthetic generator is available ONLY behind an explicit --synthetic CLI flag.
"""

import argparse
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

# Add src to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clv_causal.config import (
    DATA_DIR,
    DB_PATH,
    END_DATE,
    EXCEL_PATH,
    PARQUET_PATH,
    RAW_DIR,
    START_DATE,
    UCI_URL,
)


def generate_synthetic_dataset(
    num_records: int = 150000,
    start_date_str: str = START_DATE,
    end_date_str: str = END_DATE,
) -> pd.DataFrame:
    """Generates synthetic multi-year transaction dataset matching UCI schema."""
    print(f"Generating synthetic transaction dataset ({num_records:,} records)...")
    np.random.seed(42)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    total_days = (end_date - start_date).days

    countries = ["United Kingdom"] * 85 + ["Germany"] * 5 + ["France"] * 4 + ["EIRE"] * 3 + ["Spain"] * 2 + ["Netherlands"] * 1
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
        ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ]

    num_customers = 3500
    customer_ids = [12000 + i for i in range(num_customers)]
    cust_weights = np.random.pareto(a=1.5, size=num_customers)
    cust_probs = cust_weights / cust_weights.sum()

    chosen_custs = np.random.choice(customer_ids, size=num_records, p=cust_probs)
    cust_country_map = {cid: np.random.choice(countries) for cid in customer_ids}

    timestamps = [start_date + timedelta(seconds=int(np.random.uniform(0, total_days * 86400))) for _ in range(num_records)]
    timestamps.sort()

    invoices = []
    curr_invoice = 536365
    curr_cust = None
    last_time = None

    for i in range(num_records):
        cid = chosen_custs[i]
        ts = timestamps[i]
        if curr_cust is None or cid != curr_cust or (last_time and (ts - last_time).total_seconds() > 300):
            curr_invoice += 1
            curr_cust = cid
        invoices.append(str(curr_invoice))
        last_time = ts

    prod_indices = np.random.choice(len(products), size=num_records)
    stock_codes = [products[idx][0] for idx in prod_indices]
    descriptions = [products[idx][1] for idx in prod_indices]
    prices = [products[idx][2] for idx in prod_indices]
    quantities = np.random.choice([1, 2, 3, 4, 6, 12, 24, 48, -1, -2], size=num_records, p=[0.4, 0.25, 0.1, 0.1, 0.05, 0.04, 0.03, 0.01, 0.01, 0.01])

    for i in range(num_records):
        if quantities[i] < 0:
            invoices[i] = f"C{invoices[i]}"

    df = pd.DataFrame({
        "Invoice": invoices,
        "StockCode": stock_codes,
        "Description": descriptions,
        "Quantity": quantities,
        "InvoiceDate": timestamps,
        "Price": prices,
        "Customer ID": chosen_custs,
        "Country": [cust_country_map[cid] for cid in chosen_custs],
    })

    missing_mask = np.random.rand(num_records) < 0.05
    df.loc[missing_mask, "Customer ID"] = np.nan
    return df


def download_data(use_synthetic: bool = False) -> None:
    """Ingests dataset, loading BOTH sheets into data/raw/online_retail_II.parquet and DuckDB.

    NO SILENT FALLBACK to synthetic data: if real file cannot be fetched/read, raises RuntimeError.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    data_source_file = DATA_DIR / "DATA_SOURCE.txt"

    if use_synthetic:
        print("Explicit --synthetic flag detected. Generating synthetic dataset...")
        raw_df = generate_synthetic_dataset()
        raw_df.to_parquet(PARQUET_PATH, index=False)
        with open(data_source_file, "w", encoding="utf-8") as f:
            f.write("DATA_SOURCE=SYNTHETIC\n")
        print(f"Saved synthetic dataset to: {PARQUET_PATH}")
    else:
        # Real UCI dataset path
        if not os.path.exists(EXCEL_PATH):
            print(f"Excel file missing at {EXCEL_PATH}. Attempting download from {UCI_URL}...")
            try:
                urllib.request.urlretrieve(UCI_URL, EXCEL_PATH)
                print("Download completed successfully!")
            except Exception as e:
                err_msg = (
                    f"CRITICAL ERROR: Failed to download real UCI dataset from URL ({e}).\n"
                    f"NO SILENT FALLBACK PERMITTED. Please manually download 'online_retail_II.xlsx' and place it at:\n"
                    f"  {EXCEL_PATH}\n"
                    f"Download Link: {UCI_URL}"
                )
                print(err_msg, file=sys.stderr)
                raise RuntimeError(err_msg)

        if not os.path.exists(EXCEL_PATH):
            err_msg = (
                f"CRITICAL ERROR: 'online_retail_II.xlsx' not found at {EXCEL_PATH}.\n"
                f"Please manually place the Excel file at {EXCEL_PATH}."
            )
            print(err_msg, file=sys.stderr)
            raise RuntimeError(err_msg)

        print(f"Loading BOTH sheets from real UCI Excel file: {EXCEL_PATH}...")
        try:
            excel_dict = pd.read_excel(EXCEL_PATH, sheet_name=None)
            sheet_names = list(excel_dict.keys())
            print(f"Detected sheets in Excel file: {sheet_names}")
            raw_df = pd.concat(excel_dict.values(), ignore_index=True)

            # Ensure consistent dtypes for parquet/pyarrow export
            raw_df["Invoice"] = raw_df["Invoice"].astype(str)
            raw_df["StockCode"] = raw_df["StockCode"].astype(str)
            raw_df["Description"] = raw_df["Description"].astype(str)
            raw_df["Country"] = raw_df["Country"].astype(str)
            raw_qty = pd.to_numeric(raw_df["Quantity"], errors="coerce")
            unparseable_qty = int(raw_qty.isna().sum())
            if unparseable_qty > 0:
                raise ValueError(f"Unparseable Quantity values detected in dataset! Count: {unparseable_qty}")
            raw_df["Quantity"] = raw_qty.astype(int)
            raw_df["Price"] = pd.to_numeric(raw_df["Price"], errors="coerce").fillna(0.0).astype(float)
            raw_df["Customer ID"] = pd.to_numeric(raw_df["Customer ID"], errors="coerce")
            raw_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"])

        except Exception as e:
            err_msg = (
                f"CRITICAL ERROR: Unable to read Excel file at {EXCEL_PATH} ({e}).\n"
                f"Please verify the file integrity or re-download from:\n"
                f"  {UCI_URL}"
            )
            print(err_msg, file=sys.stderr)
            raise RuntimeError(err_msg)

        # Write data/raw/online_retail_II.parquet
        print(f"Writing concatenated dataset ({len(raw_df):,} rows) to parquet: {PARQUET_PATH}...")
        raw_df.to_parquet(PARQUET_PATH, index=False)
        with open(data_source_file, "w", encoding="utf-8") as f:
            f.write("DATA_SOURCE=REAL_UCI_ONLINE_RETAIL_II\n")

    # Ingest into DuckDB
    print(f"Ingesting {PARQUET_PATH} into DuckDB table 'raw_online_retail'...")
    conn = duckdb.connect(DB_PATH)
    conn.execute(f"CREATE OR REPLACE TABLE raw_online_retail AS SELECT * FROM '{PARQUET_PATH}'")

    # Compute summary metrics
    row_count = conn.execute("SELECT COUNT(*) FROM raw_online_retail").fetchone()[0]

    # Handle invoice date column name case
    inv_col = "InvoiceDate" if "InvoiceDate" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "invoice_date"
    cust_col = "Customer ID" if "Customer ID" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "customer_id"
    inv_no_col = "Invoice" if "Invoice" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "invoice_no"
    qty_col = "Quantity" if "Quantity" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "quantity"
    price_col = "Price" if "Price" in [c[1] for c in conn.execute("PRAGMA table_info('raw_online_retail')").fetchall()] else "price"

    min_date = conn.execute(f"SELECT MIN({inv_col}) FROM raw_online_retail").fetchone()[0]
    max_date = conn.execute(f"SELECT MAX({inv_col}) FROM raw_online_retail").fetchone()[0]
    unique_custs = conn.execute(f"SELECT COUNT(DISTINCT \"{cust_col}\") FROM raw_online_retail").fetchone()[0]

    aov_res = conn.execute(f"SELECT SUM({qty_col} * {price_col}) / NULLIF(COUNT(DISTINCT \"{inv_no_col}\"), 0) FROM raw_online_retail").fetchone()[0]
    aov = float(aov_res) if aov_res is not None else 0.0

    conn.close()

    print("\n=================== Dataset Ingestion Summary ===================")
    print(f"Data Source File:                      {PARQUET_PATH}")
    print(f"Total Record Row Count:                {row_count:,}")
    print(f"InvoiceDate Range:                     {min_date} to {max_date}")
    print(f"Unique Customer Count:                 {unique_custs:,}")
    print(f"Average Order Value (AOV):             ${aov:,.2f}")
    print("=================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UCI Online Retail II dataset into DuckDB.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset instead of real UCI data.")
    args = parser.parse_args()
    download_data(use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
