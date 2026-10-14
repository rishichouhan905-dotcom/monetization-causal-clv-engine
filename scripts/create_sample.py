"""Creates a 5,000-row stratified sample of raw transactions for fast CI testing and sample runs."""

from pathlib import Path
import os
import numpy as np
import pandas as pd

from clv_causal.config import DATA_DIR, EXCEL_PATH, PARQUET_PATH


def create_sample(sample_size: int = 5000, random_seed: int = 42) -> Path:
    """Creates a 5,000-row stratified sample of raw transactions saved to data/sample/."""
    sample_dir = DATA_DIR / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = sample_dir / "online_retail_II_sample.parquet"
    out_csv = sample_dir / "online_retail_II_sample.csv"
    data_source_file = sample_dir / "DATA_SOURCE.txt"

    # Read source data (parquet if exists, else excel)
    if Path(PARQUET_PATH).exists():
        print(f"Reading raw transactions from {PARQUET_PATH}...")
        df = pd.read_parquet(PARQUET_PATH)
    elif Path(EXCEL_PATH).exists():
        print(f"Reading raw transactions from {EXCEL_PATH}...")
        excel_dict = pd.read_excel(EXCEL_PATH, sheet_name=None)
        df = pd.concat(excel_dict.values(), ignore_index=True)
    else:
        raise FileNotFoundError(f"Neither {PARQUET_PATH} nor {EXCEL_PATH} found to generate sample from!")

    # Stratify by Country and return indicator (Quantity < 0)
    print(f"Sampling {sample_size:,} stratified rows from {len(df):,} total rows...")
    df["is_return_stratum"] = (df["Quantity"] < 0).astype(int)
    # Group top countries, combine long-tail into Other to avoid single-row strata
    top_countries = set(df["Country"].value_counts().head(10).index)
    df["country_stratum"] = df["Country"].apply(lambda c: c if c in top_countries else "Other")
    df["stratum"] = df["country_stratum"].astype(str) + "_" + df["is_return_stratum"].astype(str)

    # Proportional stratified sampling
    np.random.seed(random_seed)
    sampled_indices = []
    strata_counts = df["stratum"].value_counts(normalize=True)

    for stratum, frac in strata_counts.items():
        stratum_df = df[df["stratum"] == stratum]
        n_stratum = int(round(frac * sample_size))
        if n_stratum == 0 and len(stratum_df) > 0:
            n_stratum = 1
        n_stratum = min(n_stratum, len(stratum_df))
        sampled_indices.extend(stratum_df.sample(n=n_stratum, random_state=random_seed).index)

    # Adjust to exactly sample_size
    if len(sampled_indices) > sample_size:
        sampled_indices = list(np.random.choice(sampled_indices, size=sample_size, replace=False))
    elif len(sampled_indices) < sample_size:
        remaining = df.index.difference(sampled_indices)
        additional = np.random.choice(remaining, size=sample_size - len(sampled_indices), replace=False)
        sampled_indices.extend(additional)

    sample_df = df.loc[sampled_indices].copy()
    sample_df.drop(columns=["is_return_stratum", "country_stratum", "stratum"], inplace=True, errors="ignore")
    sample_df.sort_values(by="InvoiceDate", inplace=True)

    # Export to data/sample/
    sample_df.to_parquet(out_parquet, index=False)
    sample_df.to_csv(out_csv, index=False)
    with open(data_source_file, "w", encoding="utf-8") as f:
        f.write(f"DATA_SOURCE=STRATIFIED_SAMPLE_{sample_size}\nRANDOM_SEED={random_seed}\nROWS={len(sample_df)}\n")

    print(f"[OK] Successfully saved {len(sample_df):,} stratified sample rows to:")
    print(f"     - {out_parquet} ({os.path.getsize(out_parquet) / 1024:.1f} KB)")
    print(f"     - {out_csv} ({os.path.getsize(out_csv) / 1024:.1f} KB)")
    return out_parquet


if __name__ == "__main__":
    create_sample(sample_size=5000)
