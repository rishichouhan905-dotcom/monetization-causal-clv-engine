"""Dataset Ingestion Wrapper Module.

Delegates dataset ingestion and parquet creation to scripts/download_data.py.
"""

import sys
from pathlib import Path

# Add scripts directory to sys.path
scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from download_data import download_data


def ingest_data(use_synthetic: bool = False) -> None:
    """Ingests raw transactions dataset into DuckDB table 'raw_online_retail'."""
    download_data(use_synthetic=use_synthetic)


if __name__ == "__main__":
    use_synth = "--synthetic" in sys.argv
    ingest_data(use_synthetic=use_synth)
