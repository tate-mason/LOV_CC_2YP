
# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_legacy_server
require_legacy_server()
import os
import polars as pl

market_codes = [
    "Atlanta", "Chicago", "Houston", "Denver",
    "Phoenix", "Philadelphia", "San_Diego", "Des_Moines"
]
for m in market_codes:
    retail_m = (
        pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets/retail_{m}.parquet')
        .rename(str.lower)
        .collect()
    )

    print(f"Joined: {retail_m.height}")
