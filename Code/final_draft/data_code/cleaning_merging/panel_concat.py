
# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_legacy_server
require_legacy_server()
import os
import polars as pl

# Define Locals

base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_dir = os.path.join(base_out_dir, "combined")
os.makedirs(final_dir, exist_ok=True)

# List of Tables

concat_list = ["panelists", "purchases", "trips"]
for table in concat_list:
    file_pattern = os.path.join(base_out_dir, f"cleaned_{table}_*.parquet")
    concat_out = os.path.join(final_dir, f"{table}.parquet")

    print(f"Combining {table} files...")

    (pl.scan_parquet(file_pattern).sink_parquet(concat_out))
    print(f"Saved: {concat_out}")
