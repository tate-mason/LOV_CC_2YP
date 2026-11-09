
# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_legacy_server
require_legacy_server()
import os
import polars as pl

years = [2022, 2023, 2024]
base_raw_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS"
base_out_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS"
os.makedirs(base_out_dir, exist_ok=True)

files_to_convert = {
    "panelist.tsv": "panelists",
    "purchase.tsv": "purchases",
    "trip.tsv": "trips",
    "productattributes.tsv": "product_attr",
    "productdesc.tsv": "product_desc",
    "retailer.tsv": "retailer",
    "producthierarchy.tsv": "hierarchy",
}

for y in years:
    print(f"--- Converting year {y} ---")
    for tsv_name, out_name in files_to_convert.items():
        tsv_path = f"{base_raw_dir}/{y}/Annual_Files/{tsv_name}"
        parquet_out = f"{base_out_dir}/{out_name}_{y}.parquet"

        if os.path.exists(tsv_path):
            (
                pl.scan_csv(
                    tsv_path, separator="\t", quote_char=None, infer_schema_length=0
                )
                .rename(str.lower)
                .sink_parquet(parquet_out)
            )
            print(f"Converted {tsv_name} -> {out_name}_{y}.parquet")
        else:
            print(f"WARNING: File not found {tsv_path}")
