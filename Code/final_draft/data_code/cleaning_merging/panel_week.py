
# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_legacy_server
require_legacy_server()
import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
panel_path = os.path.join(input_dir, "merged_panel.parquet")
panel_temp = os.path.join(input_dir, "merged_panel_updated.parquet")

print("Pre-computing week_end on merged_panel.parquet...")

(
    pl.scan_parquet(panel_path)
    .with_columns(
        # Convert "YYYY-MM-DD" to Date, compute Saturday week-ending, format to YYYYMMDD Int64
        week_end=(
            pl.col("purchase_date").str.to_date()
            + pl.duration(
                days=(6 - (pl.col("purchase_date").str.to_date().dt.weekday() % 7))
            )
        )
        .dt.strftime("%Y%m%d")
        .cast(pl.Int64)
    )
    .sink_parquet(panel_temp)
)

# Overwrite original file with updated file
os.replace(panel_temp, panel_path)
print("Updated merged_panel.parquet successfully with Int64 'week_end'!")
