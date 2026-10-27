import os
import polars as pl

# Define Locals

base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_dir = os.path.join(base_out_dir, "combined")
os.makedirs(final_dir, exist_ok=True)

# List of Tables

concat_list = ["panelists", "purchases", "trips"]
for table in concat_list:
    file_pattern = os.path.join(base_out_dir, f"{table}_*.parquet")
    concat_out = os.path.join(final_dir, f"{table}.parquet")

    print(f"Combining {table} files...")

    (pl.scan_parquet(file_pattern).sink_parquet(concat_out))
    print(f"Saved: {concat_out}")
