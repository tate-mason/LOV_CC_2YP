import glob
import os
import polars as pl

base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
final_dir = os.path.join(base_out_dir, "combined")
os.makedirs(final_dir, exist_ok=True)

concat_list = ["stores", "rms", "products_extra", "3612", "3603"]

for table in concat_list:
    file_pattern = os.path.join(base_out_dir, f"cleaned_{table}_*.parquet")
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"WARNING: No files found matching {file_pattern}. Skipping {table}...")
        continue

    concat_out = os.path.join(final_dir, f"{table}.parquet")
    print(f"Combining {len(matching_files)} files for '{table}'...")

    (pl.scan_parquet(file_pattern).sink_parquet(concat_out))
    print(f"Saved: {concat_out}")
