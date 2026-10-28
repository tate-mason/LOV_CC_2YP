import glob
import os
import polars as pl

base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
final_dir = os.path.join(base_out_dir, "combined")
os.makedirs(final_dir, exist_ok=True)

# Map table names to their exact wildcard pattern
concat_targets = {
    "stores": "cleaned_stores_*.parquet",
    "products_extra": "cleaned_products_extra_*.parquet",
    "3612": "cleaned_3612_*.parquet",
    "3603": "cleaned_3603_*.parquet",
    "rms": "rms_*.parquet",  # No "cleaned_" prefix
}

for table, pattern in concat_targets.items():
    file_pattern = os.path.join(base_out_dir, pattern)
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"WARNING: No files found matching {file_pattern}. Skipping {table}...")
        continue

    concat_out = os.path.join(final_dir, f"{table}.parquet")
    print(f"Combining {len(matching_files)} files for '{table}'...")

    (pl.scan_parquet(file_pattern).sink_parquet(concat_out))
    print(f"Saved: {concat_out}")
