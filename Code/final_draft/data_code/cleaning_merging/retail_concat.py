import glob
import os
import polars as pl

base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
final_dir = os.path.join(base_out_dir, "combined")
os.makedirs(final_dir, exist_ok=True)

concat_targets = {
    "stores": "cleaned_stores_*.parquet",
    "products_extra": "cleaned_products_extra_*.parquet",
    "3612": "cleaned_3612_*.parquet",
    "3603": "cleaned_3603_*.parquet",
    "rms": "rms_*.parquet",
}

# Define columns prone to data type shifts across years across Nielsen datasets
CAST_SCHEMA = {
    "flavor_code": pl.String,
    "size1_code_uc": pl.String,
    "upc": pl.Int64,
    "store_code_uc": pl.Int64,
    "panel_year": pl.Int32,
}

for table, pattern in concat_targets.items():
    file_pattern = os.path.join(base_out_dir, pattern)
    matching_files = sorted(glob.glob(file_pattern))

    if not matching_files:
        print(f"WARNING: No files found matching {file_pattern}. Skipping {table}...")
        continue

    concat_out = os.path.join(final_dir, f"{table}.parquet")
    print(f"Combining {len(matching_files)} files for '{table}'...")

    # Load files individually and align column types to avoid SchemaError
    lazy_frames = []
    for f in matching_files:
        lf = pl.scan_parquet(f)

        # Cast any problematic columns present in this table schema
        schema_names = lf.collect_schema().names()
        cast_exprs = [
            pl.col(col).cast(target_type)
            for col, target_type in CAST_SCHEMA.items()
            if col in schema_names
        ]

        if cast_exprs:
            lf = lf.with_columns(cast_exprs)

        lazy_frames.append(lf)

    # Concatenate using relaxed type resolution
    (pl.concat(lazy_frames, how="vertical_relaxed").sink_parquet(concat_out))
    print(f"Saved: {concat_out}")
