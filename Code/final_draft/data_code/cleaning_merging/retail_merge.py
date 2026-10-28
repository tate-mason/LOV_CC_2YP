import os
import polars as pl

base_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
final_out = os.path.join(base_dir, "merged_rms.parquet")

print("Loading lazy frames...")

# 1. Load movement anchor
movements = pl.concat(
    [
        pl.scan_parquet(os.path.join(base_dir, "combined/3612.parquet")),
        pl.scan_parquet(os.path.join(base_dir, "combined/3603.parquet")),
    ]
).with_columns(
    [
        pl.col("upc").cast(pl.Int64),
        pl.col("store_code_uc").cast(pl.Int64),
        pl.col("panel_year").cast(pl.Int32),
    ]
)

# 2. RMS table (deduplicated on upc to cleanly supply upc_ver_uc)
rms = (
    pl.scan_parquet(os.path.join(base_dir, "combined/rms.parquet"))
    .with_columns([pl.col("upc").cast(pl.Int64), pl.col("upc_ver_uc").cast(pl.Int32)])
    .unique(subset=["upc"], keep="first")
)

# 3. Products table (deduplicated on exact UPC + Version Code composite key)
products = (
    pl.scan_parquet(os.path.join(base_dir, "cleaned_products.parquet"))
    .with_columns([pl.col("upc").cast(pl.Int64), pl.col("upc_ver_uc").cast(pl.Int32)])
    .unique(subset=["upc", "upc_ver_uc"], keep="first")
)

# 4. Stores table (unique on store + year)
stores = (
    pl.scan_parquet(os.path.join(base_dir, "combined/stores.parquet"))
    .rename({"year": "panel_year"})  # Safely renames 'year' to 'panel_year'
    .with_columns(
        [
            pl.col("store_code_uc").cast(pl.Int64),
            pl.col("panel_year").cast(pl.Int32),
        ]
    )
    .unique(subset=["store_code_uc", "panel_year"], keep="first")
)

# 5. Products Extra (sorted to prioritize non-null flavor values before deduplication)
products_extra = (
    pl.scan_parquet(os.path.join(base_dir, "combined/products_extra.parquet"))
    .with_columns([pl.col("upc").cast(pl.Int64), pl.col("panel_year").cast(pl.Int32)])
    .sort("flavor_descr", descending=False, nulls_last=True)
    .unique(subset=["upc", "panel_year"], keep="first")
)

# 6. Brand Variations (deduplicated on brand code)
brand_var = (
    pl.scan_parquet(os.path.join(base_dir, "brand_variations.parquet"))
    .with_columns(pl.col("brand_code_uc").cast(pl.Int64))
    .unique(subset=["brand_code_uc"], keep="first")
)

## === Optimized Merge Chain === ##
merged = (
    movements
    # Step A: Attach upc_ver_uc from RMS
    .join(rms, on="upc", how="left", suffix="_rms")
    # Step B: Match exact version in products
    .join(products, on=["upc", "upc_ver_uc"], how="left", suffix="_prod")
    # Step C: Match store metadata by store
    .join(stores, on=["store_code_uc", "panel_year"], how="left", suffix="_str")
    # Step D: Match extra annual metadata
    .join(products_extra, on=["upc", "panel_year"], how="left", suffix="_xtra")
    # Step E: Match master brand dictionary
    .join(brand_var, on="brand_code_uc", how="left", suffix="_brand")
)

print("Streaming merged RMS dataset to disk...")
merged.sink_parquet(final_out)
print(f"Successfully saved: {final_out}")

## === Run Post-Merge Audit === ##
print("\n" + "=" * 60)
print("RUNNING POST-MERGE AUDIT: RMS DATASET")
print("=" * 60)

merged_scan = pl.scan_parquet(final_out)

# 1. Total Row Count & Anchor Retention
total_merged_rows = merged_scan.select(pl.len()).collect().item()
anchor_rows = movements.select(pl.len()).collect().item()
pct_retained = (total_merged_rows / anchor_rows) * 100

print(f"\n[1] Row Count & Integrity Audit:")
print(f"    - Movements Anchor Rows: {anchor_rows:,}")
print(f"    - Merged File Rows:     {total_merged_rows:,}")
print(f"    - Row Ratio:           {pct_retained:.2f}%")

if total_merged_rows != anchor_rows:
    print("    ⚠️  WARNING: Row count discrepancy detected! Check join keys.")
else:
    print("    ✓ PERFECT MATCH: Exact 1:1 row retention (0 rows added or dropped).")

# 2. Dimension Match Rates
print("\n[2] Dimension Table Match Rates:")
match_targets = {
    "products (brand_descr)": "brand_descr",
    "stores (retailer_code)": "retailer_code",
    "products_extra (flavor_descr)": "flavor_descr",
    "brand_var": "brand_code_uc_brand"
    if "brand_code_uc_brand" in merged_scan.collect_schema().names()
    else "brand_code_uc",
}

for dim_name, col_name in match_targets.items():
    if col_name in merged_scan.collect_schema().names():
        matched_count = (
            merged_scan.filter(pl.col(col_name).is_not_null())
            .select(pl.len())
            .collect()
            .item()
        )
        match_pct = (matched_count / total_merged_rows) * 100
        print(f"    - {dim_name:<30}: {matched_count:,} matched ({match_pct:.2f}%)")

# 3. File Summary
file_size_gb = os.path.getsize(final_out) / (1024**3)
schema_cols = merged_scan.collect_schema().names()

print("\n[3] Output File Summary:")
print(f"    - Total Columns: {len(schema_cols)}")
print(f"    - File Size:     {file_size_gb:.2f} GB")
print(f"    - Output Path:   {final_out}")
print("=" * 60)
print("AUDIT COMPLETE.")
