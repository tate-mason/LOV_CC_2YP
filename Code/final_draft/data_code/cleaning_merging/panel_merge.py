import os
import polars as pl

base_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(base_dir, "merged_panel.parquet")

## === Load Data === ##
print("Loading lazy frames...")
panelists = pl.scan_parquet(os.path.join(base_dir, "combined/panelists.parquet"))
purchases = pl.scan_parquet(os.path.join(base_dir, "combined/purchases.parquet"))
trips = pl.scan_parquet(os.path.join(base_dir, "combined/trips.parquet"))
products = pl.scan_parquet(os.path.join(base_dir, "cleaned_products.parquet"))

## === Ensure Consistent Join Key Data Types === ##
# Casting join keys prevents type-mismatch crashes and speeds up join execution
purchases = purchases.with_columns(
    [pl.col("trip_code_uc").cast(pl.Int64), pl.col("upc").cast(pl.Int64)]
)

trips = trips.with_columns(
    [
        pl.col("trip_code_uc").cast(pl.Int64),
        pl.col("household_code").cast(pl.Int64),
        pl.col("panel_year").cast(pl.Int32),
    ]
)

panelists = panelists.with_columns(
    [pl.col("household_code").cast(pl.Int64), pl.col("panel_year").cast(pl.Int32)]
)

products = products.with_columns([pl.col("upc").cast(pl.Int64)])

## === Merge === ##
merged = (
    purchases.join(trips, on="trip_code_uc", how="inner")
    .join(panelists, on=["household_code", "panel_year"], how="left")
    .join(products, on="upc", how="left")
)

print("Merging purchases, trips, panelists, and products...")
print("Streaming to disk (this will take a few minutes)...")
merged.sink_parquet(final_out)
print(f"\n[SUCCESS] Successfully saved merged file to: {final_out}")

## === Run Diagnostics & Quality Checks === ##
print("\n" + "=" * 50)
print("RUNNING DIAGNOSTICS ON MERGED DATASET")
print("=" * 50)

# 1. Scan output dataset lazily
merged_scan = pl.scan_parquet(final_out)

# 2. Total Row Count Check
total_rows = merged_scan.select(pl.len()).collect().item()
print(f"\n[1] Total Row Count: {total_rows:,}")

if total_rows == 0:
    raise ValueError("CRITICAL ERROR: Merged dataset contains 0 rows! Check join keys.")

# 3. Check Row Loss Against Core Anchor (Purchases)
purchases_rows = purchases.select(pl.len()).collect().item()
pct_retained = (total_rows / purchases_rows) * 100
print(f"    - Anchor purchases rows: {purchases_rows:,}")
print(f"    - Retained post-join: {pct_retained:.2f}%")

# 4. Null / NA Check across key columns
print("\n[2] Null Value Diagnostics (Key Columns):")
key_cols = ["trip_code_uc", "household_code", "upc", "panel_year"]
existing_keys = [col for col in key_cols if col in merged_scan.collect_schema().names()]

null_counts = (
    merged_scan.select([pl.col(c).null_count().alias(c) for c in existing_keys])
    .collect()
    .to_dict(as_series=False)
)

for col, count in null_counts.items():
    null_pct = (count[0] / total_rows) * 100
    print(f"    - {col}: {count[0]:,} nulls ({null_pct:.2f}%)")

# 5. File Metadata & Summary
file_size_gb = os.path.getsize(final_out) / (1024**3)
schema_cols = merged_scan.collect_schema().names()

print("\n[3] Output File Summary:")
print(f"    - Total Columns: {len(schema_cols)}")
print(f"    - Disk Size: {file_size_gb:.2f} GB")
print(f"    - Columns: {schema_cols}")
print("=" * 50)
print("DIAGNOSTICS COMPLETE - File is verified and ready.")
