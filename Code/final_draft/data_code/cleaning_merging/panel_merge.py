import os
import polars as pl

base_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(base_dir, "merged_panel.parquet")

## === Load Data === ##

panelists = pl.scan_parquet(os.path.join(base_dir, "combined/panelists.parquet"))
purchases = pl.scan_parquet(os.path.join(base_dir, "combined/purchases.parquet"))
trips = pl.scan_parquet(os.path.join(base_dir, "combined/trips.parquet"))

products = pl.scan_parquet(os.path.join(base_dir, "cleaned_products.parquet"))
retailers = pl.scan_parquet(os.path.join(base_dir, "cleaned_retailers.parquet"))

## === Merge === ##
merged = (
    purchases.join(trips, on="trip_code_uc", how="inner")
    .join(panelists, on=["household_code", "panel_year"], how="left")
    .join(products, on="upc", how="left")
)
print("Merging all tables and streaming...")
merged.sink_parquet(final_out)
print(f"Successfully saved: {final_out}")

## === Diagnostics & Quality Checks === ##
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

if pct_retained < 80:
    print(
        "    WARNING: High row loss after joining trips! Verify trip_code_uc key matching."
    )

# 4. Null / NA Check across key join columns
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

# 5. Primary Key Uniqueness Check (trip_code_uc + upc)
print("\n[3] Uniqueness Check:")
if (
    "trip_code_uc" in merged_scan.collect_schema().names()
    and "upc" in merged_scan.collect_schema().names()
):
    unique_line_items = (
        merged_scan.select(pl.struct(["trip_code_uc", "upc"]).n_unique())
        .collect()
        .item()
    )
    dup_count = total_rows - unique_line_items
    print(f"    - Duplicate transaction lines (trip_code_uc + upc): {dup_count:,}")
    if dup_count > 0:
        print(
            "    WARNING: Duplicate records detected. Master tables may contain non-unique join keys."
        )

# 6. File Metadata & Schema Overview
file_size_gb = os.path.getsize(final_out) / (1024**3)
schema_cols = merged_scan.collect_schema().names()

print("\n[4] Output File Summary:")
print(f"    - Total Columns: {len(schema_cols)}")
print(f"    - Disk Size: {file_size_gb:.2f} GB")
print(f"    - Output File Path: {final_out}")
print("=" * 50)
print("DIAGNOSTICS COMPLETE - Dataset verified safe to proceed.")
