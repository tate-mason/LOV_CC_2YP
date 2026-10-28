import os
import polars as pl

base_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"

for name, path, keys in [
    ("products", "cleaned_products.parquet", ["upc"]),
    ("products_extra", "combined/products_extra.parquet", ["upc", "panel_year"]),
    ("stores", "combined/stores.parquet", ["store_code_uc", "panel_year"]),
    ("brand_var", "brand_variations.parquet", ["brand_code_uc"]),
]:
    full_path = os.path.join(base_dir, path)
    if os.path.exists(full_path):
        scan = pl.scan_parquet(full_path)
        total = scan.select(pl.len()).collect().item()
        unique = scan.select(pl.struct(keys).n_unique()).collect().item()
        print(f"{name:<15} | Total: {total:,} | Unique Keys ({keys}): {unique:,}")
        if total != unique:
            print(
                f"  ⚠️  WARNING: {name} contains {total - unique:,} duplicate key rows!"
            )
# base_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
# final_out = os.path.join(base_dir, "merged_rms.parquet")
#
### === Load Data === ##
# print("Loading lazy frames...")
#
# stores = pl.scan_parquet(os.path.join(base_dir, "combined/stores.parquet"))
# rms = pl.scan_parquet(os.path.join(base_dir, "combined/rms.parquet"))
# products_extra = pl.scan_parquet(
#    os.path.join(base_dir, "combined/products_extra.parquet")
# )
#
## Stack movement table fragments vertically instead of joining
# movements = pl.concat(
#    [
#        pl.scan_parquet(os.path.join(base_dir, "combined/3612.parquet")),
#        pl.scan_parquet(os.path.join(base_dir, "combined/3603.parquet")),
#    ]
# )
#
# products = pl.scan_parquet(os.path.join(base_dir, "cleaned_products.parquet"))
# brand_var = pl.scan_parquet(os.path.join(base_dir, "brand_variations.parquet"))
#
### === Merge Pipeline === ##
## Anchor on movements -> left join dimensions to preserve transaction records
# merged = (
#    movements.join(products, on="upc", how="left", suffix="_prod")
#    .join(rms, on="upc", how="left", suffix="_rms")
#    .join(stores, on=["store_code_uc", "panel_year"], how="left", suffix="_str")
#    .join(products_extra, on=["upc", "panel_year"], how="left", suffix="_xtra")
#    .join(brand_var, on="brand_code_uc", how="left", suffix="_brand")
# )
#
# print("Merging all tables and streaming to disk (this will take a few minutes)...")
# merged.sink_parquet(final_out)
# print(f"Successfully saved: {final_out}")
#
### === Post-Merge Audit & Diagnostics === ##
# print("\n" + "=" * 60)
# print("RUNNING MERGE AUDIT: RMS DATASET")
# print("=" * 60)
#
# merged_scan = pl.scan_parquet(final_out)
#
## 1. Row Count & Retention Check
# total_merged_rows = merged_scan.select(pl.len()).collect().item()
# anchor_rows = movements.select(pl.len()).collect().item()
# pct_retained = (total_merged_rows / anchor_rows) * 100
#
# print(f"\n[1] Row Count & Key Expansion Audit:")
# print(f"    - Anchor Movement Rows: {anchor_rows:,}")
# print(f"    - Merged File Rows:    {total_merged_rows:,}")
# print(f"    - Row Ratio:           {pct_retained:.2f}%")
#
# if total_merged_rows > anchor_rows:
#    print(
#        "    ⚠️  WARNING: Row explosion detected! Dimension tables contain duplicate keys."
#    )
# elif total_merged_rows < anchor_rows:
#    print(
#        "    ⚠️  WARNING: Row loss detected! Check if any inner joins dropped sales records."
#    )
#
## 2. Dimension Match Rates (Null Audits on Key Dimension Fields)
# print("\n[2] Dimension Table Match Rates:")
# match_cols = {
#    "products (brand_descr)": "brand_descr",
#    "stores (retailer_code)": "retailer_code",
#    "products_extra (flavor_descr)": "flavor_descr",
#    "brand_var": "brand_code_uc_brand"
#    if "brand_code_uc_brand" in merged_scan.collect_schema().names()
#    else "brand_code_uc",
# }
#
# for name, col in match_cols.items():
#    if col in merged_scan.collect_schema().names():
#        matched_count = (
#            merged_scan.filter(pl.col(col).is_not_null())
#            .select(pl.len())
#            .collect()
#            .item()
#        )
#        match_pct = (matched_count / total_merged_rows) * 100
#        print(f"    - {name:<30}: {matched_count:,} matched ({match_pct:.2f}%)")
#
## 3. Primary Key Uniqueness / Duplicate Check
# print("\n[3] Sales Line Key Uniqueness:")
# if all(
#    k in merged_scan.collect_schema().names()
#    for k in ["store_code_uc", "upc", "week_end"]
# ):
#    unique_sales = (
#        merged_scan.select(pl.struct(["store_code_uc", "upc", "week_end"]).n_unique())
#        .collect()
#        .item()
#    )
#    dup_records = total_merged_rows - unique_sales
#    print(f"    - Duplicate (store + upc + week_end) records: {dup_records:,}")
#
## 4. File Output Integrity
# file_size_gb = os.path.getsize(final_out) / (1024**3)
# schema_cols = merged_scan.collect_schema().names()
#
# print("\n[4] Output File Integrity:")
# print(f"    - Total Columns: {len(schema_cols)}")
# print(f"    - File Size:     {file_size_gb:.2f} GB")
# print("=" * 60)
# print("AUDIT COMPLETE.")
