import os
import polars as pl

base_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
final_out = os.path.join(base_dir, "merged_rms.parquet")

## === Load Data === ##

stores = pl.scan_parquet(os.path.join(base_dir, "combined/cleaned_stores.parquet"))
rms = pl.scan_parquet(os.path.join(base_dir, "combined/rms.parquet"))
products_extra = pl.scan_parquet(
    os.path.join(base_dir, "combined/cleaned_products_extra.parquet")
)
movements = pl.scan_parquet(os.path.join(base_dir, "combined/cleaned_3612")).join(
    pl.scan_parquet(os.path.join(base_dir, "combined/cleaned_3603")),
    on=["upc", "price"],
    how="inner",
)
products = pl.scan_parquet(os.path.join(base_dir, "cleaned_products.parquet"))
brand_var = pl.scan_parquet(os.path.join(base_dir, "cleaned_brand_variations.parquet"))

## === Merge === ##
merged = (
    movements.join(products, on="upc", how="inner", suffix="_prod")
    .join(rms, on="upc", how="left", suffix="_rms")
    .join(stores, on=["store_code_uc", "panel_year"], how="left", suffix="_str")
    .join(products_extra, on="upc", suffix="_xtra")
    .join(brand_var, on="brand", suffix="_brand")
)
print("Merging all tables and streaming...")
merged.sink_parquet(final_out)
print(f"Successfully saved: {final_out}")
