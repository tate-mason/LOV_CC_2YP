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
    .join(retailers, on="store_code_uc", how="left")
)
print("Merging all tables and streaming...")
merged.sink_parquet(final_out)
print(f"Successfully saved: {final_out}")
