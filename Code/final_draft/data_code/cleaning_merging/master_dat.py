import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")

print("Setting up lazy scan...")

# 1. Load and filter RMS to Yogurt first
# Replace 'product_module_descr' with your exact RMS product column (e.g., product_module_code == 1202)
RMS_yogurt = pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet")).filter(
    pl.col("product_module_code").is_in([3612, 3603])
)

# 2. Load HMS Panel
HMS = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet"))

# 3. Inner Join: Keeps ONLY shared UPC/store/week observations
print("Executing inner join on shared Yogurt keys...")
yogurt_sample = HMS.join(
    RMS_yogurt,
    on=["panel_year", "week_end", "upc", "store_code_uc"],
    how="inner",
    suffix="_rms",
)

# 4. Stream estimation sample to disk
print(f"Streaming estimation sample to {final_out}...")
yogurt_sample.sink_parquet(final_out)
print(f"[SUCCESS] Saved Yogurt estimation sample to: {final_out}")

## === Estimation Sample Audit === ##
print("\n" + "=" * 60)
print("RUNNING ESTIMATION SAMPLE AUDIT")
print("=" * 60)

sample_scan = pl.scan_parquet(final_out)
total_sample_rows = sample_scan.select(pl.len()).collect().item()
unique_upcs = sample_scan.select(pl.col("upc").n_unique()).collect().item()
file_size_mb = os.path.getsize(final_out) / (1024**2)

print(f"    - Total Sample Rows:  {total_sample_rows:,}")
print(f"    - Unique Yogurt UPCs: {unique_upcs:,}")
print(f"    - Output File Size:   {file_size_mb:.2f} MB")
print(f"    - Output Path:        {final_out}")
print("=" * 60)
