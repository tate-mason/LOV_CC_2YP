import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")

print("Setting up lazy scan for 10-year Yogurt sample...", flush=True)

# 1. Define composite join keys
composite_keys = ["panel_year", "week_end", "upc", "store_code_uc"]

# 2. Filter RMS to Yogurt and deduplicate within each week/year/store/UPC combination
RMS_yogurt = (
    pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))
    .filter(pl.col("product_module_code").is_in([3612, 3603]))
    .unique(subset=composite_keys, keep="first")
)

# 3. Load HMS Panel
HMS = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet"))

# 4. Inner Join: Keeps all matching Yogurt purchases across the full 10-year panel
print("Executing inner join across full 10-year panel...", flush=True)
yogurt_sample = HMS.join(RMS_yogurt, on=composite_keys, how="inner", suffix="_rms")

# 5. Stream output to disk
print(f"Streaming estimation sample to {final_out}...", flush=True)
yogurt_sample.sink_parquet(final_out)
print(
    f"[SUCCESS] 10-year Yogurt estimation sample saved to: {final_out}",
    flush=True,
)

## === ESTIMATION SAMPLE AUDIT DIAGNOSTICS === ##
print("\n" + "=" * 60, flush=True)
print("RUNNING ESTIMATION SAMPLE AUDIT", flush=True)
print("=" * 60, flush=True)

# Scan saved estimation sample
sample_scan = pl.scan_parquet(final_out)

# 1. Overall Row and Unique Key Statistics
total_sample_rows = sample_scan.select(pl.len()).collect().item()
unique_households = (
    sample_scan.select(pl.col("household_code").n_unique()).collect().item()
)
unique_upcs = sample_scan.select(pl.col("upc").n_unique()).collect().item()
unique_stores = sample_scan.select(pl.col("store_code_uc").n_unique()).collect().item()

print(f"\n[1] Record Counts & Coverage:", flush=True)
print(f"    - Total Estimation Sample Rows: {total_sample_rows:,}", flush=True)
print(f"    - Unique Households Included:    {unique_households:,}", flush=True)
print(f"    - Unique Yogurt UPCs Matched:    {unique_upcs:,}", flush=True)
print(f"    - Unique Retail Stores Matched:  {unique_stores:,}", flush=True)

# 2. Key Join Column Null Check
print(f"\n[2] Join Key Integrity (Null Check):", flush=True)
null_counts = (
    sample_scan.select([pl.col(k).null_count().alias(k) for k in composite_keys])
    .collect()
    .to_dict(as_series=False)
)

for key, count in null_counts.items():
    null_pct = (count[0] / total_sample_rows) * 100 if total_sample_rows else 0
    print(f"    - {key}: {count[0]:,} nulls ({null_pct:.2f}%)", flush=True)

# 3. Output Dataset Metadata
file_size_mb = os.path.getsize(final_out) / (1024**2)
schema_cols = sample_scan.schema.names()

print(f"\n[3] File Summary:", flush=True)
print(f"    - Total Schema Columns: {len(schema_cols)}", flush=True)
print(f"    - Disk Size:            {file_size_mb:.2f} MB", flush=True)
print(f"    - Output Path:          {final_out}", flush=True)
print("=" * 60, flush=True)
print("AUDIT COMPLETE - Yogurt Estimation Sample Verified.", flush=True)
