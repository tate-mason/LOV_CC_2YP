import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")

composite_keys = ["panel_year", "week_end", "upc", "store_code_uc"]

print("1. Loading and materializing unique Yogurt RMS keys...", flush=True)

# 1. Materialize ONLY the filtered, deduplicated Yogurt RMS table in memory (~1-2 GB max)
# Replace product_module_code == 1202 with your exact filter condition
rms_yogurt = (
    pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))
    .filter(
        pl.col("product_module_code").is_in([3612, 3603])
    )  # Use your specific filter code here
    .unique(subset=composite_keys, keep="first")
    .collect(
        streaming=False
    )  # Materialize pure Yogurt subset to prevent full-graph blowup
)

print(
    f"Yogurt RMS materialized with {len(rms_yogurt):,} rows. Joining against HMS...",
    flush=True,
)

# 2. Lazy scan your already-pruned HMS panel
HMS = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet"))

# 3. Inner join against the in-memory RMS Yogurt table
yogurt_sample = HMS.join(
    rms_yogurt.lazy(), on=composite_keys, how="inner", suffix="_rms"
)

# 4. Stream final output to disk
print(f"Streaming estimation sample to {final_out}...", flush=True)
yogurt_sample.sink_parquet(final_out)
print(f"[SUCCESS] Saved Yogurt estimation sample to: {final_out}", flush=True)

## === ESTIMATION SAMPLE AUDIT DIAGNOSTICS === ##
print("\n" + "=" * 60, flush=True)
print("RUNNING ESTIMATION SAMPLE AUDIT", flush=True)
print("=" * 60, flush=True)

sample_scan = pl.scan_parquet(final_out)

total_sample_rows = sample_scan.select(pl.len()).collect().item()
unique_households = (
    sample_scan.select(pl.col("household_code").n_unique()).collect().item()
    if "household_code" in sample_scan.schema.names()
    else "N/A"
)
unique_upcs = sample_scan.select(pl.col("upc").n_unique()).collect().item()
unique_stores = sample_scan.select(pl.col("store_code_uc").n_unique()).collect().item()

print(f"\n[1] Record Counts & Coverage:", flush=True)
print(f"    - Total Estimation Sample Rows: {total_sample_rows:,}", flush=True)
print(f"    - Unique Households Included:    {unique_households}", flush=True)
print(f"    - Unique Yogurt UPCs Matched:    {unique_upcs:,}", flush=True)
print(f"    - Unique Retail Stores Matched:  {unique_stores:,}", flush=True)

print(f"\n[2] Join Key Integrity (Null Check):", flush=True)
null_counts = (
    sample_scan.select([pl.col(k).null_count().alias(k) for k in composite_keys])
    .collect()
    .to_dict(as_series=False)
)

for key, count in null_counts.items():
    null_pct = (count[0] / total_sample_rows) * 100 if total_sample_rows else 0
    print(f"    - {key}: {count[0]:,} nulls ({null_pct:.2f}%)", flush=True)

file_size_mb = os.path.getsize(final_out) / (1024**2)
schema_cols = sample_scan.schema.names()

print(f"\n[3] File Summary:", flush=True)
print(f"    - Total Schema Columns: {len(schema_cols)}", flush=True)
print(f"    - Disk Size:            {file_size_mb:.2f} MB", flush=True)
print(f"    - Output Path:          {final_out}", flush=True)
print("=" * 60, flush=True)
print("AUDIT COMPLETE - Yogurt Estimation Sample Verified.", flush=True)
