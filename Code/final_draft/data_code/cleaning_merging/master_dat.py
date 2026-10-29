import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")

composite_keys = ["panel_year", "week_end", "upc", "store_code_uc"]

# 1. Load and materialize ONLY the filtered Yogurt RMS table (~1-2 GB RAM max)
print("Loading and materializing unique Yogurt RMS keys...", flush=True)
rms_yogurt = (
    pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))
    .filter(
        pl.col("product_module_code").is_in([3612, 3603])
    )  # Use your exact yogurt code
    .unique(subset=composite_keys, keep="first")
    .collect()
)
print(f"Yogurt RMS loaded: {len(rms_yogurt):,} rows.", flush=True)

# 2. Get all distinct years available in HMS
years = (
    pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet"))
    .select(pl.col("panel_year").unique())
    .collect()
    .get_column("panel_year")
    .sort()
    .to_list()
)

print(f"Processing panel year-by-year across years: {years}", flush=True)

# Remove old output file if it exists
if os.path.exists(final_out):
    os.remove(final_out)

# 3. Process each year sequentially to cap RAM usage
for idx, yr in enumerate(years):
    print(f"Processing Year {yr} ({idx + 1}/{len(years)})...", flush=True)

    # Filter HMS lazy frame to current year
    hms_yr = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet")).filter(
        pl.col("panel_year") == yr
    )

    # Filter in-memory RMS to current year
    rms_yr = rms_yogurt.filter(pl.col("panel_year") == yr)

    # Inner join for this year
    sample_yr = hms_yr.join(
        rms_yr.lazy(), on=composite_keys, how="inner", suffix="_rms"
    ).collect()

    # Append to parquet (write first year, then append subsequently)
    if idx == 0:
        sample_yr.write_parquet(final_out)
    else:
        # Re-open existing parquet and append via pyarrow/polars sink
        existing = pl.read_parquet(final_out)
        pl.concat([existing, sample_yr]).write_parquet(final_out)

print(f"\n[SUCCESS] Saved 10-year Yogurt estimation sample to: {final_out}", flush=True)

## === AUDIT DIAGNOSTICS === ##
print("\n" + "=" * 60, flush=True)
print("RUNNING ESTIMATION SAMPLE AUDIT", flush=True)
print("=" * 60, flush=True)

sample_scan = pl.scan_parquet(final_out)
total_sample_rows = sample_scan.select(pl.len()).collect().item()
unique_upcs = sample_scan.select(pl.col("upc").n_unique()).collect().item()
file_size_mb = os.path.getsize(final_out) / (1024**2)

print(f"    - Total Estimation Rows: {total_sample_rows:,}", flush=True)
print(f"    - Unique Yogurt UPCs:    {unique_upcs:,}", flush=True)
print(f"    - File Size on Disk:     {file_size_mb:.2f} MB", flush=True)
print("=" * 60, flush=True)
