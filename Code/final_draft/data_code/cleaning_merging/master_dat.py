import os
import glob
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")
temp_dir = os.path.join(input_dir, "temp_years")
os.makedirs(temp_dir, exist_ok=True)

composite_keys = ["panel_year", "week_end", "upc", "store_code_uc"]

# 1. Materialize ONLY filtered Yogurt RMS in memory (~1 GB RAM)
print("[STEP 1/3] Materializing unique Yogurt RMS keys...", flush=True)
rms_yogurt = (
    pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))
    .filter(
        pl.col("product_module_code").is_in([3603, 3612])
    )  # Update to your yogurt module code
    .unique(subset=composite_keys, keep="first")
    .collect()
)
print(f"--> Yogurt RMS loaded successfully: {len(rms_yogurt):,} rows.", flush=True)

# 2. Define panel years directly (Avoids scanning 750M rows just for distinct year values)
# Adjust start and end years to match your panel range
panel_years = list(range(2006, 2015))

print(
    f"[STEP 2/3] Processing panel year-by-year across {len(panel_years)} years...",
    flush=True,
)

temp_files = []

for idx, yr in enumerate(panel_years, 1):
    yr_out = os.path.join(temp_dir, f"yogurt_{yr}.parquet")
    temp_files.append(yr_out)

    print(
        f"  --> [{idx}/{len(panel_years)}] Joining & streaming Year {yr}...", flush=True
    )

    # Lazy scan HMS, filter to target year
    hms_yr = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet")).filter(
        pl.col("panel_year") == yr
    )

    # Filter in-memory RMS to target year
    rms_yr = rms_yogurt.filter(pl.col("panel_year") == yr)

    # Stream inner join directly to temporary year file (keeps RAM strictly < 10 GB)
    hms_yr.join(
        rms_yr.lazy(), on=composite_keys, how="inner", suffix="_rms"
    ).sink_parquet(yr_out)

    print(f"      [DONE] Saved Year {yr} chunk.", flush=True)

# 3. Stitch all yearly chunks into the final estimation sample
print("\n[STEP 3/3] Combining yearly files into final estimation sample...", flush=True)
pl.scan_parquet(os.path.join(temp_dir, "yogurt_*.parquet")).sink_parquet(final_out)

# Cleanup temp folder
for f in glob.glob(os.path.join(temp_dir, "yogurt_*.parquet")):
    os.remove(f)
os.rmdir(temp_dir)

print(f"\n[SUCCESS] Saved 10-year Yogurt estimation sample to: {final_out}", flush=True)

## === ESTIMATION SAMPLE AUDIT === ##
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
print("AUDIT COMPLETE - Process finished cleanly.", flush=True)
