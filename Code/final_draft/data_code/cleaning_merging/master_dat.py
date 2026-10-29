import os
import glob
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "yogurt_estimation_sample.parquet")
temp_dir = os.path.join(input_dir, "temp_years")
os.makedirs(temp_dir, exist_ok=True)

composite_keys = ["panel_year", "week_end", "upc", "store_code_uc"]

# 1. Load RMS with explicit column projection (ONLY load what you need)
print("[STEP 1/3] Materializing unique Yogurt RMS keys...", flush=True)

# Define exact RMS columns needed (join keys + filter col + price/movement cols)
rms_needed_cols = composite_keys + ["product_module_code", "price"]

rms_yogurt = (
    pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))
    # Select ONLY required columns up front (drops ~85% of memory loading)
    .select([c for c in rms_needed_cols])
    # Filter immediately
    .filter(
        pl.col("product_module_code").is_in([3603, 3612])
    )  # Update to your yogurt code
    # Deduplicate on composite keys
    .unique(subset=composite_keys, keep="first")
    .collect()
)

print(f"--> Yogurt RMS loaded successfully: {len(rms_yogurt):,} rows.", flush=True)

# 2. Define panel years directly
panel_years = list(range(2006, 2015))

print(
    f"[STEP 2/3] Processing panel year-by-year across {len(panel_years)} years...",
    flush=True,
)

for idx, yr in enumerate(panel_years, 1):
    yr_out = os.path.join(temp_dir, f"yogurt_{yr}.parquet")
    print(
        f"  --> [{idx}/{len(panel_years)}] Joining & streaming Year {yr}...", flush=True
    )

    # Scan HMS
    hms_yr = pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet")).filter(
        pl.col("panel_year") == yr
    )

    # Filter in-memory RMS to current year
    rms_yr = rms_yogurt.filter(pl.col("panel_year") == yr)

    # Stream inner join
    hms_yr.join(
        rms_yr.lazy(), on=composite_keys, how="inner", suffix="_rms"
    ).sink_parquet(yr_out)

    print(f"      [DONE] Saved Year {yr} chunk.", flush=True)

# 3. Combine yearly chunks
print("\n[STEP 3/3] Combining yearly files into final estimation sample...", flush=True)
pl.scan_parquet(os.path.join(temp_dir, "yogurt_*.parquet")).sink_parquet(final_out)

# Cleanup
for f in glob.glob(os.path.join(temp_dir, "yogurt_*.parquet")):
    os.remove(f)
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

print(f"\n[SUCCESS] Saved 10-year Yogurt estimation sample to: {final_out}", flush=True)
