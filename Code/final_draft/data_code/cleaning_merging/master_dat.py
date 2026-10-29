import glob
import os
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
panel_years = list(range(2006, 2016))

print(
    f"[STEP 2/3] Processing panel year-by-year across {len(panel_years)} years...",
    flush=True,
)

for idx, yr in enumerate(panel_years, 1):
    yr_out = os.path.join(temp_dir, f"yogurt_{yr}.parquet")
    print(
        f"  --> [{idx}/{len(panel_years)}] Joining & streaming Year {yr}...",
        flush=True,
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
print(
    "\n[STEP 3/3] Combining yearly files into final estimation sample...",
    flush=True,
)
pl.scan_parquet(os.path.join(temp_dir, "yogurt_*.parquet")).sink_parquet(final_out)

# Cleanup
for f in glob.glob(os.path.join(temp_dir, "yogurt_*.parquet")):
    os.remove(f)
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

print(
    f"\n[SUCCESS] Saved 10-year Yogurt estimation sample to: {final_out}",
    flush=True,
)

# ==============================================================================
# 4. ESTIMATION SAMPLE AUDIT & DIAGNOSTICS
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("RUNNING ESTIMATION SAMPLE AUDIT DIAGNOSTICS", flush=True)
print("=" * 70, flush=True)

# Scan final dataset lazily for memory-efficient auditing
sample_scan = pl.scan_parquet(final_out)
schema = sample_scan.collect_schema()
col_names = schema.names()

# --- DIAGNOSTIC 1: Sample Volume & Coverage ---
total_rows = sample_scan.select(pl.len()).collect().item()
unique_upcs = sample_scan.select(pl.col("upc").n_unique()).collect().item()
unique_stores = sample_scan.select(pl.col("store_code_uc").n_unique()).collect().item()

# Check household column dynamically depending on HMS schema
hh_col = next((c for c in ["household_code", "hh_id"] if c in col_names), None)
unique_hhs = (
    sample_scan.select(pl.col(hh_col).n_unique()).collect().item() if hh_col else "N/A"
)

print("\n[1] RECORD COUNTS & ENTITY COVERAGE:", flush=True)
print(f"    - Total Estimation Rows:        {total_rows:,}", flush=True)
print(
    f"    - Unique Households Matched:    {unique_hhs:,}"
    if isinstance(unique_hhs, int)
    else f"    - Unique Households Matched:    {unique_hhs}",
    flush=True,
)
print(f"    - Unique Yogurt UPCs Matched:   {unique_upcs:,}", flush=True)
print(f"    - Unique Retail Stores Matched: {unique_stores:,}", flush=True)

# --- DIAGNOSTIC 2: Join Key Integrity & Null Check ---
print("\n[2] JOIN KEY INTEGRITY (NULL CHECK):", flush=True)
null_counts = (
    sample_scan.select([pl.col(k).null_count().alias(k) for k in composite_keys])
    .collect()
    .to_dict(as_series=False)
)

for key, count in null_counts.items():
    null_pct = (count[0] / total_rows) * 100 if total_rows > 0 else 0
    print(f"    - Nulls in '{key}': {count[0]:,} ({null_pct:.2f}%)", flush=True)

# --- DIAGNOSTIC 3: Time Series Coverage by Panel Year ---
print("\n[3] PANEL YEAR DISTRIBUTION:", flush=True)
year_counts = (
    sample_scan.group_by("panel_year")
    .agg(pl.len().alias("count"))
    .sort("panel_year")
    .collect()
)

for row in year_counts.iter_rows(named=True):
    print(
        f"    - Year {row['panel_year']}: {row['count']:,} observations",
        flush=True,
    )

# --- DIAGNOSTIC 4: Price Column Summary (Price Non-Zero / Non-Null Check) ---
price_col = next((c for c in ["price", "price_rms"] if c in col_names), None)
if price_col:
    price_stats = (
        sample_scan.select(
            [
                pl.col(price_col).min().alias("min_price"),
                pl.col(price_col).mean().alias("mean_price"),
                pl.col(price_col).median().alias("median_price"),
                pl.col(price_col).max().alias("max_price"),
                pl.col(price_col).null_count().alias("null_price_count"),
                (pl.col(price_col) <= 0).sum().alias("zero_price_count"),
            ]
        )
        .collect()
        .to_dict(as_series=False)
    )

    print(
        f"\n[4] PRICE COLUMN DIAGNOSTICS ('{price_col}'):",
        flush=True,
    )
    print(f"    - Min Price:    ${price_stats['min_price'][0]:.2f}", flush=True)
    print(f"    - Mean Price:   ${price_stats['mean_price'][0]:.2f}", flush=True)
    print(
        f"    - Median Price: ${price_stats['median_price'][0]:.2f}",
        flush=True,
    )
    print(f"    - Max Price:    ${price_stats['max_price'][0]:.2f}", flush=True)
    print(
        f"    - Null Prices:  {price_stats['null_price_count'][0]:,}",
        flush=True,
    )
    print(
        f"    - Zero/Negative:{price_stats['zero_price_count'][0]:,}",
        flush=True,
    )

# --- DIAGNOSTIC 5: Output Dataset File Metadata ---
file_size_mb = os.path.getsize(final_out) / (1024**2)

print("\n[5] OUTPUT FILE SUMMARY:", flush=True)
print(f"    - Total Schema Columns: {len(col_names)}", flush=True)
print(f"    - File Size on Disk:    {file_size_mb:.2f} MB", flush=True)
print(f"    - Output Location:      {final_out}", flush=True)

print("=" * 70, flush=True)
print("AUDIT COMPLETE — Yogurt estimation sample verified.", flush=True)
