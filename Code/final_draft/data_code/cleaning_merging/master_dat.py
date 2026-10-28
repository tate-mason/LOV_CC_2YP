import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/Data"
final_out = os.path.join(input_dir, "merged_df.parquet")

HMS = (
    pl.scan_parquet(os.path.join(input_dir, "merged_panel.parquet"))
    .with_columns(
        purchase_date=pl.col("purchase_date").cast(pl.String).str.to_date("%Y%m%d")
    )
    .with_columns(
        week_end=pl.col("purchase_date")
        + pl.duration(days=(6 - pl.col("purchase_date").dt.weekday()) % 7)
    )
)
RMS = pl.scan_parquet(os.path.join(input_dir, "merged_rms.parquet"))

merged = HMS.join(
    RMS,
    on=["panel_year", "week_end", "upc"],
    how="left",
)

print("Streaming HMS-RMS merge to disk (this will take a few minutes)...")
merged.sink_parquet(final_out)
print(f"\n[SUCCESS] Saved merged file to: {final_out}")

## === Run Merge Audit Diagnostics === ##
print("\n" + "=" * 60)
print("RUNNING MERGE AUDIT: HMS x RMS DATASETS")
print("=" * 60)

# 1. Scan output dataset lazily from disk
merged_scan = pl.scan_parquet(final_out)

# 2. Total Row Count Check & HMS Anchor Retention
total_merged_rows = merged_scan.select(pl.len()).collect().item()
hms_anchor_rows = HMS.select(pl.len()).collect().item()
pct_retained = (total_merged_rows / hms_anchor_rows) * 100

print(f"\n[1] Row Count & Retention Check:")
print(f"    - HMS Anchor Rows:      {hms_anchor_rows:,}")
print(f"    - Merged File Rows:     {total_merged_rows:,}")
print(f"    - Anchor Retention:     {pct_retained:.2f}%")

if total_merged_rows > hms_anchor_rows:
    print(
        "    WARNING: Row explosion detected! RMS may contain duplicate keys for (panel_year, week_end, upc)."
    )

# 3. Match Rate Check (How many HMS purchases matched to RMS store prices)
print("\n[2] RMS Join Match Rate:")
# Pick a column unique to RMS (e.g., 'price' or 'store_code_uc')
rms_cols = [
    c
    for c in RMS.collect_schema().names()
    if c not in ["panel_year", "week_end", "upc"]
]
match_check_col = rms_cols[0] if rms_cols else None

if match_check_col:
    matched_rows = (
        merged_scan.filter(pl.col(match_check_col).is_not_null())
        .select(pl.len())
        .collect()
        .item()
    )
    match_pct = (matched_rows / total_merged_rows) * 100
    print(
        f"    - Matched to RMS ({match_check_col}): {matched_rows:,} ({match_pct:.2f}%)"
    )
    print(
        f"    - Unmatched HMS records: {total_merged_rows - matched_rows:,} ({100 - match_pct:.2f}%)"
    )

# 4. Key Join Columns Null Check
print("\n[3] Key Column Null Diagnostics:")
join_keys = ["panel_year", "week_end", "upc"]
null_counts = (
    merged_scan.select([pl.col(k).null_count().alias(k) for k in join_keys])
    .collect()
    .to_dict(as_series=False)
)

for key, count in null_counts.items():
    null_pct = (count[0] / total_merged_rows) * 100
    print(f"    - {key}: {count[0]:,} nulls ({null_pct:.2f}%)")

# 5. Disk & Column Summary
file_size_gb = os.path.getsize(final_out) / (1024**3)
schema_cols = merged_scan.collect_schema().names()

print("\n[4] Output File Summary:")
print(f"    - Total Columns: {len(schema_cols)}")
print(f"    - Disk Size:     {file_size_gb:.2f} GB")
print(f"    - Output Path:   {final_out}")
print("=" * 60)
print("AUDIT COMPLETE - Merge verified.")
