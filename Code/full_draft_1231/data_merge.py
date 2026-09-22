"""Merge HMS and RMS by year, with diagnostics for large SLURM jobs.

Run: python -u data_merge_diagnostics.py
Optional: RUN_KEY_AUDIT=0 python -u data_merge_diagnostics.py

The key audit counts distinct RMS join keys and can itself be expensive.
Set RUN_KEY_AUDIT=0 to proceed directly to the merge if that audit fails.
"""

import os
import time
import traceback

# Set before importing Polars. Fewer threads can lower peak memory; override
# POLARS_MAX_THREADS explicitly if the full SLURM CPU allocation is too much.
os.environ.setdefault(
    "POLARS_MAX_THREADS", os.environ.get("SLURM_CPUS_PER_TASK") or "8"
)
os.environ.setdefault("RUST_BACKTRACE", "1")

import polars as pl


HMS_PATH = (
    "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/full_panel.parquet"
)
RMS_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets/full_retail.parquet"
OUTPUT_DIR = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts"
YEARS = (2022, 2023, 2024)
JOIN_KEYS = ["week_end", "store_code_uc", "upc"]
RUN_KEY_AUDIT = os.environ.get("RUN_KEY_AUDIT", "1") == "1"


def log(message):
    print(message, flush=True)


def count(label, frame):
    """Execute a narrow, scalar count, never collecting the underlying rows."""
    started = time.monotonic()
    log(f"START {label}")
    result = frame.select(pl.len().alias("n")).collect(engine="streaming")
    n = result.item(0, 0)
    log(f"DONE  {label}: {n:,} rows ({time.monotonic() - started:.1f}s)")
    return n


def build_panel():
    panel = (
        pl.scan_parquet(HMS_PATH)
        .with_columns(pl.all().name.to_lowercase())
        .with_columns(
            pl.col("product_module_code_hms").cast(pl.Utf8).str.strip_chars(),
            pl.col("size1_amount_hms").cast(pl.Float64, strict=False),
            pl.col("size1_unit_hms").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("household_size").cast(pl.Int64, strict=False),
            pl.col("quantity").cast(pl.Int64, strict=False),
            pl.col("deal_flag_uc").cast(pl.Int64, strict=False),
            pl.col("household_code").cast(pl.Int64, strict=False),
            pl.col("store_code_uc").cast(pl.Int64, strict=False),
            pl.col("upc").cast(pl.Int64, strict=False),
        )
    )

    # Activity is measured in the entire original HMS panel, before restrictions.
    active_hhs = (
        panel.select("household_code", "trip_code_uc")
        .group_by("household_code")
        .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
        .filter(pl.col("total_trips") > 2)
        .select("household_code")
    )

    panel = (
        panel.join(active_hhs, on="household_code", how="semi")
        .filter(pl.col("household_size") == 1)
        .filter(pl.col("size1_unit_hms") == "OZ")
        .filter(pl.col("size1_amount_hms").is_between(5000, 8001))
        .with_columns(
            pl.col("purchase_date")
            .cast(pl.Utf8)
            .str.replace_all("-", "")
            .str.to_date("%Y%m%d", strict=False)
            .alias("parsed_date"),
            pl.col("household_income").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("male_head_age").cast(pl.Float64, strict=False).replace(0, None),
            pl.col("female_head_age").cast(pl.Float64, strict=False).replace(0, None),
        )
        .with_columns(
            # ISO weekday is Monday=1 through Sunday=7. Saturday=6;
            # (13 - weekday) % 7 maps Saturday to 0 and Sunday to 6.
            (
                pl.col("parsed_date")
                + pl.duration(days=(13 - pl.col("parsed_date").dt.weekday()) % 7)
            )
            .cast(pl.Datetime("ms"))
            .alias("week_end"),
            pl.coalesce("male_head_age", "female_head_age").alias("head_age"),
        )
        .with_columns(
            pl.col("flavor").cast(pl.Utf8).fill_null("").alias("flavor_str"),
            pl.col("flavor_cd").cast(pl.Int64, strict=False).fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("flavor_str").str.contains("(?i)berry"))
            .then(1)
            .when(pl.col("flavor_cd").is_in([67676592, 66987057]))
            .then(2)
            .otherwise(0)
            .alias("flavor"),
            pl.when(
                pl.col("product_module_code_hms").is_in(["3603", "3612"])
                & (pl.col("quantity") > 0)
            )
            .then(1)
            .otherwise(0)
            .alias("yogurt_purchase"),
        )
    )
    return panel


def build_retail():
    return (
        pl.scan_parquet(RMS_PATH)
        .with_columns(pl.all().name.to_lowercase())
        .with_columns(
            pl.col("week_end")
            .str.to_datetime("%Y-%m-%d", strict=False)
            .dt.cast_time_unit("ms"),
            pl.col("store_code_uc").cast(pl.Int64, strict=False),
            pl.col("upc").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("week_end").is_not_null())
    )


def main():
    log(f"Polars version: {pl.__version__}; index type: {pl.get_index_type()}")
    log(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'none')}")
    log(f"SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE', 'unset')} MB")
    log(f"POLARS_MAX_THREADS: {os.environ['POLARS_MAX_THREADS']}")
    log(f"RUN_KEY_AUDIT: {RUN_KEY_AUDIT}")
    log(f"HMS: {HMS_PATH}")
    log(f"RMS: {RMS_PATH}")

    # These counts are on the original files; unlike a full collect they only
    # return one number each. They help assess whether rt64 is worth testing.
    count("HMS source", pl.scan_parquet(HMS_PATH))
    count("RMS source", pl.scan_parquet(RMS_PATH))

    panel = build_panel()
    retail = build_retail()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for year in YEARS:
        log(f"========== YEAR {year} ==========")
        panel_year = panel.filter(pl.col("week_end").dt.year() == year)
        retail_year = retail.filter(pl.col("week_end").dt.year() == year)

        panel_rows = count(f"{year} filtered HMS", panel_year)
        retail_rows = count(f"{year} parsed RMS", retail_year)

        if RUN_KEY_AUDIT:
            # Project ONLY the three keys. This still builds distinct-key state:
            # if it is too large, rerun with RUN_KEY_AUDIT=0.
            distinct_keys = count(
                f"{year} distinct RMS join keys",
                retail_year.select(JOIN_KEYS).unique(subset=JOIN_KEYS),
            )
            log(
                f"{year} RMS duplicate key rows: "
                f"{retail_rows - distinct_keys:,} of {retail_rows:,}"
            )
            if distinct_keys >= 2**32 and pl.get_index_type() == pl.UInt32:
                log(
                    f"{year} WARNING: distinct keys exceed UInt32 index range; test polars[rt64]."
                )

        # Do not drop RMS variables: output retains the original set of columns.
        # The yearly restriction precedes the wide unique and join.
        retail_unique = retail_year.unique(subset=JOIN_KEYS)
        merged = panel_year.join(retail_unique, on=JOIN_KEYS, how="left")
        output_path = os.path.join(OUTPUT_DIR, f"scanner_panel_{year}.parquet")
        temp_path = output_path + ".incomplete"

        log(f"START {year} streaming join and parquet sink -> {temp_path}")
        started = time.monotonic()
        # An abort leaves only .incomplete; a completed year is not overwritten
        # until its new output file has been fully written.
        merged.sink_parquet(temp_path, engine="streaming")
        os.replace(temp_path, output_path)
        log(f"DONE  {year} parquet sink ({time.monotonic() - started:.1f}s)")

        output_rows = count(f"{year} written output", pl.scan_parquet(output_path))
        if output_rows != panel_rows:
            raise RuntimeError(
                f"{year}: output has {output_rows:,} rows but HMS has {panel_rows:,}; "
                "inspect the join keys and RMS deduplication"
            )

    # Continue analysis here. The combined scan remains lazy.
    combined_lazy = pl.scan_parquet(os.path.join(OUTPUT_DIR, "scanner_panel_*.parquet"))
    log("MERGE COMPLETE; combined_lazy is ready for downstream analysis")
    return combined_lazy


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("PYTHON EXCEPTION (the preceding START line identifies the stage)")
        traceback.print_exc()
        raise
