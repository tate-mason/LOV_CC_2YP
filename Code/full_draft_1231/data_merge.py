"""Merge HMS and RMS by year, with diagnostics for large SLURM jobs.

Run: python -u data_merge_diagnostics.py
The RMS selection stage writes one numeric row ID per distinct join key.
This avoids wide unique(), which triggered a Polars first/last reducer panic.
The audit checks every RMS value column using two row fingerprints.
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
ROW_ID = "__rms_row_id"


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
        # Assigned before filtering; both branches of the later semi join use
        # the same source row IDs. The source has fewer than 2**32 rows.
        .with_row_index(ROW_ID)
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


def audit_merge(year, panel_rows, output_path, ids_path, retail_year, value_columns):
    """Check actual output match coverage and conflicting RMS values."""
    written = pl.scan_parquet(output_path)
    selected = pl.scan_parquet(ids_path)

    # A left join retains unmatched HMS rows. Count real key matches separately.
    matched = count(
        f"{year} HMS rows with an RMS key match",
        written.select(JOIN_KEYS).join(
            selected.select(JOIN_KEYS), on=JOIN_KEYS, how="semi"
        ),
    )
    log(
        f"{year} RMS match rate: {matched / panel_rows:.1%} "
        f"({panel_rows - matched:,} unmatched HMS rows)"
        if panel_rows
        else f"{year} RMS match rate: n/a (no HMS rows)"
    )

    # Restrict the large RMS scan to keys appearing in the actual HMS output.
    # Hash ALL RMS value columns, then check whether fingerprints differ within
    # each key. Two seeds make an undetected hash collision extremely unlikely.
    # Min/max of two UInt64 hashes avoids aggregating the wide value rows.
    output_keys = written.select(JOIN_KEYS).unique(subset=JOIN_KEYS)
    values = pl.struct(value_columns)
    grouped = (
        retail_year.select(
            JOIN_KEYS
            + [
                values.hash(seed=0).alias("_rms_hash_0"),
                values.hash(seed=1).alias("_rms_hash_1"),
            ]
        )
        .join(output_keys, on=JOIN_KEYS, how="semi")
        .group_by(JOIN_KEYS)
        .agg(
            pl.len().alias("rms_rows"),
            pl.col("_rms_hash_0").min().alias("hash_0_min"),
            pl.col("_rms_hash_0").max().alias("hash_0_max"),
            pl.col("_rms_hash_1").min().alias("hash_1_min"),
            pl.col("_rms_hash_1").max().alias("hash_1_max"),
        )
        .with_columns(
            (
                (pl.col("hash_0_min") != pl.col("hash_0_max"))
                | (pl.col("hash_1_min") != pl.col("hash_1_max"))
            ).alias("values_disagree")
        )
    )

    log(f"START {year} RMS duplicate-value audit across {len(value_columns)} columns")
    summary = grouped.select(
        pl.len().alias("matched_rms_keys"),
        (pl.col("rms_rows") > 1).sum().alias("keys_with_multiple_rms_rows"),
        pl.col("values_disagree").sum().alias("keys_with_conflicting_values"),
    ).collect(engine="streaming")
    log(f"DONE  {year} RMS duplicate-value audit:\n{summary}")

    conflicting = grouped.filter(pl.col("values_disagree")).select(
        JOIN_KEYS + ["rms_rows"]
    )
    log(f"START {year} sample of up to 10 conflicting RMS keys")
    log(str(conflicting.head(10).collect(engine="streaming")))


def main():
    log(f"Polars version: {pl.__version__}; index type: {pl.get_index_type()}")
    log(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'none')}")
    log(f"SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE', 'unset')} MB")
    log(f"POLARS_MAX_THREADS: {os.environ['POLARS_MAX_THREADS']}")
    log(f"HMS: {HMS_PATH}")
    log(f"RMS: {RMS_PATH}")

    # These counts are on the original files; unlike a full collect they only
    # return one number each. They help assess whether rt64 is worth testing.
    count("HMS source", pl.scan_parquet(HMS_PATH))
    count("RMS source", pl.scan_parquet(RMS_PATH))

    panel = build_panel()
    retail = build_retail()
    value_columns = [
        column
        for column in retail.collect_schema().names()
        if column not in JOIN_KEYS and column != ROW_ID
    ]
    if not value_columns:
        raise ValueError("RMS has no non-key value columns to audit")
    log(f"RMS value columns audited ({len(value_columns)}): {', '.join(value_columns)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for year in YEARS:
        log(f"========== YEAR {year} ==========")
        panel_year = panel.filter(pl.col("week_end").dt.year() == year)
        retail_year = retail.filter(pl.col("week_end").dt.year() == year)

        panel_rows = count(f"{year} filtered HMS", panel_year)
        retail_rows = count(f"{year} parsed RMS", retail_year)

        # Only group the keys and a numeric row ID. min() chooses a consistent
        # source row without aggregating all the wide RMS value columns.
        ids_path = os.path.join(OUTPUT_DIR, f"scanner_panel_{year}.rms_ids.parquet")
        ids_temp_path = ids_path + ".incomplete"
        chosen_ids = (
            retail_year.select(JOIN_KEYS + [ROW_ID])
            .group_by(JOIN_KEYS)
            .agg(pl.col(ROW_ID).min().alias(ROW_ID))
        )
        log(f"START {year} selecting one RMS row ID per key -> {ids_temp_path}")
        chosen_ids.sink_parquet(ids_temp_path, engine="streaming")
        os.replace(ids_temp_path, ids_path)
        distinct_keys = count(f"{year} selected RMS keys", pl.scan_parquet(ids_path))
        log(
            f"{year} RMS duplicate key rows: {retail_rows - distinct_keys:,} of {retail_rows:,}"
        )

        # Semi join recovers ALL columns of exactly the chosen RMS rows.
        retail_unique = retail_year.join(
            pl.scan_parquet(ids_path).select(ROW_ID), on=ROW_ID, how="semi"
        ).drop(ROW_ID)
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
        audit_merge(year, panel_rows, output_path, ids_path, retail_year, value_columns)
        os.remove(ids_path)

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
