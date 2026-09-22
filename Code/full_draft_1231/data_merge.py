"""
Data Processing & Summary Statistics Pipeline (Pure Polars Native)
"""

import os

os.environ["POLARS_MAX_THREADS"] = "8"

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HMS_PATH = (
    "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/full_panel.parquet"
)
RMS_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets/full_retail.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
PLOT_OUTPUT_DIR = "../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. LAZY PANEL LOADING & FILTERING
# ==============================================================================
print("Loading panel data with Polars...")

raw_panel = (
    pl.scan_parquet(HMS_PATH)
    .with_columns(pl.all().name.to_lowercase())
    .with_columns(
        [
            pl.col("product_module_code_hms").cast(pl.Utf8).str.strip_chars(),
            pl.col("size1_amount_hms").cast(pl.Float64, strict=False),
            pl.col("size1_unit_hms").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("household_size").cast(pl.Int64, strict=False),
            pl.col("quantity").cast(pl.Int64, strict=False),
            pl.col("deal_flag_uc").cast(pl.Int64, strict=False),
            pl.col("household_code").cast(pl.Int64, strict=False),
            pl.col("store_code_uc").cast(pl.Int64, strict=False),
            pl.col("upc").cast(pl.Int64, strict=False),
        ]
    )
)

# Active households filter (>2 trips)
active_hhs = (
    raw_panel.group_by("household_code")
    .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
    .filter(pl.col("total_trips") > 2)
    .select("household_code")
)

# Apply panel filters natively
lazy_panel = (
    raw_panel.join(active_hhs, on="household_code", how="inner")
    .filter(pl.col("household_size") == 1)
    .filter(pl.col("size1_unit_hms") == "OZ")
    .filter(pl.col("size1_amount_hms").is_between(5000, 8001))
)

# ==============================================================================
# 2. NATIVE POLARS CLEANING & DATE ALIGNMENT
# ==============================================================================
# Native Date parsing & Saturday Week-End calculation
# 1. Parse date to Date type first
# 1. Parse date to Date type first
lazy_panel = lazy_panel.with_columns(
    [
        pl.col("purchase_date")
        .cast(pl.Utf8)
        .str.replace_all("-", "")
        .str.to_date("%Y%m%d", strict=False)
        .alias("parsed_date"),
        pl.col("household_income").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("male_head_age").cast(pl.Float64, strict=False).replace(0, None),
        pl.col("female_head_age").cast(pl.Float64, strict=False).replace(0, None),
    ]
)

# 2. Format the calculated days into a duration string for offset_by()
lazy_panel = lazy_panel.with_columns(
    [
        pl.col("parsed_date")
        .dt.offset_by(pl.format("{}d", (6 - pl.col("parsed_date").dt.weekday()) % 7))
        .cast(pl.Datetime("ms"))
        .alias("week_end"),
        pl.coalesce(["male_head_age", "female_head_age"]).alias("head_age"),
    ]
)

# Flavor & Purchase Indicators
lazy_panel = lazy_panel.with_columns(
    [
        pl.col("flavor").cast(pl.Utf8).fill_null("").alias("flavor_str"),
        pl.col("flavor_cd").cast(pl.Int64, strict=False).fill_null(0),
    ]
).with_columns(
    [
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
    ]
)

# ==============================================================================
# 3. SCAN RETAIL & MERGE (PURE LAZY STREAM)
# ==============================================================================
print("Scanning retail parquet and aligning key types...")

raw_retail = (
    pl.scan_parquet(RMS_PATH)
    .with_columns(pl.all().name.to_lowercase())
    .with_columns(
        [
            pl.col("week_end")
            .str.to_datetime("%Y-%m-%d", strict=False)
            .dt.cast_time_unit("ms"),
            pl.col("store_code_uc").cast(pl.Int64, strict=False),
            pl.col("upc").cast(pl.Int64, strict=False),
        ]
    )
    .filter(pl.col("week_end").is_not_null())
    .unique(subset=["week_end", "store_code_uc", "upc"])
)

# Join LazyFrames directly
master_df = lazy_panel.join(
    raw_retail, on=["week_end", "store_code_uc", "upc"], how="left"
)

# ==============================================================================
# 4. STREAM DIRECTLY TO PARQUET
# ==============================================================================
print(f"Streaming final joined data directly to {OUT_PATH}...")

# Because both sides are LazyFrames with identical primitive schema types,
# Polars can now stream this to disk without in-memory conversion spikes.
master_df.sink_parquet(OUT_PATH, engine="streaming")

print("Done! Streaming write completed successfully.")
