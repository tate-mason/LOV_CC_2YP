"""
Data Processing & Merge Pipeline
Pure Polars Native / SLURM Optimized

Main memory strategy:
    1. Keep all large datasets lazy.
    2. Project narrow columns for expensive intermediate calculations.
    3. Filter RMS by year BEFORE deduplicating.
    4. Join and immediately stream each year to disk.
    5. Never collect the full HMS/RMS merge into memory.
"""

import os

# ==============================================================================
# 0. SLURM / POLARS CONFIGURATION
# ==============================================================================

# Respect the CPU allocation given by SLURM.
#
# IMPORTANT:
# This must happen BEFORE importing Polars.
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

if slurm_cpus is not None:
    os.environ["POLARS_MAX_THREADS"] = slurm_cpus
else:
    # Sensible fallback for running interactively / outside SLURM
    os.environ.setdefault("POLARS_MAX_THREADS", "8")


import polars as pl


# ==============================================================================
# 1. PATHS / SETTINGS
# ==============================================================================

HMS_PATH = (
    "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/full_panel.parquet"
)

RMS_PATH = (
    "/scratch/dtm63837/Kilts_Panel/"
    "nielsen_extracts/RMS/output_markets/full_retail.parquet"
)

OUTPUT_DIR = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts"

OUTPUT_GLOB = os.path.join(
    OUTPUT_DIR,
    "scanner_panel_*.parquet",
)

YEARS = [2022, 2023, 2024]


print("=" * 80)
print("POLARS / SLURM CONFIGURATION")
print("=" * 80)
print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
print(f"POLARS_MAX_THREADS: {os.environ.get('POLARS_MAX_THREADS')}")
print()


# ==============================================================================
# 2. LAZY PANEL LOADING
# ==============================================================================

print("=" * 80)
print("SCANNING HMS PANEL")
print("=" * 80)


raw_panel = (
    pl.scan_parquet(HMS_PATH)
    # Normalize column names
    .with_columns(pl.all().name.to_lowercase())
    # Normalize key variables / variables used downstream
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


# ==============================================================================
# 3. IDENTIFY ACTIVE HOUSEHOLDS
# ==============================================================================

print("Building active-household filter...")


# IMPORTANT:
#
# We only need household_code and trip_code_uc to determine whether a household
# has >2 unique trips.
#
# Explicitly narrowing the query here prevents the group-by from carrying
# irrelevant columns through this portion of the execution plan.
#
# The calculation is intentionally done BEFORE the other sample restrictions.
# This preserves the semantics of your original script:
#
#     "active" = >2 trips in the original HMS panel
#
# rather than:
#
#     "active" = >2 trips after applying the yogurt/sample filters.
#

active_hhs = (
    raw_panel.select(
        [
            "household_code",
            "trip_code_uc",
        ]
    )
    .group_by("household_code")
    .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
    .filter(pl.col("total_trips") > 2)
    .select("household_code")
)


# ==============================================================================
# 4. APPLY PANEL SAMPLE FILTERS
# ==============================================================================

print("Applying HMS sample restrictions...")


# A semi join is all we need here:
#
#     retain rows whose household_code occurs in active_hhs
#
# We do not need to physically attach anything from active_hhs to raw_panel.
# This is cleaner than an inner join for a pure membership restriction.

lazy_panel = (
    raw_panel.join(
        active_hhs,
        on="household_code",
        how="semi",
    )
    # Original sample restrictions
    .filter(pl.col("household_size") == 1)
    .filter(pl.col("size1_unit_hms") == "OZ")
    .filter(
        pl.col("size1_amount_hms").is_between(
            5000,
            8001,
        )
    )
)


# ==============================================================================
# 5. PANEL CLEANING / DATE ALIGNMENT
# ==============================================================================

print("Cleaning HMS variables and constructing week_end...")


# ------------------------------------------------------------------------------
# Parse purchase date
# ------------------------------------------------------------------------------

lazy_panel = lazy_panel.with_columns(
    [
        pl.col("purchase_date")
        .cast(pl.Utf8)
        .str.replace_all("-", "")
        .str.to_date(
            "%Y%m%d",
            strict=False,
        )
        .alias("parsed_date"),
        pl.col("household_income")
        .cast(
            pl.Float64,
            strict=False,
        )
        .fill_null(0.0),
        pl.col("male_head_age")
        .cast(
            pl.Float64,
            strict=False,
        )
        .replace(0, None),
        pl.col("female_head_age")
        .cast(
            pl.Float64,
            strict=False,
        )
        .replace(0, None),
    ]
)


# ------------------------------------------------------------------------------
# Align purchase date to Saturday week-end
# ------------------------------------------------------------------------------

lazy_panel = lazy_panel.with_columns(
    [
        pl.col("parsed_date")
        .dt.offset_by(
            pl.format(
                "{}d",
                (6 - pl.col("parsed_date").dt.weekday()) % 7,
            )
        )
        .cast(pl.Datetime("ms"))
        .alias("week_end"),
        pl.coalesce(
            [
                "male_head_age",
                "female_head_age",
            ]
        ).alias("head_age"),
    ]
)


# ==============================================================================
# 6. FLAVOR / PURCHASE VARIABLES
# ==============================================================================

print("Constructing flavor and yogurt purchase indicators...")


# First normalize the raw flavor variables.

lazy_panel = lazy_panel.with_columns(
    [
        pl.col("flavor").cast(pl.Utf8).fill_null("").alias("flavor_str"),
        pl.col("flavor_cd")
        .cast(
            pl.Int64,
            strict=False,
        )
        .fill_null(0),
    ]
)


# Then construct the variables used in the analysis.

lazy_panel = lazy_panel.with_columns(
    [
        # ----------------------------------------------------------------------
        # Flavor
        #
        # 0 = other
        # 1 = berry
        # 2 = flavor codes identified below
        # ----------------------------------------------------------------------
        pl.when(pl.col("flavor_str").str.contains("(?i)berry"))
        .then(1)
        .when(
            pl.col("flavor_cd").is_in(
                [
                    67676592,
                    66987057,
                ]
            )
        )
        .then(2)
        .otherwise(0)
        .alias("flavor"),
        # ----------------------------------------------------------------------
        # Yogurt purchase indicator
        # ----------------------------------------------------------------------
        pl.when(
            pl.col("product_module_code_hms").is_in(
                [
                    "3603",
                    "3612",
                ]
            )
            & (pl.col("quantity") > 0)
        )
        .then(1)
        .otherwise(0)
        .alias("yogurt_purchase"),
    ]
)


# ==============================================================================
# 7. LAZY RMS SCAN
# ==============================================================================

print("=" * 80)
print("SCANNING RMS")
print("=" * 80)


# IMPORTANT:
#
# DO NOT call .unique() here.
#
# The previous version effectively did:
#
#     entire RMS
#        -> parse
#        -> unique entire RMS
#        -> select year
#
# For a huge scanner dataset, that requires Polars to build deduplication state
# for the entire RMS extract.
#
# Instead we keep this as an un-deduplicated lazy scan and perform:
#
#     entire RMS
#        -> select year
#        -> unique that year
#        -> join that year
#        -> sink
#
# This substantially reduces the peak state needed by unique() and the join.

raw_retail = (
    pl.scan_parquet(RMS_PATH)
    # Normalize names
    .with_columns(pl.all().name.to_lowercase())
    # Normalize join keys
    .with_columns(
        [
            pl.col("week_end")
            .str.to_datetime(
                "%Y-%m-%d",
                strict=False,
            )
            .dt.cast_time_unit("ms"),
            pl.col("store_code_uc").cast(
                pl.Int64,
                strict=False,
            ),
            pl.col("upc").cast(
                pl.Int64,
                strict=False,
            ),
        ]
    )
    .filter(pl.col("week_end").is_not_null())
)


# ==============================================================================
# 8. YEAR-BY-YEAR RMS / HMS MERGE
# ==============================================================================

print()
print("=" * 80)
print("BEGINNING YEARLY MERGE")
print("=" * 80)


JOIN_KEYS = [
    "week_end",
    "store_code_uc",
    "upc",
]


for yr in sorted(YEARS):
    print()
    print("-" * 80)
    print(f"YEAR: {yr}")
    print("-" * 80)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"scanner_panel_{yr}.parquet",
    )

    # ==========================================================================
    # 8A. FILTER HMS TO YEAR
    # ==========================================================================

    print(f"[{yr}] Building HMS lazy query...")

    panel_sub = lazy_panel.filter(pl.col("week_end").dt.year() == yr)

    # ==========================================================================
    # 8B. FILTER RMS TO YEAR *BEFORE* DEDUPLICATION
    # ==========================================================================

    print(f"[{yr}] Building RMS lazy query...")

    retail_sub = (
        raw_retail
        # --------------------------------------------------------------
        # CRITICAL MEMORY OPTIMIZATION:
        #
        # Cut RMS down to this year before unique().
        # --------------------------------------------------------------
        .filter(pl.col("week_end").dt.year() == yr)
        # --------------------------------------------------------------
        # Guarantee one RMS observation per join key.
        #
        # This is deliberately AFTER the year restriction.
        # --------------------------------------------------------------
        .unique(
            subset=JOIN_KEYS,
        )
    )

    # ==========================================================================
    # OPTIONAL FURTHER OPTIMIZATION
    # ==========================================================================
    #
    # If you know exactly which RMS columns you eventually need, add a .select()
    # BEFORE .unique().
    #
    # For example:
    #
    # retail_sub = (
    #     raw_retail
    #     .filter(
    #         pl.col("week_end").dt.year() == yr
    #     )
    #     .select(
    #         [
    #             "week_end",
    #             "store_code_uc",
    #             "upc",
    #             "price",
    #             "feature",
    #             "display",
    #         ]
    #     )
    #     .unique(
    #         subset=JOIN_KEYS,
    #     )
    # )
    #
    # This can save a VERY large amount of memory if full_retail.parquet is wide.
    #
    # For now I leave every RMS variable intact because I do not want to silently
    # change the contents of your merged dataset.

    # ==========================================================================
    # 8C. JOIN
    # ==========================================================================

    print(f"[{yr}] Constructing HMS x RMS join...")

    master_sub = panel_sub.join(
        retail_sub,
        on=JOIN_KEYS,
        how="left",
    )

    # ==========================================================================
    # 8D. IMMEDIATELY STREAM RESULT TO DISK
    # ==========================================================================

    print(f"[{yr}] Streaming merged data to:")
    print(f"       {output_path}")

    master_sub.sink_parquet(
        output_path,
        engine="streaming",
    )

    print(f"[{yr}] Complete.")


# ==============================================================================
# 9. COMBINED LAZY DATASET
# ==============================================================================

print()
print("=" * 80)
print("BUILDING COMBINED LAZY SCAN")
print("=" * 80)


# This does NOT load the yearly files into RAM.
#
# Polars treats the glob as one logical lazy dataset. Any downstream filters,
# selects, group-bys, etc. can continue to take advantage of predicate and
# projection pushdown.

combined_lazy = pl.scan_parquet(OUTPUT_GLOB)


print(f"Combined scan: {OUTPUT_GLOB}")


# ==============================================================================
# 10. DOWNSTREAM ANALYSIS
# ==============================================================================
#
# Add the rest of your analysis below this point.
#
# Examples:
#
#
# summary = (
#     combined_lazy
#     .group_by("year")
#     .agg(
#         [
#             pl.len().alias("n"),
#         ]
#     )
#     .collect(engine="streaming")
# )
#
#
# or:
#
#
# analysis_sample = (
#     combined_lazy
#     .filter(pl.col("yogurt_purchase") == 1)
# )
#
#
# Keep analysis lazy for as long as possible. In particular, avoid:
#
#     combined_lazy.collect()
#
# on the full dataset unless you actually need the entire merged dataset
# materialized in memory.


print()
print("=" * 80)
print("MERGE COMPLETE")
print("=" * 80)
