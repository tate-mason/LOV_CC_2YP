"""
Data Processing & Summary Statistics Pipeline
- Full sample summary statistics (Everyone)
- Strict estimation sample summary statistics (Single-serve yogurt purchasers)
- Flavor switching analysis & heatmaps
- Native price enforcement (No price imputation)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.traceback import install

install()
console = Console()

pd.set_option("display.max_rows", None, "display.max_columns", None)

HMS_PATH = (
    "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/full_panel.parquet"
)
MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
PLOT_OUTPUT_DIR = "../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA LOADING & POLARS LAZY PIPELINE (NATIVE PRICES ONLY)
# ==============================================================================
console.print("[bold green]Loading raw panel with Polars...[/bold green]")

raw_panel = (
    pl.scan_parquet(HMS_PATH)
    .with_columns(pl.all().name.to_lowercase())
    .with_columns(
        [
            pl.col("product_module_code_hms").cast(pl.Int64, strict=False),
            pl.col("serving_per_container_cd").cast(pl.Int64, strict=False),
            pl.col("size1_amount_hms").cast(pl.Float64, strict=False),
            pl.col("size1_unit_hms").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("household_size").cast(pl.Int64, strict=False),
            pl.col("quantity").cast(pl.Int64, strict=False),
            pl.col("deal_flag_uc").cast(pl.Int64, strict=False),
            pl.col("household_income").cast(pl.Int64, strict=False),
        ]
    )
)

# Active households across overall panel
active_hhs = (
    raw_panel.group_by("household_code")
    .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
    .filter(pl.col("total_trips") > 2)
    .select("household_code")
)

# Base Sample: All active 1-person households
base_lazy = raw_panel.join(active_hhs, on="household_code", how="inner").filter(
    pl.col("household_size") == 1
)

# Full Panel Collect
df_full_panel = base_lazy.collect().to_pandas()

schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

exprs = [
    pl.col("quantity").cast(pl.Int64),
    pl.col("head_age").cast(pl.Int64),
    pl.col("household_income").cast(pl.Int64),
    pl.col("household_size").cast(pl.Int64),
    pl.col("serving_per_container_cd").cast(pl.Int64),
]

if "product_module_code_hms" in schema_names:
    exprs.append(pl.col("product_module_code_hms").cast(pl.Int64))

if "price" in schema_names:
    exprs.append(pl.col("price").cast(pl.Float64))
elif "total_price_paid" in schema_names:
    exprs.append(
        (
            pl.col("total_price_paid").cast(pl.Float64)
            / pl.col("quantity").cast(pl.Float64)
        ).alias("price")
    )

scan_q = pl.scan_parquet(MERGED_PATH).with_columns(exprs)

if "product_module_code_hms" in schema_names:
    scan_q = scan_q.filter(pl.col("product_module_code_hms").is_in([3612, 3603]))

df_estimation = (
    scan_q.filter(pl.col("household_size") == 1)
    .filter(pl.col("serving_per_container_cd").is_in([67181961, 65622705]))
    .collect()
    .to_pandas()
)

console.print(
    f"[bold cyan]Full Single-Person Household Sample Loaded:[/bold cyan] {len(df_full_panel):,} rows | "
    f"{df_full_panel['household_code'].nunique():,} unique HHs"
)
console.print(
    f"[bold cyan]Strict Estimation Sample Loaded:[/bold cyan]          {len(df_estimation):,} rows | "
    f"{df_estimation['household_code'].nunique():,} unique HHs"
)

# Save processed parquet file for estimation scripts retaining native price
# pl.from_pandas(df_estimation).write_parquet(OUT_PATH)
# console.print(
#    f"[bold green]Saved estimation parquet directly to: {OUT_PATH}[/bold green]"
# )

# ==============================================================================
# 2. FLAVOR & SWITCHING ENCODING (ESTIMATION SAMPLE)
# ==============================================================================
# 0 = Plain, 1 = Other, 2 = Berry
descr_cols = [
    c
    for c in df_estimation.columns
    if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"])
]

df_estimation["full_text"] = ""
for c in descr_cols:
    ["full_text"] += " " + df_estimation[c].fillna("").astype(str)
df_estimation["full_text"] = df_estimation["full_text"].str.lower()

berry_regex = r"berry|straw|blue|rasp|black|cran|cherry|wildberry"
plain_regex = r"plain|unflavored"

is_plain = df_estimation["full_text"].str.contains(plain_regex, na=False)
is_berry = df_estimation["full_text"].str.contains(berry_regex, na=False) & (~is_plain)
df_estimation["flavor_cat"] = np.select([is_plain, is_berry], [0, 2], default=1)

df_estimation = df_estimation.sort_values(
    ["household_code", "purchase_date", "trip_code_uc"]
)
df_estimation["trip_seq"] = df_estimation.groupby("household_code").cumcount() + 1
df_estimation["prev_flavor"] = df_estimation.groupby("household_code")[
    "flavor_cat"
].shift(1)

df_estimation["switched"] = np.where(
    df_estimation["trip_seq"] > 1,
    (df_estimation["flavor_cat"] != df_estimation["prev_flavor"]).astype(int),
    0,
)

# ==============================================================================
# 3. SUMMARY STATISTICS COMPARISON TABLE
# ==============================================================================
console.print(
    "\n[bold yellow]=====================================================[/bold yellow]"
)
console.print(
    "[bold yellow]          SUMMARY STATISTICS COMPARISON              [/bold yellow]"
)
console.print(
    "[bold yellow]=====================================================[/bold yellow]"
)

stats_table = Table(show_header=True, header_style="bold magenta")
stats_table.add_column("Metric / Characteristic", style="cyan", justify="left")
stats_table.add_column("Full Panel Sample (Everyone)", justify="right")
stats_table.add_column("Strict Estimation Sample", justify="right")

full_hhs = df_full_panel["household_code"].nunique()
est_hhs = df_estimation["household_code"].nunique()

full_trips = df_full_panel.groupby("household_code")["trip_code_uc"].nunique().mean()
est_trips = df_estimation.groupby("household_code")["trip_code_uc"].nunique().mean()

full_inc_mean = df_full_panel["household_income"].mean()
est_inc_mean = df_estimation["household_income"].mean()

full_inc_med = df_full_panel["household_income"].median()
est_inc_med = df_estimation["household_income"].median()

full_age = df_full_panel["head_age"].mean()
est_age = df_estimation["head_age"].mean()

stats_table.add_row("Unique Households", f"{full_hhs:,}", f"{est_hhs:,}")
stats_table.add_row(
    "Total Purchases / Rows", f"{len(df_full_panel):,}", f"{len(df_estimation):,}"
)
stats_table.add_row("Mean Shopping Trips / HH", f"{full_trips:.2f}", f"{est_trips:.2f}")
stats_table.add_row(
    "Mean Household Income ($)",
    f"${full_inc_mean:,.2f}",
    f"${est_inc_mean:,.2f}",
)
stats_table.add_row(
    "Median Household Income ($)",
    f"${full_inc_med:,.2f}",
    f"${est_inc_med:,.2f}",
)
stats_table.add_row("Mean Head Age", f"{full_age:.1f}", f"{est_age:.1f}")

est_price = df_estimation["price"].mean()
stats_table.add_row("Mean Native Unit Price ($)", "N/A", f"${est_price:.2f}")

console.print(stats_table)

# ==============================================================================
# 4. FLAVOR SWITCHING METRICS (ESTIMATION SAMPLE)
# ==============================================================================
df_estimation["flavor_spell_id"] = df_estimation.groupby("household_code")[
    "switched"
].cumsum()
df_estimation["flavor_spell_buys"] = (
    df_estimation.groupby(["household_code", "flavor_spell_id"]).cumcount() + 1
)
df_estimation["spell_length"] = df_estimation.groupby(
    ["household_code", "flavor_spell_id"]
)["flavor_spell_buys"].transform("max")

switching_sample = df_estimation[df_estimation["switched"] == 1]

console.print("\n[bold yellow]=== FLAVOR SWITCHING METRICS ===[/bold yellow]")
console.print(
    f"Mean Consecutive Buys per Flavor Spell:               {df_estimation['spell_length'].mean():.2f} trips\n"
    f"Mean Flavor Switches per HH:                         {df_estimation.groupby('household_code')['switched'].sum().mean():.2f}\n"
    f"Percent of HHs who Ever Switch Flavors:              {(df_estimation.groupby('household_code')['flavor_cat'].nunique() > 1).mean() * 100:.2f}%"
)

# ==============================================================================
# 5. HEATMAP VISUALIZATION
# ==============================================================================
if not switching_sample.empty:
    console.print("\n[bold green]Generating flavor switching heatmap...[/bold green]")

    heat_flav = (
        switching_sample.groupby(["prev_flavor", "flavor_cat"])["spell_length"]
        .mean()
        .unstack()
        .rename(
            columns={0: "Plain", 1: "Other", 2: "Berry"},
            index={0: "Plain", 1: "Other", 2: "Berry"},
        )
    )

    cell_labs = np.array(
        [
            [f"{val:.1f} trips" if not np.isnan(val) else "" for val in row]
            for row in heat_flav.to_numpy()
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat_flav,
        annot=cell_labs,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Spell Length (Trips)"},
        ax=ax,
    )

    ax.set_xlabel("Flavor Switched To", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flavor Switched From", fontsize=11, fontweight="bold")
    ax.set_title(
        "Mean Spell Length Upon Switching Flavors",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = os.path.join(PLOT_OUTPUT_DIR, "3_flav_heatmap.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

    console.print(
        f"[bold green]Heatmap successfully saved to: {output_path}[/bold green]"
    )
