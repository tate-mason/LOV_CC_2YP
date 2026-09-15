"""
Data Processing & Summary Statistics Pipeline
- Full sample summary statistics
- Yogurt purchasers summary statistics
- Flavor switching analysis & heatmaps
"""


import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.traceback import install

install()
console = Console()

pd.set_option("display.max_rows", None, "display.max_columns", None)

HMS_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/full_panel.parquet"
PLOT_OUTPUT_DIR = "../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA LOADING & POLARS FILTER PUSHDOWN
# ==============================================================================
console.print("[bold green]Loading data with Polars...[/bold green]")

# Scan raw parquet with strict string formatting for module codes
raw_scan = (
    pl.scan_parquet(HMS_PATH)
    .with_columns(pl.all().name.to_lowercase())
    .with_columns([
        pl.col("product_module_code_hms").cast(pl.Utf8).str.strip_chars(),
        pl.col("size1_amount_hms").cast(pl.Float64, strict=False),
        pl.col("size1_unit_hms").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("household_size").cast(pl.Int64, strict=False),
        pl.col("quantity").cast(pl.Int64, strict=False),
        pl.col("deal_flag_uc").cast(pl.Int64, strict=False),
    ])
)

# 1. Household trip thresholds across the entire dataset
active_hhs = (
    raw_scan.group_by("household_code")
    .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
    .filter(pl.col("total_trips") > 2)
    .select("household_code")
)

# 2. Apply filters one at a time to see where rows disappear
lazy_panel = raw_scan.join(active_hhs, on="household_code", how="inner")
console.print("Rows after household join:", lazy_panel.select(pl.len()).collect().item())

lazy_panel = lazy_panel.filter(pl.col("household_size") == 1)
console.print("Rows after household size filter:", lazy_panel.select(pl.len()).collect().item())

lazy_panel = lazy_panel.filter(pl.col("product_module_code_hms").is_in(["3603", "3612"]))
console.print("Rows after module code filter:", lazy_panel.select(pl.len()).collect().item())

lazy_panel = lazy_panel.filter(pl.col("size1_unit_hms") == "OZ")
console.print("Rows after OZ filter:", lazy_panel.select(pl.len()).collect().item())

console.print("Most common size amounts before the 5–8 filter:")
console.print(
    lazy_panel.group_by("size1_amount_hms")
    .len()
    .sort("len", descending=True)
    .collect()
)

lazy_panel = lazy_panel.filter(pl.col("size1_amount_hms").is_between(5, 8))
console.print("Rows after size amount filter:", lazy_panel.select(pl.len()).collect().item())

agent_panel = lazy_panel.collect().to_pandas()

console.print(
    f"Filtered panel loaded: {len(agent_panel):,} rows | "
    f"{agent_panel['household_code'].nunique():,} unique single-person yogurt-purchasing HHs"
)

# ==============================================================================
# 2. DATA CLEANING & SAFE TYPE CASTING
# ==============================================================================

# Parse dates
agent_panel["purchase_date"] = pd.to_datetime(
    agent_panel["purchase_date"].astype(str).str.replace("-", "", regex=False),
    format="%Y%m%d",
    errors="coerce"
)
agent_panel["week_end"] = agent_panel["purchase_date"] + pd.offsets.Week(weekday=5, n=0)

# Explicit numeric casting for Pandas/PyArrow safety
numeric_cols = ["quantity", "household_income", "deal_flag_uc", "male_head_age", "female_head_age"]
for col in numeric_cols:
    if col in agent_panel.columns:
        agent_panel[col] = pd.to_numeric(agent_panel[col], errors="coerce").fillna(0)

# Re-evaluate age logic safely
agent_panel["male_head_age"] = agent_panel["male_head_age"].replace(0, np.nan)
agent_panel["female_head_age"] = agent_panel["female_head_age"].replace(0, np.nan)
agent_panel["head_age"] = agent_panel["male_head_age"].fillna(agent_panel["female_head_age"])

# Safe Flavor Encoding
agent_panel["flavor_str"] = agent_panel["flavor"].fillna("").astype(str)
agent_panel["flavor_cd"] = pd.to_numeric(agent_panel["flavor_cd"], errors="coerce").fillna(0)

agent_master = agent_panel.copy()
agent_master["flavor"] = np.select(
    [
        agent_master["flavor_str"].str.contains("berry", case=False, na=False),
        agent_master["flavor_cd"].isin([67676592, 66987057]),
    ],
    [1, 2],
    default=0
)

# Yogurt Purchase Dummy
agent_master["yogurt_purchase"] = (
    agent_master["product_module_code_hms"].isin(["3603", "3612"]) &
    (agent_master["quantity"] > 0)
).astype(int)

# Outside Option Analysis
trip_yogurt = agent_master.groupby(["household_code", "trip_code_uc"])["yogurt_purchase"].max().reset_index()
trip_yogurt["chose_outside_option"] = (trip_yogurt["yogurt_purchase"] == 0).astype(int)
outside_option_rate = trip_yogurt["chose_outside_option"].mean()

# Filter for yogurt purchases safely
agent_yogurt = agent_master[agent_master["yogurt_purchase"] == 1].copy()

# Sort chronologically for switching metrics
agent_yogurt = agent_yogurt.sort_values(["household_code", "purchase_date", "trip_code_uc"])

# Trip sequence numbers per household
agent_yogurt["trip_seq"] = agent_yogurt.groupby("household_code").cumcount() + 1
agent_yogurt["prev_flavor"] = agent_yogurt.groupby("household_code")["flavor"].shift(1)

# Switching dummy (only valid from trip 2 onwards)
agent_yogurt["switched"] = np.where(
    agent_yogurt["trip_seq"] > 1,
    (agent_yogurt["flavor"] != agent_yogurt["prev_flavor"]).astype(int),
    0
)

# ==============================================================================
# 3. SUMMARY STATISTICS
# ==============================================================================
console.print("\n[bold yellow]=== FULL SAMPLE SUMMARY STATISTICS ===[/bold yellow]")

console.print(
    f"Number of households in full sample:                  {agent_master['household_code'].nunique():,}\n"
    f"Number of yogurt-purchasing households:               {agent_yogurt['household_code'].nunique():,}\n"
    f"Mean number of trips per HH:                          {agent_master.groupby('household_code')['trip_code_uc'].nunique().mean():.2f}\n"
    f"Number of yogurt purchases per trip (purchasers):     {agent_yogurt.groupby(['household_code', 'trip_code_uc'])['quantity'].sum().mean():.2f}\n"
    f"Mean household income:                                ${agent_master['household_income'].mean():,.2f}\n"
    f"Median household income:                              ${agent_master['household_income'].median():,.2f}\n"
    f"Percent taking outside option each trip:              {outside_option_rate * 100:.2f}%\n"
    f"Percent purchasing with coupon:                       {agent_yogurt.groupby(['household_code', 'trip_code_uc'])['deal_flag_uc'].max().mean() * 100:.2f}%\n"
    f"Average Age (Overall):                                {agent_master['head_age'].mean():.1f}\n"
    f"Average Age (Male):                                   {agent_master['male_head_age'].mean():.1f}\n"
    f"Average Age (Female):                                 {agent_master['female_head_age'].mean():.1f}"
)

# ==============================================================================
# 4. FLAVOR SWITCHING METRICS
# ==============================================================================
# Sequence indicators
agent_yogurt["next_flavor"] = agent_yogurt.groupby("household_code")["flavor"].shift(-1)

# Flavor spells
agent_yogurt["flavor_spell_id"] = agent_yogurt.groupby("household_code")["switched"].cumsum()
agent_yogurt["flavor_spell_buys"] = agent_yogurt.groupby(["household_code", "flavor_spell_id"]).cumcount() + 1
agent_yogurt["spell_length"] = agent_yogurt.groupby(["household_code", "flavor_spell_id"])["flavor_spell_buys"].transform("max")

# Filtered Switch Datasets
switching_sample = agent_yogurt[agent_yogurt["switched"] == 1]
switches_coupon = switching_sample[switching_sample["deal_flag_uc"] == 1]

# Guarded percent calculation to avoid ZeroDivisionError
coupon_switch_pct = (len(switches_coupon) / len(switching_sample) * 100) if len(switching_sample) > 0 else 0.0

console.print("\n[bold yellow]=== FLAVOR SWITCHING METRICS ===[/bold yellow]")
console.print(
    f"Mean consecutive buys by flavor x HH:                 {agent_yogurt['spell_length'].mean():.2f}\n"
    f"Mean times switching by flavor x HH:                  {agent_yogurt.groupby(['household_code', 'flavor'])['switched'].sum().mean():.2f}\n"
    f"Percent of HH who ever-switch flavors:                 {(agent_yogurt.groupby('household_code')['flavor'].nunique() > 1).mean() * 100:.2f}%\n"
    f"Percent switching due to coupon/deal:                 {coupon_switch_pct:.2f}%"
)

# ==============================================================================
# 5. HEATMAP VISUALIZATION
# ==============================================================================
if not switching_sample.empty:
    console.print("\n[bold green]Generating flavor switching heatmap...[/bold green]")
    
    heat_flav = (
        switching_sample.groupby(["prev_flavor", "flavor"])["spell_length"]
        .mean()
        .unstack()
        .rename(columns={0: "Other", 1: "Berry", 2: "Plain"}, index={0: "Other", 1: "Berry", 2: "Plain"})
    )

    cell_labs = np.array([[f"{val:.1f} trips" if not np.isnan(val) else "" for val in row] for row in heat_flav.to_numpy()])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat_flav,
        annot=cell_labs,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Spell Length (Trips)"},
        ax=ax
    )

    ax.set_xlabel("Flavor Switched To", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flavor Switched From", fontsize=11, fontweight="bold")
    ax.set_title("Mean Spell Length Upon Switching Flavors", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    output_path = os.path.join(PLOT_OUTPUT_DIR, "3_flav_heatmap.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    console.print(f"[bold green]Heatmap successfully saved to: {output_path}[/bold green]")
else:
    console.print("[bold red]No switching records found to generate heatmap.[/bold red]")
