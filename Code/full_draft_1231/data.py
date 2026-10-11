"""
Data Processing & Summary Statistics Pipeline
- Full sample vs. Estimation sample summary statistics
- Figure 1: Distribution of Household Flavor Variety (HHI)
- Figure 2: Flavor Basket Composition by Purchase Volume Quintiles
- Figure 3: Time Series of Weekly Flavor Purchase Shares
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import seaborn as sns

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
# 1. DATA LOADING & POLARS FILTER PUSHDOWN (FULL SAMPLE)
# ==============================================================================
console.print("[bold green]Loading full panel with Polars...[/bold green]")

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
            pl.col("serving_per_container_cd").cast(pl.Int64, strict=False),
        ]
    )
)

# Active households with >2 trips overall
active_hhs = (
    raw_panel.group_by("household_code")
    .agg(pl.col("trip_code_uc").n_unique().alias("total_trips"))
    .filter(pl.col("total_trips") > 2)
    .select("household_code")
)

agent_panel = (
    raw_panel.join(active_hhs, on="household_code", how="inner").collect().to_pandas()
)

# Track total store trips across ALL grocery shopping per household BEFORE yogurt filtering
total_hh_trips = agent_panel.groupby("household_code")["trip_code_uc"].nunique()

# Parse dates & clean demographics
agent_panel["purchase_date"] = pd.to_datetime(
    agent_panel["purchase_date"].astype(str).str.replace("-", "", regex=False),
    format="%Y%m%d",
    errors="coerce",
)
agent_panel["week_end"] = agent_panel["purchase_date"] + pd.offsets.Week(weekday=5, n=0)

for col in [
    "quantity",
    "household_income",
    "deal_flag_uc",
    "male_head_age",
    "female_head_age",
    "price",
]:
    if col in agent_panel.columns:
        agent_panel[col] = pd.to_numeric(agent_panel[col], errors="coerce").fillna(0)

agent_panel["male_head_age"] = agent_panel["male_head_age"].replace(0, np.nan)
agent_panel["female_head_age"] = agent_panel["female_head_age"].replace(0, np.nan)
agent_panel["head_age"] = agent_panel["male_head_age"].fillna(
    agent_panel["female_head_age"]
)

# Yogurt Purchase Dummy
agent_panel["yogurt_purchase"] = (
    agent_panel["product_module_code_hms"].isin(["3603", "3612"])
    & (agent_panel["quantity"] > 0)
).astype(int)

# ==============================================================================
# 2. FILTER ESTIMATION SAMPLE (EXACT EST_STEP2 MATCHING)
# ==============================================================================
import statsmodels.formula.api as smf

# 1. Base Filters
est_master = agent_panel[
    (agent_panel["household_size"] == 1)
    & (agent_panel["serving_per_container_cd"].isin([67181961, 65622705]))
    & (agent_panel["product_module_code_hms"].isin(["3603", "3612"]))
].copy()

# 2. Price & Instrument Residual Filter (matches est_step2.py)
est_master = est_master[(est_master["price"] > 0.1) | (est_master["price"].isna())]

# Build Market Price IV
market_price = (
    est_master.groupby(["upc", "week_end", "market_name"])["price"].mean().reset_index()
)
totals = (
    market_price.groupby(["upc", "week_end"])["price"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(columns={"sum": "price_sum_all", "count": "n_markets_all"})
)
market_price = market_price.merge(totals, on=["upc", "week_end"], how="left")
market_price["price_iv"] = (market_price["price_sum_all"] - market_price["price"]) / (
    market_price["n_markets_all"] - 1
)
market_price["price_iv"] = market_price["price_iv"].replace([np.inf, -np.inf], np.nan)

est_master = est_master.merge(
    market_price[["upc", "week_end", "market_name", "price_iv"]],
    on=["upc", "week_end", "market_name"],
    how="left",
)

# Run OLS to isolate valid rows (dropping rows with missing brand/IV)
iv_res = smf.ols(
    "price ~ price_iv + brand_cd + C(week_end)", data=est_master, missing="drop"
).fit()

# Keep only the valid OLS rows
est_sample = est_master.loc[iv_res.model.data.row_labels].copy()

# Flavor Encoding (0: Other, 1: Berry, 2: Plain)
est_sample["flavor_str"] = est_sample["flavor"].fillna("").astype(str)
est_sample["flavor_cd"] = pd.to_numeric(
    est_sample["flavor_cd"], errors="coerce"
).fillna(0)

est_sample["flavor"] = np.select(
    [
        est_sample["flavor_str"].str.contains("berry", case=False, na=False),
        est_sample["flavor_cd"].isin([67676592, 66987057]),
    ],
    [1, 2],
    default=0,
)

# ==============================================================================
# 3. WEEKLY SHARES COMPUTATION
# ==============================================================================
weekly_flavor_qty = (
    est_sample.groupby(["household_code", "week_end", "flavor"])["quantity"]
    .sum()
    .reset_index()
)

weekly_total_qty = (
    weekly_flavor_qty.groupby(["household_code", "week_end"])["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"quantity": "total_weekly_qty"})
)

weekly_shares = weekly_flavor_qty.merge(
    weekly_total_qty, on=["household_code", "week_end"], how="left"
)
weekly_shares["share"] = weekly_shares["quantity"] / weekly_shares["total_weekly_qty"]

shares_wide = (
    weekly_shares.pivot_table(
        index=["household_code", "week_end"],
        columns="flavor",
        values="share",
        fill_value=0.0,
    )
    .reset_index()
    .rename(columns={0: "share_other", 1: "share_berry", 2: "share_plain"})
)

# ==============================================================================
# 4. FIGURE 1: HOUSEHOLD FLAVOR VARIETY (HHI DISTRIBUTION)
# ==============================================================================
console.print(
    "\n[bold green]Generating Figure 1: Household HHI Distribution...[/bold green]"
)

hh_flavor_qty = (
    est_sample.groupby(["household_code", "flavor"])["quantity"].sum().reset_index()
)
hh_total_qty = (
    hh_flavor_qty.groupby("household_code")["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"quantity": "hh_total_quantity"})
)
hh_shares = hh_flavor_qty.merge(hh_total_qty, on="household_code")
hh_shares["s_j"] = hh_shares["quantity"] / hh_shares["hh_total_quantity"]

# Calculate Herfindahl-Hirschman Index (HHI = sum(s_j^2)) per household
hh_hhi = (
    hh_shares.groupby("household_code")["s_j"]
    .apply(lambda x: np.sum(x**2))
    .reset_index()
    .rename(columns={"s_j": "hhi"})
)

plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")

sns.histplot(
    hh_hhi["hhi"],
    bins=25,
    kde=True,
    color="#2b5c8f",
    edgecolor="white",
    alpha=0.7,
)

plt.axvline(
    hh_hhi["hhi"].mean(),
    color="#d73027",
    linestyle="--",
    linewidth=2,
    label=f"Mean HHI ({hh_hhi['hhi'].mean():.2f})",
)

plt.title(
    "Distribution of Household Flavor Concentration (HHI)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel(
    "Herfindahl-Hirschman Index (1.0 = Single Flavor Loyalist)",
    fontsize=11,
    fontweight="bold",
)
plt.ylabel("Number of Households", fontsize=11, fontweight="bold")
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()

fig1_path = os.path.join(PLOT_OUTPUT_DIR, "fig1_flavor_hhi_distribution.pdf")
plt.savefig(fig1_path, format="pdf", bbox_inches="tight")
plt.close()

console.print(f"[bold green]Figure 1 saved to: {fig1_path}[/bold green]")

# ==============================================================================
# 5. FIGURE 2: BASKET COMPOSITION BY PURCHASE VOLUME QUINTILES
# ==============================================================================
console.print(
    "\n[bold green]Generating Figure 2: Basket Composition by Volume Quintiles...[/bold green]"
)

hh_total_qty["volume_quintile"] = pd.qcut(
    hh_total_qty["hh_total_quantity"],
    q=5,
    labels=["Q1 (Light)", "Q2", "Q3", "Q4", "Q5 (Heavy)"],
)

quintile_shares = hh_shares.merge(
    hh_total_qty[["household_code", "volume_quintile"]], on="household_code"
)

quintile_flavor_summary = (
    quintile_shares.groupby(["volume_quintile", "flavor"], observed=True)["quantity"]
    .sum()
    .reset_index()
)
quintile_totals = (
    quintile_flavor_summary.groupby("volume_quintile", observed=True)["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"quantity": "q_total"})
)

q_df = quintile_flavor_summary.merge(quintile_totals, on="volume_quintile")
q_df["basket_share"] = q_df["quantity"] / q_df["q_total"]

q_pivot = q_df.pivot_table(
    index="volume_quintile", columns="flavor", values="basket_share", fill_value=0.0
).rename(columns={0: "Other Share", 1: "Berry Share", 2: "Plain Share"})

fig, ax = plt.subplots(figsize=(9, 6))

colors = ["#7fbf7b", "#d73027", "#4575b4"]
q_pivot[["Berry Share", "Plain Share", "Other Share"]].plot(
    kind="bar", stacked=True, color=colors, ax=ax, width=0.6, edgecolor="white"
)

plt.title(
    "Flavor Basket Share Across Household Purchase Volume Quintiles",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Household Purchase Volume Quintile", fontsize=11, fontweight="bold")
plt.ylabel("Proportion of Total Yogurt Basket", fontsize=11, fontweight="bold")
plt.xticks(rotation=0)
plt.ylim(0, 1.0)
plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
plt.tight_layout()

fig2_path = os.path.join(PLOT_OUTPUT_DIR, "fig2_basket_share_by_volume.pdf")
plt.savefig(fig2_path, format="pdf", bbox_inches="tight")
plt.close()

console.print(f"[bold green]Figure 2 saved to: {fig2_path}[/bold green]")

# ==============================================================================
# 6. FIGURE 3: TIME SERIES OF FLAVOR CONSUMPTION SHARES
# ==============================================================================
console.print(
    "\n[bold green]Generating Figure 3: Time Series of Shares...[/bold green]"
)

ts_shares = (
    shares_wide.groupby("week_end")[["share_berry", "share_plain", "share_other"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

color_dict = {
    "share_berry": "#d73027",
    "share_plain": "#4575b4",
    "share_other": "#7fbf7b",
}
label_dict = {
    "share_berry": "Berry Flavored Share",
    "share_plain": "Plain Flavored Share",
    "share_other": "Other Flavored Share",
}

for col in ["share_berry", "share_plain", "share_other"]:
    plt.plot(
        ts_shares["week_end"],
        ts_shares[col],
        linewidth=2,
        label=label_dict[col],
        color=color_dict[col],
    )

plt.title("Average Flavor Shares Over Time", fontsize=13, fontweight="bold")
plt.xlabel("Week", fontsize=11, fontweight="bold")
plt.ylabel("Mean Flavor Share of Weekly Basket", fontsize=11, fontweight="bold")
plt.ylim(0, 1.0)
plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
plt.tight_layout()

fig3_path = os.path.join(PLOT_OUTPUT_DIR, "fig3_flavor_share_timeseries.pdf")
plt.savefig(fig3_path, format="pdf", bbox_inches="tight")
plt.close()

console.print(f"[bold green]Figure 3 saved to: {fig3_path}[/bold green]")

# ==============================================================================
# 7. SUMMARY STATISTICS COMPARISON TABLE
# ==============================================================================
console.print(
    "\n[bold yellow]=== SAMPLE COMPARISON SUMMARY STATISTICS ===[/bold yellow]"
)

table = Table(
    title="Summary Statistics: Full Sample vs. Estimation Sample",
    show_header=True,
)
table.add_column("Statistic / Metric", style="cyan")
table.add_column("Full Panel Sample", justify="right")
table.add_column("Estimation Sample", justify="right")

hh_full = agent_panel["household_code"].nunique()
hh_est = est_sample["household_code"].nunique()

trips_full = total_hh_trips.mean()
trips_est = total_hh_trips.loc[
    total_hh_trips.index.isin(est_sample["household_code"])
].mean()
yogurt_trips_est = est_sample.groupby("household_code")["trip_code_uc"].nunique().mean()
choice_weeks_est = est_sample.groupby("household_code")["week_end"].nunique().mean()

inc_full = agent_panel["household_income"].mean()
inc_est = est_sample["household_income"].mean()

age_full = agent_panel["head_age"].mean()
age_est = est_sample["head_age"].mean()

deal_full = (
    agent_panel.groupby(["household_code", "trip_code_uc"])["deal_flag_uc"].max().mean()
    * 100
)
deal_est = (
    est_sample.groupby(["household_code", "trip_code_uc"])["deal_flag_uc"].max().mean()
    * 100
)

units_est = est_sample.groupby(["household_code", "week_end"])["quantity"].sum().mean()

table.add_row("Unique Households", f"{hh_full:,}", f"{hh_est:,}")
table.add_row("Mean Total Store Trips / HH", f"{trips_full:.2f}", f"{trips_est:.2f}")
table.add_row("Mean Yogurt Purchase Trips / HH", "N/A", f"{yogurt_trips_est:.2f}")
table.add_row("Mean Observed Choice Weeks / HH", "N/A", f"{choice_weeks_est:.2f}")
table.add_row("Mean Household Income", f"${inc_full:,.2f}", f"${inc_est:,.2f}")
table.add_row("Mean Head Age", f"{age_full:.1f} yrs", f"{age_est:.1f} yrs")
table.add_row("Deal / Coupon Purchase Rate", f"{deal_full:.2f}%", f"{deal_est:.2f}%")
table.add_row("Mean Weekly Yogurt Units (Purchasers)", "N/A", f"{units_est:.2f} units")

console.print(table)
