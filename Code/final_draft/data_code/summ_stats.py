import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Directories
data_dir = "/scratch/dtm63837/Kilts_Panel/Data"
output_dir = os.path.join(data_dir, "summary_outputs")
os.makedirs(output_dir, exist_ok=True)

# File Paths
panel_path = os.path.join(data_dir, "merged_panel.parquet")
estimation_path = os.path.join(data_dir, "yogurt_estimation_sample.parquet")
flavor_table_path = os.path.join(
    data_dir,
    "../nielsen_extracts/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv",
)  # Adjust path if needed


# ==============================================================================
# SECTION 1: OVERALL PANEL SUMMARY STATISTICS
# ==============================================================================
print("\n[SECTION 1] Processing Overall Panel Demographics & Behavior...", flush=True)

# Lazy scan of overall panel
panel_lazy = pl.scan_parquet(panel_path)

# Demographics & Weekly Trips Calculation
# Grouping by Household to get demographic distributions and weekly behavior
hh_panel_stats = (
    panel_lazy.group_by("household_code")
    .agg(
        [
            pl.col("panel_year").min().alias("start_year"),
            pl.col("panel_year").max().alias("end_year"),
            # Categorical brackets: take mode/first non-null observation per household
            pl.col("age").drop_nulls().first().alias("age_bracket"),
            pl.col("race").drop_nulls().first().alias("race_bracket"),
            pl.col("income").drop_nulls().first().alias("income_bracket"),
            # Total distinct purchases/products
            pl.col("upc").n_unique().alias("distinct_products_bought"),
            pl.col("upc").count().alias("total_items_bought"),
            # Distinct trips per household using purchase_date
            pl.col("purchase_date").n_unique().alias("total_trips"),
        ]
    )
    .with_columns(
        [
            # Approximate weeks active to compute average weekly trips
            ((pl.col("end_year") - pl.col("start_year") + 1) * 52).alias(
                "active_weeks"
            ),
        ]
    )
    .with_columns(
        [(pl.col("total_trips") / pl.col("active_weeks")).alias("trips_per_week")]
    )
    .collect()
)

# Build Demographic Distributions (Shares by Bracket)
age_dist = (
    hh_panel_stats["age_bracket"].value_counts(normalize=True).sort("age_bracket")
)
race_dist = (
    hh_panel_stats["race_bracket"].value_counts(normalize=True).sort("race_bracket")
)
income_dist = (
    hh_panel_stats["income_bracket"].value_counts(normalize=True).sort("income_bracket")
)

# Summary Table for Overall Panel
panel_summary_dict = {
    "Metric": [
        "Total Households",
        "Trips per Week (Mean)",
        "Trips per Week (Median)",
        "Distinct Products Bought per HH (Mean)",
        "Total Items Bought per HH (Mean)",
    ],
    "Value": [
        f"{len(hh_panel_stats):,}",
        f"{hh_panel_stats['trips_per_week'].mean():.2f}",
        f"{hh_panel_stats['trips_per_week'].median():.2f}",
        f"{hh_panel_stats['distinct_products_bought'].mean():.2f}",
        f"{hh_panel_stats['total_items_bought'].mean():.2f}",
    ],
}

df_panel_summary = pd.DataFrame(panel_summary_dict)

# Export Section 1 LaTeX
latex_panel_summary = df_panel_summary.to_latex(
    index=False,
    caption="Overall Panel Demographics and Weekly Purchase Behavior",
    label="tab:overall_panel_summary",
    column_format="lc",
)

with open(os.path.join(output_dir, "table1_overall_panel.tex"), "w") as f:
    f.write(latex_panel_summary)

print("  --> Saved table1_overall_panel.tex", flush=True)


# ==============================================================================
# SECTION 2: YOGURT ESTIMATION SAMPLE & FLAVOR DYNAMICS
# ==============================================================================
print(
    "\n[SECTION 2] Processing Yogurt Estimation Sample & Flavor Dynamics...", flush=True
)

# Load Estimation Sample and Merge Flavor Data
est_lazy = pl.scan_parquet(estimation_path)

if os.path.exists(flavor_table_path):
    flavor_df = pl.scan_csv(flavor_table_path)
    est_lazy = est_lazy.join(flavor_df, on="upc", how="left")
else:
    # Placeholder if flavor table is merged downstream: uses mock/fallback column
    print(
        "  [NOTE] Flavor dictionary path not found; using existing 'flavor' column.",
        flush=True,
    )

# Ensure sorted order for sequence and spell calculations
est_df = (
    est_lazy.filter(pl.col("flavor").is_not_null())
    .sort(["household_code", "purchase_date"])
    .collect()
)

# ------------------------------------------------------------------------------
# 2A. Compute Household Flavor Switching & Spells
# ------------------------------------------------------------------------------
# Identify switches when consecutive purchases within a household differ in flavor
est_df = est_df.with_columns(
    [
        pl.col("flavor").shift(1).over("household_code").alias("prev_flavor"),
        pl.col("purchase_date").cast(pl.Date),
    ]
)

est_df = est_df.with_columns(
    [
        (
            (pl.col("flavor") != pl.col("prev_flavor"))
            & pl.col("prev_flavor").is_not_null()
        )
        .cast(pl.Int32)
        .alias("is_switch")
    ]
)

# Compute consecutive run/spell IDs for each household-flavor streak
est_df = est_df.with_columns(
    [pl.col("is_switch").cum_sum().over("household_code").alias("spell_id")]
)

# Aggregate spell lengths (number of consecutive purchases per flavor spell)
spell_lengths = est_df.group_by(["household_code", "spell_id"]).agg(
    [
        pl.col("flavor").first().alias("flavor"),
        pl.len().alias("spell_length_purchases"),
        (pl.col("purchase_date").max() - pl.col("purchase_date").min())
        .dt.total_days()
        .alias("spell_duration_days"),
    ]
)

# Household-level summary metrics
hh_est_stats = est_df.group_by("household_code").agg(
    [
        pl.col("flavor").n_unique().alias("n_flavors_bought"),
        pl.col("is_switch").sum().alias("total_flavor_switches"),
        pl.col("upc").count().alias("total_yogurt_purchases"),
        pl.col("purchase_date").n_unique().alias("yogurt_trips"),
    ]
)

# Merge back household spell averages
hh_avg_spells = spell_lengths.group_by("household_code").agg(
    [
        pl.col("spell_length_purchases").mean().alias("avg_spell_purchases"),
        pl.col("spell_duration_days").mean().alias("avg_spell_days"),
    ]
)

hh_est_stats = hh_est_stats.join(hh_avg_spells, on="household_code", how="left")

# Format Estimation Sample Summary Table
est_summary_dict = {
    "Metric": [
        "Estimation Sample Households",
        "Distinct Flavors Bought per HH (Mean)",
        "Distinct Flavors Bought per HH (Median)",
        "Total Flavor Switches per HH (Mean)",
        "Average Spell Length (Purchases)",
        "Average Spell Duration (Days)",
    ],
    "Value": [
        f"{len(hh_est_stats):,}",
        f"{hh_est_stats['n_flavors_bought'].mean():.2f}",
        f"{hh_est_stats['n_flavors_bought'].median():.2f}",
        f"{hh_est_stats['total_flavor_switches'].mean():.2f}",
        f"{hh_est_stats['avg_spell_purchases'].mean():.2f}",
        f"{hh_est_stats['avg_spell_days'].mean():.2f}",
    ],
}

df_est_summary = pd.DataFrame(est_summary_dict)

# Export Section 2 LaTeX
latex_est_summary = df_est_summary.to_latex(
    index=False,
    caption="Estimation Sample Flavor Switching and Purchase Spells",
    label="tab:estimation_sample_summary",
    column_format="lc",
)

with open(os.path.join(output_dir, "table2_estimation_sample.tex"), "w") as f:
    f.write(latex_est_summary)

print("  --> Saved table2_estimation_sample.tex", flush=True)


# ==============================================================================
# SECTION 3: VISUALIZATIONS & FLAVOR HHI / SWITCHING HEATMAP
# ==============================================================================
print("\n[SECTION 3] Generating Visualizations...", flush=True)

# ------------------------------------------------------------------------------
# 3A. Top 10 Flavor Shares & HHI Calculation
# ------------------------------------------------------------------------------
flavor_shares = (
    est_df["flavor"].value_counts(normalize=True).sort("count", descending=True)
)

# Compute HHI across all flavors: Sum of (Share * 100)^2
hhi_value = ((flavor_shares["count"] * 100) ** 2).sum()

top_10_flavors = flavor_shares.head(10)
top_10_list = top_10_flavors["flavor"].to_list()

# Bar Chart for Top 10 Flavors
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=top_10_flavors.to_pandas(),
    x="count",
    y="flavor",
    palette="Blues_r",
)
plt.title(
    f"Top 10 Flavor Market Shares (Overall Yogurt HHI: {hhi_value:.1f})",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Market Share")
plt.ylabel("Flavor")

# Format x-axis as percentage
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
hhi_fig_path = os.path.join(output_dir, "fig1_flavor_shares_hhi.png")
plt.savefig(hhi_fig_path, dpi=300)
plt.close()

print(f"  --> Saved fig1_flavor_shares_hhi.png (HHI = {hhi_value:.1f})", flush=True)

# ------------------------------------------------------------------------------
# 3B. Transition / Switching Heatmap for Top 10 Flavors
# ------------------------------------------------------------------------------
# Filter transitions where both previous and current flavor are in top 10
switching_df = (
    est_df.filter(
        pl.col("flavor").is_in(top_10_list)
        & pl.col("prev_flavor").is_in(top_10_list)
        & (pl.col("flavor") != pl.col("prev_flavor"))  # Only include actual switches
    )
    .group_by(["prev_flavor", "flavor"])
    .len()
    .to_pandas()
)

# Pivot into square matrix
transition_matrix = switching_df.pivot(
    index="prev_flavor", columns="flavor", values="len"
).fillna(0)

# Reindex matrix to match top 10 order
transition_matrix = transition_matrix.reindex(
    index=top_10_list, columns=top_10_list, fill_value=0
)

# Normalize by row (probability of switching to flavor Y given starting at flavor X)
row_sums = transition_matrix.sum(axis=1)
transition_prob = transition_matrix.div(row_sums, axis=0).fillna(0)

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    transition_prob,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar_kws={"label": "Transition Probability"},
    linewidths=0.5,
)
plt.title(
    "Flavor Switching Transition Matrix (Top 10 Flavors)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Switched To (Flavor t)")
plt.ylabel("Switched From (Flavor t-1)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
heatmap_fig_path = os.path.join(output_dir, "fig2_flavor_switching_heatmap.png")
plt.savefig(heatmap_fig_path, dpi=300)
plt.close()

print("  --> Saved fig2_flavor_switching_heatmap.png", flush=True)


# ==============================================================================
# SECTION 4: OTHER RECOMMENDED SUMMARY STATISTICS (COMMENTED OUT)
# ==============================================================================
"""
ADDITIONAL RECOMMENDED SUMMARY STATISTICS FOR STRUCTURAL/DEMAND MODELS:

1. PRICE ELASTICITY & PROMOTIONAL SENSITIVITY:
   - Average regular price vs. promotional price paid per ounce.
   - Percentage of yogurt purchases made under feature/display promotions.
   
   Example Code:
   promo_stats = est_df.select([
       (pl.col("featured") == 1).mean().alias("pct_featured"),
       (pl.col("display") == 1).mean().alias("pct_display"),
       pl.col("price_per_oz").mean().alias("mean_price_per_oz"),
   ])

2. HOUSEHOLD BRAND LOYALTY & CHOICE SET SIZE:
   - Share of household wallet devoted to their primary favorite flavor (Max Share per HH).
   - Household-level HHI for flavor choices.

   Example Code:
   hh_flavor_hhi = (
       est_df.group_by(["household_code", "flavor"])
       .len()
       .with_columns((pl.col("len") / pl.col("len").sum().over("household_code")).alias("share"))
       .group_by("household_code")
       .agg(((pl.col("share") * 100)**2).sum().alias("hh_flavor_hhi"))
   )

3. INTER-PURCHASE TIME DYNAMICS (INVENTORY EFFECTS):
   - Average days elapsed between consecutive yogurt purchase trips per household.

   Example Code:
   est_df = est_df.with_columns([
       (pl.col("purchase_date") - pl.col("purchase_date").shift(1).over("household_code"))
       .dt.total_days()
       .alias("days_since_last_purchase")
   ])
"""

print(f"\n[COMPLETE] All tables and figures successfully written to: {output_dir}\n")
