"""Readable, unweighted summaries for the full panel and matched yogurt sample.

Run on the server: python3 summ_stats.py --help
No data are loaded when this file is imported.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path("/scratch/dtm63837/Kilts_Panel/Data")
# Set this to the flavor-reference CSV's full path on the Slurm server.
FLAVOR_FILE = (
    DATA_DIR
    / "../nielsen_extracts/nielsen_extracts/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv"
)
FLAVORS = ["other", "berry", "plain"]

# Add labels ONLY after checking the codebook for the years you use.
# Income and head age are often coded categories, not dollars or exact ages.
# Example structure: RACE_LABELS = {"verified code": "verified label"}
RACE_LABELS = {}
INCOME_LABELS = {}
HISPANIC_LABELS = {}
AGE_LABELS = {}


def valid_flavor_labels(frame):
    labels = frame["flavor_descr"].fillna("").astype(str).str.strip().str.lower()
    return ~labels.isin(["", "na", "n/a", "nan", "null", "none"])


def scan_purchases(path, first_year, last_year):
    """Project needed columns before loading; accept the names in current code."""
    scan = pl.scan_parquet(path)
    names = scan.collect_schema().names()
    required = [
        "household_code",
        "panel_year",
        "purchase_date",
        "trip_code_uc",
        "upc",
        "quantity",
    ]
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    aliases = {
        "race": ["race"],
        "income": ["household_income", "income"],
        "hispanic_origin": ["hispanic_origin", "hispanic"],
        "household_size": ["household_size"],
        "female_head_age": ["female_head_age"],
        "male_head_age": ["male_head_age"],
        "dma_cd": ["dma_cd"],
    }
    expressions = [pl.col(c) for c in required]
    for target, choices in aliases.items():
        source = next((c for c in choices if c in names), None)
        expression = (
            pl.col(source).cast(pl.String) if source else pl.lit(None, dtype=pl.String)
        )
        expressions.append(expression.alias(target))
    # Price is optional: it is audited, not used to drop demographic observations.
    expressions.append(
        pl.col("price").cast(pl.Float64)
        if "price" in names
        else pl.lit(None, dtype=pl.Float64).alias("price")
    )
    purchases = (
        scan.select(expressions)
        .filter(pl.col("panel_year").is_between(first_year, last_year))
        .with_columns(
            pl.col("purchase_date").cast(pl.String).str.slice(0, 10).str.to_date(),
            pl.col("upc").cast(pl.String),
            pl.col("quantity").cast(pl.Float64),
        )
    )
    # Create overall head age from the gender-specific categories.
    # Treat zero as absent; prefer female head age when both are available.
    age_columns = ["female_head_age", "male_head_age"]
    purchases = purchases.with_columns(
        [
            pl.when(pl.col(c) == "0").then(None).otherwise(pl.col(c)).alias(c)
            for c in age_columns
        ]
    )
    return purchases.with_columns(pl.coalesce(age_columns).alias("head_age"))


def repeated_yogurt_households(scan):
    """Keep households with two distinct Saturday-ending purchase weeks.

    Apply after year selection; multiple lines/trips in one week count once.
    Summaries precede the estimation-only menu and IV exclusions.
    """
    week = pl.col("purchase_date") + pl.duration(
        days=(6 - pl.col("purchase_date").dt.weekday()) % 7
    )
    ids = (
        scan.group_by("household_code")
        .agg(week.n_unique().alias("observed_weeks"))
        .filter(pl.col("observed_weeks") >= 2)
        .select("household_code")
    )
    return scan.join(ids, on="household_code", how="semi")


def household_summary(scan):
    """One row per household, so frequent shoppers do not dominate demographics."""
    demographics = [
        "race",
        "income",
        "hispanic_origin",
        "household_size",
        "head_age",
        "female_head_age",
        "male_head_age",
        "dma_cd",
    ]
    # Zero means no head of that sex in the older processing code.
    scan = scan.with_columns(
        [
            pl.when(pl.col(c) == "0").then(None).otherwise(pl.col(c)).alias(c)
            for c in ["female_head_age", "male_head_age"]
        ]
    )
    expressions = [
        pl.col("panel_year").n_unique().alias("observed_years"),
        pl.col("panel_year").min().alias("first_year"),
        pl.col("panel_year").max().alias("last_year"),
        pl.col("trip_code_uc").n_unique().alias("trips"),
        pl.col("purchase_date").n_unique().alias("shopping_days"),
        pl.col("upc").n_unique().alias("distinct_upcs"),
        pl.len().alias("purchase_lines"),
        pl.col("quantity").sum().alias("packages"),
    ]
    # Earliest available nonmissing value, ordered by year and purchase date.
    for column in demographics:
        expressions.append(
            pl.col(column)
            .sort_by(["panel_year", "purchase_date"])
            .drop_nulls()
            .first()
            .alias(column)
        )
    households = scan.group_by("household_code").agg(expressions).collect().to_pandas()
    # Use Hispanic as a race category, following the original script's code=1.
    # Retain reported race for optional alternative tabulations.
    households["race_original"] = households["race"]
    households.loc[households["hispanic_origin"] == "1", "race"] = "Hispanic"
    # Only observed panel years count: a gap between years is NOT enrollment.
    # This is an approximate full-year exposure measure, not verified enrollment.
    households["approx_observed_weeks"] = households["observed_years"] * 365.25 / 7
    households["trips_per_observed_week"] = (
        households["trips"] / households["approx_observed_weeks"]
    )
    return households


def save_table(frame, output, name, caption):
    """Keep machine-readable CSV beside the paper-ready LaTeX table."""
    print(f"\n{caption} [{name}]", flush=True)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"), flush=True)
    frame.to_csv(output / f"{name}.csv", index=False)
    (output / f"{name}.tex").write_text(
        frame.to_latex(index=False, escape=True, float_format="%.2f", caption=caption)
    )


def demographic_tables(households, output, prefix):
    """Main race table includes Hispanic; origin is also available separately."""
    categories = [
        ("race", RACE_LABELS),
        ("hispanic_origin", HISPANIC_LABELS),
        ("income", INCOME_LABELS),
        ("household_size", {}),
        ("head_age", AGE_LABELS),
        ("female_head_age", AGE_LABELS),
        ("male_head_age", AGE_LABELS),
    ]
    for column, labels in categories:
        values = households[column].fillna("Missing").astype(str)
        counts = values.value_counts(dropna=False).sort_index()
        table = counts.rename_axis("code").reset_index(name="households")
        table["category"] = table["code"].map(labels).fillna(table["code"])
        # Missing is part of the denominator and appears as its own row.
        table["percent_all_households"] = 100 * table["households"] / len(households)
        save_table(
            table,
            output,
            f"{prefix}_{column}",
            f"{prefix.replace('_', ' ').title()}: {column.replace('_', ' ')} (unweighted)",
        )
    metrics = pd.DataFrame(
        {
            "metric": [
                "Households",
                "Mean observed panel years",
                "Mean trips per household",
                "Mean trips per approximate observed week",
                "Median trips per approximate observed week",
                "Mean distinct UPCs per household",
                "Mean purchase lines per household",
                "Mean packages per household",
            ],
            "value": [
                len(households),
                households["observed_years"].mean(),
                households["trips"].mean(),
                households["trips_per_observed_week"].mean(),
                households["trips_per_observed_week"].median(),
                households["distinct_upcs"].mean(),
                households["purchase_lines"].mean(),
                households["packages"].mean(),
            ],
        }
    )
    # These source fields are categorical codes. Do not label their means as
    # years or dollars without a verified codebook conversion.
    age = pd.to_numeric(households["head_age"], errors="coerce")
    income = pd.to_numeric(households["income"], errors="coerce")
    demographic_metrics = [
        ("Mean head age (source category code)", age.mean()),
        ("Households with numeric head age", age.count()),
        ("Mean household income (source category code)", income.mean()),
        ("Households with numeric income", income.count()),
    ]
    race = households["race"].fillna("Missing").astype(str)
    for code, count in race.value_counts().sort_index().items():
        label = RACE_LABELS.get(code, code)
        demographic_metrics.append(
            (f"Race: {label} (% of all households)", 100 * count / len(households))
        )
    metrics = pd.concat(
        [metrics, pd.DataFrame(demographic_metrics, columns=["metric", "value"])],
        ignore_index=True,
    )
    save_table(
        metrics, output, f"{prefix}_overview",
        f"{prefix.replace('_', ' ').title()}: Household purchase summary (unweighted)",
    )


def flavor_reference(path, detailed=False, output=None):
    """Classify descriptors; module/product_module_code is not flavor text."""
    frame = pd.read_csv(path, dtype={"upc": str})
    missing = {"upc", "flavor_descr"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["upc"] = frame["upc"].str.strip()
    if frame["upc"].isna().any() or frame["upc"].eq("").any():
        raise ValueError(f"{path}: flavor reference contains missing UPCs.")
    # Missing-flavor exclusion disabled pending checks of older reference files.
    # has_description = valid_flavor_labels(frame)
    # print(f"Dropping {(~has_description).sum():,} reference rows with missing/NA flavor.", flush=True)
    # frame = frame.loc[has_description].copy()
    # if frame.empty:
    #     raise ValueError(f"{path}: no nonmissing flavor labels remain.")
    text = frame["flavor_descr"].fillna("")
    if "product_descr" in frame.columns:
        text = text + " " + frame["product_descr"].fillna("")
    text = text.str.lower()
    # Plain takes precedence over berry, matching the estimation script.
    frame["flavor"] = np.select(
        [
            text.str.contains("plain|unflavored"),
            text.str.contains("berry|straw|blue|rasp|black|cran|cherry|wildberry"),
        ],
        ["plain", "berry"],
        default="other",
    )
    if detailed:
        # Preserve individual reported flavors and combinations; never infer a
        # single flavor from one keyword in a mixed-flavor description.
        frame["flavor"] = (
            frame["flavor_descr"].fillna("").str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace({"": "unknown", "unflavored": "plain"})
        )
    counts = frame.groupby("upc")["flavor"].nunique()
    conflicts = frame.loc[frame["upc"].isin(counts[counts > 1].index)].sort_values("upc")
    if not conflicts.empty:
        columns = [c for c in ["upc", "flavor_descr", "product_descr", "module", "product_module_code", "flavor"] if c in conflicts]
        conflicts = conflicts[columns].drop_duplicates()
        print("\nConflicting flavor-reference rows (first 30):", flush=True)
        print(conflicts.head(30).to_string(index=False), flush=True)
        location = ""
        if output is not None:
            conflict_path = output / ("detailed_flavor_conflicts.csv" if detailed else "aggregated_flavor_conflicts.csv")
            conflicts.to_csv(conflict_path, index=False)
            location = f" See {conflict_path}."
        print(
            f"{len(counts[counts > 1]):,} UPCs have conflicting flavor classifications; "
            "keeping the first flavor in CSV row order."
            + location,
            flush=True,
        )
    print(
        f"Flavor-reference rows with blank descriptors: {(frame['flavor_descr'].fillna('').str.strip() == '').sum():,}"
    )
    return pl.from_pandas(frame[["upc", "flavor"]].drop_duplicates(subset="upc", keep="first"))


def yogurt_weeks(scan, reference, flavors=FLAVORS):
    """Aggregate mixed baskets before computing flavor shares or transitions."""
    joined = scan.join(reference.lazy(), on="upc", how="left")
    audit = (
        joined.select(
            pl.len().alias("purchase_lines"),
            pl.col("flavor").is_null().sum().alias("unmatched_flavor_lines"),
            ((~pl.col("price").is_finite()) | (pl.col("price") <= 0))
            .fill_null(True)
            .sum()
            .alias("missing_or_nonpositive_quoted_prices"),
            ((~pl.col("quantity").is_finite()) | (pl.col("quantity") <= 0))
            .fill_null(True)
            .sum()
            .alias("invalid_quantity_lines"),
        )
        .collect()
        .to_pandas()
    )
    if audit.loc[0, "unmatched_flavor_lines"]:
        raise ValueError(
            "Purchased UPCs lack flavor labels; update the flavor reference."
        )
    if audit.loc[0, "invalid_quantity_lines"]:
        raise ValueError(
            "Nonpositive or missing quantities; audit the matched sample first."
        )
    joined = joined.with_columns(
        (
            pl.col("purchase_date")
            + pl.duration(days=(6 - pl.col("purchase_date").dt.weekday()) % 7)
        ).alias("week")
    )
    grouped = (
        joined.group_by(["household_code", "week", "flavor"])
        .agg(pl.col("quantity").sum())
        .collect()
        .to_pandas()
    )
    weekly = (
        grouped.pivot(
            index=["household_code", "week"], columns="flavor", values="quantity"
        )
        .fillna(0)
        .reindex(columns=flavors, fill_value=0)
        .reset_index()
    )
    weekly["week"] = pd.to_datetime(weekly["week"])
    weekly = weekly.sort_values(["household_code", "week"]).reset_index(drop=True)
    return weekly, audit


def flavor_summaries(weekly, output, flavors=FLAVORS):
    totals = weekly[flavors].sum()
    shares = pd.DataFrame(
        {
            "flavor": flavors,
            "packages": totals.to_numpy(),
            "percent_packages": 100 * totals.to_numpy() / totals.sum(),
        }
    )
    save_table(
        shares, output, "yogurt_flavor_shares", "Flavor shares of purchased packages"
    )
    hh_counts = weekly.groupby("household_code")[flavors].sum()
    hh_shares = hh_counts.div(hh_counts.sum(axis=1), axis=0)
    households = pd.DataFrame(
        {
            "flavors_bought": (hh_counts > 0).sum(axis=1),
            "flavor_HHI": (hh_shares**2).sum(axis=1),
            "largest_flavor_share": hh_shares.max(axis=1),
        }
    )
    households.to_csv(output / "yogurt_household_flavors.csv")
    weekly["packages"] = weekly[flavors].sum(axis=1)
    weekly["flavors_in_basket"] = (weekly[flavors] > 0).sum(axis=1)
    # No arbitrary ordering of multiple flavors bought on the same trip/week.
    weekly["single_flavor"] = (
        weekly[flavors].idxmax(axis=1).where(weekly["flavors_in_basket"] == 1)
    )
    group = weekly.groupby("household_code")
    weekly["previous_flavor"] = group["single_flavor"].shift()
    weekly["gap_weeks"] = group["week"].diff().dt.days / 7
    eligible = (
        weekly["single_flavor"].notna()
        & weekly["previous_flavor"].notna()
        & (weekly["gap_weeks"] == 1)
    )
    transitions = pd.crosstab(
        weekly.loc[eligible, "previous_flavor"], weekly.loc[eligible, "single_flavor"]
    )
    transitions = transitions.reindex(index=flavors, columns=flavors, fill_value=0)
    print("\nYogurt transition counts (previous flavor → current flavor)", flush=True)
    print(transitions.to_string(), flush=True)
    transitions.to_csv(output / "yogurt_transition_counts.csv")
    probabilities = transitions.div(transitions.sum(axis=1).replace(0, np.nan), axis=0)
    print("\nYogurt transition probabilities (previous flavor → current flavor)", flush=True)
    print(probabilities.to_string(float_format=lambda value: f"{value:.2f}"), flush=True)
    probabilities.to_csv(output / "yogurt_transition_probabilities.csv")
    # Include STAYS in the denominator; otherwise a switching rate is overstated.
    switches = (
        weekly.loc[eligible, "single_flavor"] != weekly.loc[eligible, "previous_flavor"]
    )
    continued = eligible & (weekly["single_flavor"] == weekly["previous_flavor"])
    weekly["spell"] = (~continued).cumsum()
    spells = (
        weekly.loc[weekly["single_flavor"].notna()]
        .groupby(["household_code", "spell"])
        .agg(flavor=("single_flavor", "first"), weeks=("week", "size"))
    )
    spells.to_csv(output / "yogurt_single_flavor_spells.csv")
    metrics = pd.DataFrame(
        {
            "metric": [
                "Matched households",
                "Observed yogurt purchase weeks",
                "Mean flavors bought per household",
                "Mean household flavor HHI (0 to 1)",
                "Mixed-flavor weeks (%)",
                "Weeks above seven packages (%)",
                "Eligible adjacent single-flavor week pairs",
                "Switch rate in eligible pairs (%)",
                "Mean single-flavor spell length (consecutive calendar weeks)",
                "Median gap between purchase weeks",
            ],
            "value": [
                len(households),
                len(weekly),
                households["flavors_bought"].mean(),
                households["flavor_HHI"].mean(),
                100 * (weekly["flavors_in_basket"] > 1).mean(),
                100 * (weekly["packages"] > 7).mean(),
                eligible.sum(),
                100 * switches.mean(),
                spells["weeks"].mean(),
                weekly["gap_weeks"].median(),
            ],
        }
    )
    save_table(
        metrics, output, "yogurt_dynamics", "Matched yogurt sample: purchase dynamics"
    )
    weekly.to_csv(output / "yogurt_weekly_baskets.csv", index=False)
    return shares, probabilities


def make_figures(shares, transitions, output):
    flavors = shares["flavor"].tolist()
    # Import plotting only if requested (use --no-figures for tables alone).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(flavors) * 0.35), 5))
    ax.bar(shares["flavor"], shares["percent_packages"], color="steelblue")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set(ylabel="Share of purchased packages (%)", title="Yogurt flavor shares")
    fig.tight_layout()
    fig.savefig(output / "flavor_shares.pdf")
    plt.close(fig)
    side = max(6, len(flavors) * 0.4)
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(transitions.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set(
        xticks=range(len(flavors)),
        yticks=range(len(flavors)),
        xticklabels=flavors,
        yticklabels=flavors,
        xlabel="Current flavor",
        ylabel="Previous flavor",
        title="Adjacent single-flavor purchase weeks",
    )
    ax.tick_params(axis="x", labelrotation=90)
    for i in range(len(flavors)):
        for j in range(len(flavors)):
            value = transitions.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}" if pd.notna(value) else "N/A",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "flavor_transitions.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--flavors",
        type=Path,
        default=FLAVOR_FILE,
        help="Flavor-reference CSV (defaults to FLAVOR_FILE set above).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--last-year", type=int, default=2015)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    output = args.output or args.data_dir / "summary_outputs"
    output.mkdir(parents=True, exist_ok=True)
    print("Summarizing household demographics and purchases...", flush=True)
    panel = scan_purchases(
        args.data_dir / "merged_panel.parquet", args.first_year, args.last_year
    )
    yogurt = scan_purchases(
        args.data_dir / "yogurt_estimation_sample.parquet",
        args.first_year,
        args.last_year,
    )
    yogurt = repeated_yogurt_households(yogurt)
    panel_households = household_summary(panel)
    yogurt_households = household_summary(yogurt)
    if panel_households.empty or yogurt_households.empty:
        raise ValueError("Empty panel or yogurt sample in selected years.")
    demographic_tables(panel_households, output, "full_panel")
    demographic_tables(yogurt_households, output, "matched_yogurt")
    panel_households.to_csv(output / "full_panel_households.csv", index=False)
    yogurt_households.to_csv(output / "matched_yogurt_households.csv", index=False)
    for name, source in [("full_panel", panel), ("matched_yogurt", yogurt)]:
        coverage = (
            source.group_by("panel_year")
            .agg(
                pl.col("household_code").n_unique().alias("households"),
                pl.col("trip_code_uc").n_unique().alias("trips"),
                pl.len().alias("purchase_lines"),
                pl.col("quantity").sum().alias("packages"),
            )
            .sort("panel_year")
            .collect()
            .to_pandas()
        )
        save_table(coverage, output, f"{name}_year_coverage", "Sample coverage by year")
    # Exclude purchases only when their reference UPC has no valid flavor.
    # UPCs absent from the reference still fail the unmatched-flavor audit.
    # Disabled pending checks of older reference files.
    # raw_reference = pd.read_csv(args.flavors, dtype={"upc": str})
    # raw_reference["upc"] = raw_reference["upc"].str.strip()
    # valid_labels = valid_flavor_labels(raw_reference)
    # excluded_upcs = set(raw_reference.loc[~valid_labels, "upc"].dropna()) - set(
    #     raw_reference.loc[valid_labels, "upc"].dropna()
    # )
    # flavor_yogurt = yogurt.filter(~pl.col("upc").is_in(sorted(excluded_upcs)))
    # dropped_lines = yogurt.filter(pl.col("upc").is_in(sorted(excluded_upcs))).select(pl.len()).collect().item()
    # print(f"Excluded {len(excluded_upcs):,} UPCs and {dropped_lines:,} purchase lines with missing/NA flavor from flavor summaries.", flush=True)
    flavor_yogurt = yogurt
    reference = flavor_reference(args.flavors, detailed=True, output=output)
    flavors = sorted(reference["flavor"].unique().to_list())
    reference.write_csv(output / "detailed_flavor_reference.csv")
    print("\nDetailed flavors use normalized flavor_descr; missing labels are retained as unknown.", flush=True)
    weekly, audit = yogurt_weeks(flavor_yogurt, reference, flavors)
    print("\nYogurt sample audit", flush=True)
    print(audit.to_string(index=False), flush=True)
    audit.to_csv(output / "yogurt_sample_audit.csv", index=False)
    shares, transitions = flavor_summaries(weekly, output, flavors)
    if not args.no_figures:
        make_figures(shares, transitions, output)
    # Keep the estimation's three flavor groups as a separate comparison.
    grouped_output = output / "aggregated_flavors"
    grouped_output.mkdir(parents=True, exist_ok=True)
    grouped_weekly, _ = yogurt_weeks(flavor_yogurt, flavor_reference(args.flavors, output=output))
    print("\nThree-group comparison (other, berry, plain)", flush=True)
    grouped_shares, grouped_transitions = flavor_summaries(grouped_weekly, grouped_output)
    if not args.no_figures:
        make_figures(grouped_shares, grouped_transitions, grouped_output)
    print(f"Saved summaries to {output}", flush=True)

    # OPTIONAL: compare flavor loyalty by income (codes stay categorical).
    # loyalty = pd.read_csv(output / "yogurt_household_flavors.csv")
    # comparison = yogurt_households.merge(loyalty, on="household_code", validate="one_to_one")
    # comparison.groupby("income", dropna=False)["flavor_HHI"].agg(["count", "mean"]).to_csv(
    #     output / "flavor_loyalty_by_income.csv")

    # OPTIONAL: distribution of the actual estimation inputs AFTER menu/IV exclusions.
    # inputs = pd.read_csv("/path/to/estimation_output/weekly_inputs.csv")
    # inputs.filter(regex=r"^(price_|iv_|cf_|stock_|novelty_|xi_)").describe().T.to_csv(
    #     output / "estimation_variable_summary.csv")

    # OPTIONAL: demographic selection into the final estimation sample.
    # ids = inputs[["household_code"]].drop_duplicates()
    # final_households = yogurt_households.merge(ids, on="household_code", validate="one_to_one")
    # demographic_tables(final_households, output, "final_estimation")

    # OPTIONAL: survey-weighted demographics require household-year weights and
    # an explicit choice of reference year. Do not weight repeated purchase lines.


if __name__ == "__main__":
    main()
