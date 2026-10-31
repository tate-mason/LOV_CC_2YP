"""Readable, unweighted summaries for the full panel and matched yogurt sample.

Run on the server: python3 summ_stats.py --help
No data are loaded when this file is imported.
"""

from pathlib import Path
import argparse
import tempfile

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


def normalize_upcs(values):
    """Canonical numeric UPC keys without floating-point conversion."""
    text = values.astype("string").str.strip()
    numeric = text.str.fullmatch(r"[0-9]+(?:\.0+)?", na=False)
    canonical = text.str.replace(r"\.0+$", "", regex=True).str.lstrip("0")
    return text.where(~numeric, canonical.replace("", "0"))


def normalized_upc_expr():
    text = pl.col("upc").cast(pl.String).str.strip_chars()
    numeric = text.str.contains(r"^[0-9]+(?:\.0+)?$")
    canonical = text.str.replace(r"\.0+$", "").str.strip_chars_start("0")
    return pl.when(numeric).then(
        pl.when(canonical == "").then(pl.lit("0")).otherwise(canonical)
    ).otherwise(text).alias("upc")


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
    price_column = next((c for c in ["price", "price_rms"] if c in names), None)
    expressions.append(
        pl.col(price_column).cast(pl.Float64).alias("price")
        if price_column
        else pl.lit(None, dtype=pl.Float64).alias("price")
    )
    purchases = (
        scan.select(expressions)
        .filter(pl.col("panel_year").is_between(first_year, last_year))
        .with_columns(
            pl.col("purchase_date").cast(pl.String).str.slice(0, 10).str.to_date(),
            normalized_upc_expr(),
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
        .agg(week.min().alias("first_week"), week.max().alias("last_week"))
        .filter(pl.col("first_week") != pl.col("last_week"))
        .select("household_code")
    )
    # Materialize only household IDs so the repeated scan cannot cache all purchases.
    ids = ids.collect(engine="streaming")
    return scan.join(ids.lazy(), on="household_code", how="semi")


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
    households = scan.group_by("household_code").agg(expressions)
    return households.with_columns(
        pl.col("race").alias("race_original"),
        pl.when(pl.col("hispanic_origin") == "1").then(pl.lit("Hispanic"))
        .otherwise(pl.col("race")).alias("race"),
        (pl.col("observed_years") * 365.25 / 7).alias("approx_observed_weeks"),
    ).with_columns(
        (pl.col("trips") / pl.col("approx_observed_weeks")).alias("trips_per_observed_week")
    )


def as_lazy(frame):
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame).lazy()
    return frame.lazy()


def household_batches(scan, work, prefix, batches):
    # Exact distinct trip/UPC counts can have large state even in streaming
    # mode. Each household belongs to one hash batch, preserving all its rows.
    for batch in range(batches):
        print(f"{prefix}: household batch {batch + 1}/{batches}", flush=True)
        subset = scan.filter(pl.col("household_code").hash(seed=0) % batches == batch)
        household_summary(subset).sink_parquet(work / f"{prefix}_{batch}.parquet", engine="streaming")
    return pl.scan_parquet(work / f"{prefix}_*.parquet")


def save_table(frame, output, name, caption):
    """Keep machine-readable CSV beside the paper-ready LaTeX table."""
    if isinstance(frame, (pl.LazyFrame, pl.DataFrame)):
        frame = as_lazy(frame).collect(engine="streaming").to_pandas()
    print(f"\n{caption} [{name}]", flush=True)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"), flush=True)
    frame.to_csv(output / f"{name}.csv", index=False)
    (output / f"{name}.tex").write_text(
        frame.to_latex(index=False, escape=True, float_format="%.2f", caption=caption)
    )


def demographic_tables(households, output, prefix):
    households = as_lazy(households)
    count = households.select(pl.len()).collect(engine="streaming").item()
    race_table = None
    for column, labels in [("race", RACE_LABELS), ("hispanic_origin", HISPANIC_LABELS),
                           ("income", INCOME_LABELS), ("household_size", {}),
                           ("head_age", AGE_LABELS), ("female_head_age", AGE_LABELS),
                           ("male_head_age", AGE_LABELS)]:
        table = (households.select(pl.col(column).cast(pl.String).fill_null("Missing").alias("code"))
                 .group_by("code").len(name="households").sort("code")
                 .collect(engine="streaming").to_pandas())
        table["category"] = table["code"].map(labels).fillna(table["code"])
        table["percent_all_households"] = 100 * table["households"] / count
        save_table(table, output, f"{prefix}_{column}",
                   f"{prefix.replace('_', ' ').title()}: {column.replace('_', ' ')} (unweighted)")
        if column == "race":
            race_table = table
    specs = [
        ("Households", pl.len()),
        ("Mean observed panel years", pl.col("observed_years").mean()),
        ("Mean trips per household", pl.col("trips").mean()),
        ("Mean trips per approximate observed week", pl.col("trips_per_observed_week").mean()),
        ("Median trips per approximate observed week", pl.col("trips_per_observed_week").median()),
        ("Mean distinct UPCs per household", pl.col("distinct_upcs").mean()),
        ("Mean purchase lines per household", pl.col("purchase_lines").mean()),
        ("Mean packages per household", pl.col("packages").mean()),
        ("Mean head age (source category code)", pl.col("head_age").cast(pl.Float64, strict=False).mean()),
        ("Households with numeric head age", pl.col("head_age").cast(pl.Float64, strict=False).count()),
        ("Mean household income (source category code)", pl.col("income").cast(pl.Float64, strict=False).mean()),
        ("Households with numeric income", pl.col("income").cast(pl.Float64, strict=False).count()),
    ]
    values = households.select([expr.alias(str(i)) for i, (_, expr) in enumerate(specs)]).collect(engine="streaming").row(0)
    rows = [(name, value) for (name, _), value in zip(specs, values)]
    rows.extend((f"Race: {row.category} (% of all households)", row.percent_all_households)
                for row in race_table.itertuples())
    save_table(pd.DataFrame(rows, columns=["metric", "value"]), output, f"{prefix}_overview",
               f"{prefix.replace('_', ' ').title()}: Household purchase summary (unweighted)")


def scan_reference(path):
    return pl.scan_csv(path, infer_schema=False,
                       null_values=["NA", "N/A", "NaN", "nan", "NULL", "null", "None", "<NA>"])


def recover_flavor_reference(reference_path, products_path, output):
    reference = scan_reference(reference_path).with_columns(normalized_upc_expr())
    if not {"upc", "flavor_descr"}.issubset(reference.collect_schema().names()):
        raise ValueError("Reference needs upc and flavor_descr columns.")
    products = pl.scan_parquet(products_path)
    names = products.collect_schema().names()
    if not {"upc", "flavor_descr"}.issubset(names):
        raise ValueError(f"{products_path}: needs upc and flavor_descr columns.")
    columns = [c for c in ["upc", "flavor_descr", "panel_year", "product_descr"] if c in names]
    products = (products.select(columns).with_columns(normalized_upc_expr())
                .join(reference.select("upc").unique(), on="upc", how="anti"))
    products = products.filter(
        pl.col("upc").is_not_null() & pl.col("flavor_descr").is_not_null()
        & ~pl.col("flavor_descr").cast(pl.String).str.strip_chars().str.to_lowercase()
        .is_in(["", "na", "n/a", "nan", "null", "none"])
    )
    if "panel_year" in names:
        recovered = products.group_by("upc").agg([
            pl.col(c).sort_by("panel_year", nulls_last=True).first().alias(c)
            for c in columns if c != "upc"
        ])
    else:
        recovered = products.unique(subset="upc", keep="first", maintain_order=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(reference_path).resolve():
        raise ValueError("Choose a separate output file to preserve the original reference.")
    recovered_path = output.with_name(output.stem + "_recovered_rows.csv")
    recovered.sink_csv(recovered_path, engine="streaming")
    additions = scan_reference(recovered_path)
    pl.concat([reference, additions], how="diagonal_relaxed").sink_csv(output, engine="streaming")
    count = additions.select(pl.len()).collect(engine="streaming").item()
    print(f"Recovered labels for {count:,} additional UPCs. Existing entries retained. Saved {output}", flush=True)
    return output


def flavor_reference(path, detailed=False, output=None):
    """Use first reference row per normalized UPC; keep missing descriptors."""
    frame = scan_reference(path)
    names = frame.collect_schema().names()
    missing = {"upc", "flavor_descr"} - set(names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.with_columns(normalized_upc_expr())
    if frame.select((pl.col("upc").is_null() | (pl.col("upc") == "")).any()).collect(engine="streaming").item():
        raise ValueError(f"{path}: flavor reference contains missing UPCs.")
    text = pl.col("flavor_descr").fill_null("")
    if "product_descr" in names:
        text = text + pl.lit(" ") + pl.col("product_descr").fill_null("")
    text = text.str.to_lowercase()
    if detailed:
        flavor = (pl.col("flavor_descr").fill_null("").str.to_lowercase().str.strip_chars()
                  .str.replace_all(r"\s+", " ").replace({"": "unknown", "unflavored": "plain"}))
    else:
        flavor = (pl.when(text.str.contains("plain|unflavored")).then(pl.lit("plain"))
                  .when(text.str.contains("berry|straw|blue|rasp|black|cran|cherry|wildberry"))
                  .then(pl.lit("berry")).otherwise(pl.lit("other")))
    frame = frame.with_columns(flavor.alias("flavor"))
    conflicting = frame.group_by("upc").agg(pl.col("flavor").n_unique().alias("n")).filter(pl.col("n") > 1)
    count = conflicting.select(pl.len()).collect(engine="streaming").item()
    if count:
        print(f"{count:,} UPCs have conflicting flavor classifications; keeping the first CSV row.", flush=True)
        if output is not None:
            name = "detailed_flavor_conflicts.csv" if detailed else "aggregated_flavor_conflicts.csv"
            frame.join(conflicting.select("upc"), on="upc", how="semi").sink_csv(output / name, engine="streaming")
    blanks = frame.select((pl.col("flavor_descr").fill_null("").str.strip_chars() == "").sum()).collect(engine="streaming").item()
    print(f"Flavor-reference rows with blank descriptors: {blanks:,}", flush=True)
    return frame.select("upc", "flavor").unique(subset="upc", keep="first", maintain_order=True).collect(engine="streaming")


def yogurt_weeks(scan, reference, flavors=FLAVORS, output=None):
    """Keep sparse household/week/flavor counts rather than a dense pivot."""
    joined = scan.join(reference.lazy(), on="upc", how="left")
    audit = joined.select(
        pl.len().alias("purchase_lines"),
        pl.col("flavor").is_null().sum().alias("unmatched_flavor_lines"),
        ((~pl.col("price").is_finite()) | (pl.col("price") <= 0)).fill_null(True).sum().alias("missing_or_nonpositive_quoted_prices"),
        ((~pl.col("quantity").is_finite()) | (pl.col("quantity") <= 0)).fill_null(True).sum().alias("invalid_quantity_lines"),
    ).collect(engine="streaming")
    if audit["unmatched_flavor_lines"][0]:
        unmatched = (joined.filter(pl.col("flavor").is_null()).group_by("upc")
                     .agg(pl.len().alias("purchase_lines"), pl.col("quantity").sum().alias("packages")))
        if output is not None:
            unmatched.sink_csv(output / "unmatched_purchase_upcs.csv", engine="streaming")
        raise ValueError("Purchased UPCs still lack flavor labels; see unmatched_purchase_upcs.csv.")
    if audit["invalid_quantity_lines"][0]:
        raise ValueError("Nonpositive or missing quantities; audit the matched sample first.")
    week = pl.col("purchase_date") + pl.duration(days=(6 - pl.col("purchase_date").dt.weekday()) % 7)
    grouped = (joined.with_columns(week.alias("week"))
               .group_by(["household_code", "week", "flavor"]).agg(pl.col("quantity").sum()))
    return grouped, audit


def flavor_summaries(weekly, output, flavors=FLAVORS):
    weekly = as_lazy(weekly)
    totals = weekly.group_by("flavor").agg(pl.col("quantity").sum().alias("packages")).collect(engine="streaming")
    if flavors != FLAVORS:
        flavors = sorted(totals["flavor"].to_list())
    # Detailed tables include observed flavors; the three estimation groups stay fixed.
    shares = (pl.DataFrame({"flavor": flavors}).join(totals, on="flavor", how="left")
              .with_columns(pl.col("packages").fill_null(0))
              .with_columns((100 * pl.col("packages") / pl.col("packages").sum()).alias("percent_packages")))
    # Match the table's two-decimal display; keep the original denominator.
    shares = shares.filter(pl.col("percent_packages").round(2) > 0).sort(
        ["percent_packages", "flavor"], descending=[True, False]
    )
    save_table(shares, output, "yogurt_flavor_shares", "Flavor shares of purchased packages")
    counts = weekly.group_by(["household_code", "flavor"]).agg(pl.col("quantity").sum())
    households = (counts.group_by("household_code").agg(
        pl.len().alias("flavors_bought"), pl.col("quantity").sum().alias("packages"),
        pl.col("quantity").pow(2).sum().alias("squared_packages"),
        pl.col("quantity").max().alias("largest_packages"))
        .with_columns((pl.col("squared_packages") / pl.col("packages").pow(2)).alias("flavor_HHI"),
                      (pl.col("largest_packages") / pl.col("packages")).alias("largest_flavor_share"))
        .select("household_code", "flavors_bought", "flavor_HHI", "largest_flavor_share"))
    hh_path = output / "yogurt_household_flavors.csv"
    households.sink_csv(hh_path, engine="streaming")
    hh_metrics = pl.scan_csv(hh_path).select(pl.len(), pl.col("flavors_bought").mean(), pl.col("flavor_HHI").mean()).collect(engine="streaming").row(0)
    baskets = (weekly.group_by(["household_code", "week"]).agg(
        pl.col("quantity").sum().alias("packages"), pl.len().alias("flavors_in_basket"),
        pl.when(pl.len() == 1).then(pl.col("flavor").first()).otherwise(None).alias("single_flavor"))
        .sort(["household_code", "week"])
        .with_columns(pl.col("single_flavor").shift().over("household_code").alias("previous_flavor"),
                      (pl.col("week").diff().over("household_code").dt.total_days() / 7).alias("gap_weeks"))
        .with_columns((pl.col("single_flavor").is_not_null() & pl.col("previous_flavor").is_not_null()
                       & (pl.col("gap_weeks") == 1)).fill_null(False).alias("eligible"))
        .with_columns((~(pl.col("eligible") & (pl.col("single_flavor") == pl.col("previous_flavor"))).fill_null(False))
                      .cast(pl.UInt64).cum_sum().alias("spell")))
    # Reuse a disk checkpoint for transitions, spells, and global medians.
    with tempfile.TemporaryDirectory(prefix="baskets-", dir=output) as tmp:
        checkpoint = Path(tmp) / "baskets.parquet"
        baskets.sink_parquet(checkpoint, engine="streaming")
        baskets = pl.scan_parquet(checkpoint)
        baskets.sink_csv(output / "yogurt_weekly_baskets.csv", engine="streaming")
        weekly.sink_csv(output / "yogurt_weekly_flavor_quantities.csv", engine="streaming")
        transitions = (baskets.filter(pl.col("eligible")).group_by(["previous_flavor", "single_flavor"])
                       .len().collect(engine="streaming").to_pandas())
        if transitions.empty:
            matrix = pd.DataFrame(0, index=flavors, columns=flavors)
        else:
            matrix = transitions.pivot(index="previous_flavor", columns="single_flavor", values="len").fillna(0).reindex(index=flavors, columns=flavors, fill_value=0)
        matrix.index.name = "previous_flavor"
        matrix.columns.name = "single_flavor"
        matrix.to_csv(output / "yogurt_transition_counts.csv")
        probabilities = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
        probabilities.to_csv(output / "yogurt_transition_probabilities.csv")
        print("\nYogurt transition counts", flush=True)
        print(matrix.to_string(), flush=True)
        print("\nYogurt transition probabilities", flush=True)
        print(probabilities.to_string(float_format=lambda v: f"{v:.2f}"), flush=True)
        spells = (baskets.filter(pl.col("single_flavor").is_not_null()).group_by(["household_code", "spell"])
                  .agg(pl.col("single_flavor").first().alias("flavor"), pl.len().alias("weeks")))
        spells_path = Path(tmp) / "spells.parquet"
        spells.sink_parquet(spells_path, engine="streaming")
        spells = pl.scan_parquet(spells_path)
        spells.sink_csv(output / "yogurt_single_flavor_spells.csv", engine="streaming")
        spell_mean = spells.select(pl.col("weeks").mean()).collect(engine="streaming").item()
        bm = baskets.select(
            pl.len().alias("weeks"), (100 * (pl.col("flavors_in_basket") > 1).mean()).alias("mixed"),
            (100 * (pl.col("packages") > 7).mean()).alias("above_seven"), pl.col("eligible").sum().alias("pairs"),
            (100 * (pl.col("single_flavor") != pl.col("previous_flavor")).filter(pl.col("eligible")).mean()).alias("switch_rate"),
            pl.col("gap_weeks").median().alias("gap"),
        ).collect(engine="streaming").row(0, named=True)
    rows = [
        ("Matched households", hh_metrics[0]), ("Observed yogurt purchase weeks", bm["weeks"]),
        ("Mean flavors bought per household", hh_metrics[1]), ("Mean household flavor HHI (0 to 1)", hh_metrics[2]),
        ("Mixed-flavor weeks (%)", bm["mixed"]), ("Weeks above seven packages (%)", bm["above_seven"]),
        ("Eligible adjacent single-flavor week pairs", bm["pairs"]), ("Switch rate in eligible pairs (%)", bm["switch_rate"]),
        ("Mean single-flavor spell length (consecutive calendar weeks)", spell_mean), ("Median gap between purchase weeks", bm["gap"]),
    ]
    save_table(pd.DataFrame(rows, columns=["metric", "value"]), output, "yogurt_dynamics", "Matched yogurt sample: purchase dynamics")
    return shares.to_pandas(), probabilities


def make_hhi_figures(output):
    """Equal-household distributions using all flavor labels, without top-20 truncation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = pd.read_csv(output / "yogurt_household_flavors.csv", usecols=["flavor_HHI"])["flavor_HHI"].dropna().to_numpy()
    if not len(values):
        return
    for kind in ["histogram", "cdf"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        if kind == "histogram":
            ax.hist(values, bins=np.linspace(0, 1, 41), weights=np.full(len(values), 100 / len(values)), color="steelblue")
            ax.set_ylabel("Households (%)")
        else:
            ax.step(np.r_[0., np.sort(values)], np.r_[0., 100 * np.arange(1, len(values) + 1) / len(values)], where="post")
            ax.set_ylabel("Cumulative households (%)")
        ax.set(xlim=(0, 1), xlabel="Household flavor HHI (quantity shares)", title="Yogurt flavor concentration")
        fig.tight_layout()
        fig.savefig(output / f"flavor_hhi_{kind}.pdf")
        plt.close(fig)


def make_figures(shares, transitions, output):
    # Limit figure dimensions; all flavors remain in the saved tables.
    shares = shares.sort_values("packages", ascending=False).head(20)
    flavors = shares["flavor"].tolist()
    transitions = transitions.reindex(index=flavors, columns=flavors)
    # Import plotting only if requested (use --no-figures for tables alone).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(flavors) * 0.35), 5))
    ax.bar(shares["flavor"], shares["percent_packages"], color="steelblue")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set(ylabel="Share of purchased packages (%)", title="Yogurt flavor shares (up to 20 most purchased flavors)")
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


def run(args, output, work):
    products_extra = args.products_extra or args.data_dir / "RMS/combined/products_extra.parquet"
    flavor_path = args.flavors
    if products_extra.exists():
        recovered_path = output / "recovered_flavor_reference.csv"
        if recovered_path.resolve() == flavor_path.resolve():
            recovered_path = output / "supplemented_flavor_reference.csv"
        recover_flavor_reference(flavor_path, products_extra, recovered_path)
        flavor_path = recovered_path
    else:
        print(f"Flavor recovery source not found: {products_extra}; using the CSV reference.", flush=True)
    sources = {}
    for prefix, filename in [("full_panel", "merged_panel.parquet"), ("matched_yogurt", "yogurt_estimation_sample.parquet")]:
        source = scan_purchases(args.data_dir / filename, args.first_year, args.last_year)
        if prefix == "matched_yogurt":
            covered = (pl.col("price").is_finite() & (pl.col("price") > 0)).fill_null(False)
            audit = source.select(pl.len().alias("purchase_lines_before_price_filter"),
                                  (~covered).sum().alias("purchase_lines_without_price_coverage"))
            save_table(audit, output, "yogurt_price_coverage", "Yogurt matched-price coverage")
            source = source.filter(covered)
            source = repeated_yogurt_households(source)
        sources[prefix] = source
        households = household_batches(source, work, prefix, args.household_batches)
        if households.select(pl.len()).collect(engine="streaming").item() == 0:
            raise ValueError(f"Empty {prefix} sample in selected years.")
        demographic_tables(households, output, prefix)
        households.sink_csv(output / f"{prefix}_households.csv", engine="streaming")
        # Exact distinct-trip counts for one year at a time bound aggregation state.
        years = source.select("panel_year").unique().sort("panel_year").collect(engine="streaming")["panel_year"]
        coverage_rows = []
        for year in years:
            coverage_rows.append(source.filter(pl.col("panel_year") == year).select(
                pl.lit(year).alias("panel_year"),
                pl.col("household_code").n_unique().alias("households"),
                pl.col("trip_code_uc").n_unique().alias("trips"), pl.len().alias("purchase_lines"),
                pl.col("quantity").sum().alias("packages")).collect(engine="streaming"))
        coverage = pl.concat(coverage_rows)
        save_table(coverage, output, f"{prefix}_year_coverage", "Sample coverage by year")
    for detailed in [True, False]:
        folder = output if detailed else output / "aggregated_flavors"
        folder.mkdir(parents=True, exist_ok=True)
        reference = flavor_reference(flavor_path, detailed=detailed, output=output)
        flavors = sorted(reference["flavor"].unique().to_list()) if detailed else FLAVORS
        if detailed:
            reference.write_csv(output / "detailed_flavor_reference.csv")
        # Process household batches independently, keeping weekly baskets long.
        audit_total = None
        tag = "detailed" if detailed else "grouped"
        for batch in range(args.household_batches):
            print(f"{tag}: yogurt week batch {batch + 1}/{args.household_batches}", flush=True)
            subset = sources["matched_yogurt"].filter(pl.col("household_code").hash(seed=0) % args.household_batches == batch)
            weekly, audit = yogurt_weeks(subset, reference, flavors, output=folder)
            weekly.sink_parquet(work / f"{tag}_weeks_{batch}.parquet", engine="streaming")
            audit_total = audit if audit_total is None else pl.concat([audit_total, audit]).select(pl.all().sum())
        if detailed:
            save_table(audit_total, output, "yogurt_sample_audit", "Yogurt sample audit")
        weekly = pl.scan_parquet(work / f"{tag}_weeks_*.parquet")
        shares, transitions = flavor_summaries(weekly, folder, flavors)
        if not args.no_figures:
            make_figures(shares, transitions, folder)
            make_hhi_figures(folder)
    # The last pass uses the estimation's three groups. Persist its exact
    # resolved mapping so estimation can consume labels without reclassifying.
    reference.write_csv(output / "estimation_flavor_reference.csv")
    print(f"Saved estimation flavor mapping to {output / 'estimation_flavor_reference.csv'}", flush=True)
    # Retain every original purchase column needed downstream, including store
    # and DMA fields that demographic projections do not otherwise need.
    eligible = sources["matched_yogurt"].select("household_code").unique().collect(engine="streaming")
    prepared = pl.scan_parquet(args.data_dir / "yogurt_estimation_sample.parquet")
    original_names = prepared.collect_schema().names()
    price_column = next((c for c in ["price", "price_rms"] if c in original_names), None)
    if price_column is None:
        raise ValueError("Yogurt purchase table needs price or price_rms for matched-price coverage.")
    prepared = prepared.filter(pl.col(price_column).cast(pl.Float64).is_finite()
                               & (pl.col(price_column).cast(pl.Float64) > 0))
    if "flavor" in original_names:
        prepared = prepared.drop("flavor")
    prepared = (prepared.filter(pl.col("panel_year").is_between(args.first_year, args.last_year))
                .with_columns(normalized_upc_expr(),
                              pl.col("purchase_date").cast(pl.String).str.slice(0, 10).str.to_date())
                .join(eligible.lazy(), on="household_code", how="semi")
                .join(reference.lazy(), on="upc", how="left"))
    sample_path = output / "yogurt_estimation_sample.parquet"
    if sample_path.resolve() == (args.data_dir / "yogurt_estimation_sample.parquet").resolve():
        raise ValueError("Choose an output folder separate from the original data directory.")
    prepared.sink_parquet(sample_path, engine="streaming")
    print(f"Saved full prepared yogurt purchase table to {sample_path}", flush=True)
    print(f"Saved summaries to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument("--products-extra", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--last-year", type=int, default=2015)
    parser.add_argument("--household-batches", type=int, default=16,
                        help="More batches reduce aggregation memory but require more scans (default 16).")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.household_batches < 1:
        parser.error("--household-batches must be positive")
    output = args.output or args.data_dir / "summary_outputs"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="summary-work-", dir=output) as tmp:
        run(args, output, Path(tmp))


if __name__ == "__main__":
    main()
