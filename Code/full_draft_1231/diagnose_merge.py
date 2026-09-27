"""Cluster-only merge diagnostics; prints column names and aggregate counts.

Run after data_merge.py has written merge_filtered_hms.parquet:
    python -u diagnose_merge.py --year 2022
No individual keys or source values are printed.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import data_merge as merge
import polars as pl


def conflict_counts(retail, columns, batch_size=8):
    """Count keys with multiple distinct values, including null versus non-null."""
    results = []
    for start in range(0, len(columns), batch_size):
        batch = columns[start : start + batch_size]
        aliases = [f"n_{i}" for i in range(len(batch))]
        summary = (
            retail.group_by(merge.JOIN_KEYS)
            .agg([
                pl.col(column).n_unique().alias(alias)
                for column, alias in zip(batch, aliases)
            ])
            .select([(pl.col(alias) > 1).sum() for alias in aliases])
            .collect(engine="streaming")
        )
        results.extend(
            (column, summary.item(0, alias))
            for column, alias in zip(batch, aliases)
        )
    return sorted(results, key=lambda item: (-item[1], item[0]))


def main(year):
    cache = Path(merge.OUTPUT_DIR) / "merge_filtered_hms.parquet"
    if not cache.exists():
        raise SystemExit("Run data_merge.py first to create the filtered HMS cache.")
    panel = pl.scan_parquet(cache).filter(pl.col("week_end").dt.year() == year)
    keys = panel.select(merge.JOIN_KEYS).unique()
    retail = merge.build_retail().filter(pl.col("week_end").dt.year() == year)
    panel_rows = merge.count(f"{year} filtered panel rows", panel)

    # Scanner conversion reads YOGURT_{year}.tsv; report its target category
    # separately from the broader panel sample used for outside-option work.
    yogurt = panel.filter(pl.col("yogurt_purchase") == 1)
    yogurt_rows = merge.count("Panel yogurt purchase rows", yogurt)
    yogurt_matches = merge.count(
        "Panel yogurt purchase rows with full scanner key match",
        yogurt.select(merge.JOIN_KEYS).join(
            retail.select(merge.JOIN_KEYS).unique(), on=merge.JOIN_KEYS, how="semi"
        ),
    )
    if yogurt_rows:
        merge.log(f"Yogurt full-key coverage: {yogurt_matches / yogurt_rows:.2%}")

    # These marginal matches help locate mismatches but do not validate a
    # crosswalk or justify dropping any key from the actual merge.
    for subset in [["store_code_uc"], ["upc"], ["week_end"], merge.JOIN_KEYS]:
        matched = merge.count(
            "Panel rows matching scanner on " + ", ".join(subset),
            panel.select(subset).join(
                retail.select(subset).unique(), on=subset, how="semi"
            ),
        )
        if panel_rows:
            merge.log(f"Coverage: {matched / panel_rows:.2%}")

    # Persist only relevant scanner rows on the cluster, so column batches do
    # not each rescan the full scanner source. Temporary files are removed.
    with TemporaryDirectory(prefix="merge_diagnostic_", dir=merge.OUTPUT_DIR) as work:
        path = Path(work) / "matched_scanner.parquet"
        merge.log("START saving scanner rows matching panel keys for diagnosis")
        retail.join(keys, on=merge.JOIN_KEYS, how="semi").drop(merge.ROW_ID).sink_parquet(
            path, engine="streaming"
        )
        matched_retail = pl.scan_parquet(path)
        sizes = matched_retail.group_by(merge.JOIN_KEYS).agg(pl.len().alias("n"))
        summary = sizes.select(
            pl.len().alias("matched_keys"),
            (pl.col("n") > 1).sum().alias("duplicate_keys"),
            pl.col("n").max().alias("max_rows_per_key"),
        ).collect(engine="streaming")
        merge.log(str(summary))
        columns = [c for c in matched_retail.collect_schema().names() if c not in merge.JOIN_KEYS]
        merge.log("Column -> number of matched scanner keys with differing values")
        results = conflict_counts(matched_retail, columns)
        for column, conflicts in results:
            if conflicts:
                merge.log(f"{column}: {conflicts:,}")
        merge.log(f"Columns with no conflicts: {sum(n == 0 for _, n in results)}")
    merge.log("DONE diagnostics; no merge outputs changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022)
    main(parser.parse_args().year)
