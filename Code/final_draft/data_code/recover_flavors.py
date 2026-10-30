"""Supplement the flavor CSV from annual RMS product metadata on the server."""
import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from summ_stats import DATA_DIR, FLAVOR_FILE, normalize_upcs, normalized_upc_expr, valid_flavor_labels


def recover(reference_path, products_path, output):
    reference = pd.read_csv(reference_path, dtype={"upc": str})
    if not {"upc", "flavor_descr"}.issubset(reference.columns):
        raise ValueError("Reference needs upc and flavor_descr columns.")
    reference["upc"] = normalize_upcs(reference["upc"])
    products = pl.scan_parquet(products_path)
    names = products.collect_schema().names()
    if not {"upc", "flavor_descr"}.issubset(names):
        raise ValueError(f"{products_path}: needs upc and flavor_descr columns.")
    columns = [c for c in ["upc", "flavor_descr", "panel_year", "product_descr"] if c in names]
    products = products.select(columns).with_columns(normalized_upc_expr())
    products = products.filter(
        pl.col("upc").is_not_null()
        & ~pl.col("upc").is_in(reference["upc"].dropna().tolist())
        & pl.col("flavor_descr").is_not_null()
        & ~pl.col("flavor_descr").cast(pl.String).str.strip_chars().str.to_lowercase()
        .is_in(["", "na", "n/a", "nan", "null", "none"])
    )
    # Process earlier years first; ties preserve source row order.
    if "panel_year" in names:
        products = products.sort("panel_year", maintain_order=True, nulls_last=True)
    candidates = products.unique(subset=["upc", "flavor_descr"], keep="first", maintain_order=True).collect().to_pandas()
    candidates = candidates.loc[valid_flavor_labels(candidates)]
    conflicts = candidates.groupby("upc")["flavor_descr"].nunique().gt(1).sum()
    recovered = candidates.drop_duplicates("upc", keep="first")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(reference_path).resolve():
        raise ValueError("Choose a separate output file to preserve the original reference.")
    recovered.to_csv(output.with_name(output.stem + "_recovered_rows.csv"), index=False)
    result = pd.concat([reference, recovered], ignore_index=True)
    result.to_csv(output, index=False)
    print(f"Recovered labels for {len(recovered):,} additional UPCs.", flush=True)
    print(f"Recovered UPCs with multiple labels (first retained): {conflicts:,}", flush=True)
    print(f"Existing reference entries, including blank labels, retained. Saved {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument("--products", type=Path, default=DATA_DIR / "RMS/combined/products_extra.parquet")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "recovered_flavor_reference.csv")
    args = parser.parse_args()
    recover(args.flavors, args.products, args.output)


if __name__ == "__main__":
    main()
