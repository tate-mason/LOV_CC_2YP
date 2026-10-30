"""Supplement the flavor CSV from annual RMS product metadata on the server."""
import argparse
from pathlib import Path


from summ_stats import DATA_DIR, FLAVOR_FILE, recover_flavor_reference as recover




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument("--products", type=Path, default=DATA_DIR / "RMS/combined/products_extra.parquet")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "recovered_flavor_reference.csv")
    args = parser.parse_args()
    recover(args.flavors, args.products, args.output)


if __name__ == "__main__":
    main()
