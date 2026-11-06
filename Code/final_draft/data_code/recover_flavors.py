"""Supplement the flavor CSV from annual RMS product metadata on the server."""

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server
import argparse
from pathlib import Path


from summ_stats import DATA_DIR, FLAVOR_FILE, recover_flavor_reference as recover




@server_only
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument("--products", type=Path, default=DATA_DIR / "RMS/combined/products_extra.parquet")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "recovered_flavor_reference.csv")
    args = parser.parse_args()
    recover(args.flavors, args.products, args.output)


if __name__ == "__main__":
    main()
