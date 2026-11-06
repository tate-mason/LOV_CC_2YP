#!/usr/bin/env python3
"""Compatibility launcher. Prefer the two explicit standalone entry points."""

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server
import argparse
from pathlib import Path
import subprocess
import sys


@server_only
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exercise", choices=["supply", "households", "both"], default="supply")
    args, remaining = parser.parse_known_args()
    if args.exercise == "both":
        raise SystemExit("Use run_supply_exercises.py and run_empirical_targeting.py separately; see README.md.")
    script = "run_supply_exercises.py" if args.exercise == "supply" else "run_empirical_targeting.py"
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script), *remaining], check=True)


if __name__ == "__main__":
    main()
