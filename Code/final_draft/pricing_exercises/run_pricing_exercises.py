#!/usr/bin/env python3
"""Compatibility launcher. Prefer the two explicit standalone entry points."""
import argparse
from pathlib import Path
import subprocess
import sys


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
