"""Export distributions from an existing split fit; edit Parameters or use flags."""
from pathlib import Path
import argparse
import json
import pandas as pd
try:
    from .estimation import export_history_distributions
except ImportError:
    from estimation import export_history_distributions

# === Parameters: edit these values === #
fit_dir = None
output = None  # None writes alongside the saved fit.

# === Run: read saved coefficients and export population distributions === #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, default=fit_dir)
    parser.add_argument('--output', type=Path, default=output)
    args = parser.parse_args()
    if args.fit_dir is None:
        parser.error('Set fit_dir above or supply --fit-dir.')
    fit = json.loads((args.fit_dir/'fit.json').read_text())
    if fit['model'] not in ['separate_history', 'heterogeneous_history']:
        parser.error('Requires a split-history fit')
    export_history_distributions(pd.read_csv(args.fit_dir/'point_estimates.csv'),
                                 args.output or args.fit_dir, fit.get('converged', False))

if __name__ == '__main__':
    main()
