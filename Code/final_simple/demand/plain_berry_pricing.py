"""Compare split-model pricing for plain and berry, retaining all demand categories.

Edit Parameters below or supply server flags. Uses the same household solver as
split_pricing.py. Other remains a demand alternative and history state, while
only plain and berry are considered as focal sellers in separate counterfactuals.
"""
from pathlib import Path
from types import SimpleNamespace
from pprint import pformat
import argparse
import json
import sys

import numpy as np
import pandas as pd

if __package__:
    from . import split_pricing as pricing
else:
    import split_pricing as pricing


# === Parameters: edit these values on the server === #
household_input = None   # Path to regenerated weekly_inputs.csv.
fit_dir = None           # Matching heterogeneous_history fitted-results folder.
output = Path(__file__).resolve().parents[1] / "results" / "plain_berry_pricing"
discounts = np.linspace(0., 1., 101)
demand_delta = .92
type_treatment = "posterior-mean"   # Or posterior-integrated.
batch_size = 24
save_state_policies = False

# This selects focal sellers, not the categories retained in demand estimation.
FOCAL_FLAVORS = ("plain", "berry")


# === 1. Optional command-line overrides === #
def read_command_line(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--household-input', type=Path)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--discounts', type=float, nargs='+')
    parser.add_argument('--demand-delta', type=float)
    parser.add_argument('--type-treatment', choices=['posterior-mean', 'posterior-integrated'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--save-state-policies', action='store_true')
    parser.set_defaults(**vars(settings))
    return parser.parse_args()


# === 2. Solve plain and berry separately and write their comparison === #
def run_comparison(settings):
    if settings.household_input is None or settings.fit_dir is None:
        raise ValueError('Set household_input and fit_dir, or supply their server flags.')
    grid = np.unique(np.asarray(settings.discounts, dtype=float))
    if (grid.size == 0 or not np.isfinite(grid).all() or grid[0] != 0. or grid[-1] > 1.
            or not 0 <= settings.demand_delta <= 1 or settings.batch_size < 1):
        raise ValueError('Discounts must include zero and lie in [0,1]; delta in [0,1], batch size >=1.')
    settings.output.mkdir(parents=True, exist_ok=True)
    reports, comparison = {}, []
    for focal in FOCAL_FLAVORS:
        # Each case uses the same fit and full three-category household table.
        # The shared solver moves this focal flavor into its priced coordinate.
        case = SimpleNamespace(**vars(settings))
        case.focal_flavor = focal
        case.discounts = grid
        case.output = settings.output / focal
        case.output.mkdir(parents=True, exist_ok=True)
        reports[focal] = pricing.run_households(case)
        for row in reports[focal]['aggregate']:
            comparison.append(dict(focal_flavor=focal, **row))

    pd.DataFrame(comparison).to_csv(settings.output / 'focal_flavor_comparison.csv', index=False)
    manifest = dict(
        focal_flavors=list(FOCAL_FLAVORS), demand_categories=list(pricing.FLAVORS),
        settings={key: str(value) if isinstance(value, Path) else value.tolist()
                  if isinstance(value, np.ndarray) else value for key, value in vars(settings).items()},
        results=reports, script_sha256=pricing.digest(Path(__file__).resolve()),
        solver_sha256=pricing.digest(Path(pricing.__file__).resolve()),
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    (settings.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (settings.output / 'run_notes.txt').write_text(pformat(manifest, sort_dicts=False, width=90)+'\n')
    print(f'Plain and berry comparison completed: {settings.output}')
    return manifest


# === 3. Run the comparison === #
if __name__ == '__main__':
    settings = SimpleNamespace(household_input=household_input, fit_dir=fit_dir,
        output=output, discounts=discounts, demand_delta=demand_delta,
        type_treatment=type_treatment, batch_size=batch_size,
        save_state_policies=save_state_policies)
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_comparison(settings)
