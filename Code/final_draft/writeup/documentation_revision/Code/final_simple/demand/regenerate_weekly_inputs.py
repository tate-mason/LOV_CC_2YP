"""Regenerate weekly inputs on the server without refitting saved demand parameters."""

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_data_path
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from .estimation import prepare_data, make_arrays, make_draws, parameter_names, household_loglike, add_price_controls
except ImportError:
    from estimation import prepare_data, make_arrays, make_draws, parameter_names, household_loglike, add_price_controls

# === Parameters: edit these values on the server === #
fit_dir = None
sample = None     # Optional override if the saved prepared-sample path moved.
flavors = None    # Optional override if the saved flavor-reference path moved.

# === 1. Reconstruct, verify, and save the missing export === #


@server_only
def regenerate(fit_dir, sample=None, flavors=None, weekly_output=None):
    fit_dir = Path(fit_dir)
    fit = json.loads((fit_dir / 'fit.json').read_text())
    settings = dict(fit['settings'])
    settings.setdefault('flavor_spec', 'three')
    settings.setdefault('novelty_measure', 'previous_share')
    required = ['sample', 'first_year', 'last_year', 'max_households', 'capacity',
                'no_price_control', 'household_dma_column', 'habit_weight', 'draws', 'seed']
    missing = set(required)-settings.keys()
    if missing:
        raise ValueError(f'Saved fit lacks reconstruction settings: {sorted(missing)}')
    settings['model'] = fit['model']
    settings['output'] = Path(weekly_output).parent if weekly_output is not None else Path('/scratch/dtm63837/Kilts_Panel/Data/temporary') / fit_dir.name
    for key in ['sample', 'flavors', 'products_extra']:
        if settings.get(key) is not None:
            settings[key] = Path(settings[key])
        else:
            settings[key] = None
    if sample is not None:
        settings['sample'] = Path(sample)
    if flavors is not None:
        settings['flavors'] = Path(flavors)
    args = SimpleNamespace(**settings)
    frame = prepare_data(args)
    if not args.no_price_control:
        # Reconstruct the original first stage; preserve saved inference artifacts.
        frame, _ = add_price_controls(frame)
    data = make_arrays(frame, args.capacity)
    if len(frame) != fit['weeks'] or data['n_households'] != fit['households']:
        raise ValueError('Regenerated household/week totals do not match the saved fit; no CSV written.')
    estimates = pd.read_csv(fit_dir / 'point_estimates.csv')
    parameters = estimates.set_index('parameter').estimate
    names = parameter_names(args.model, data['outside'], data['control'])
    theta = np.array([parameters[name] for name in names])
    draws = make_draws(args.draws, args.seed, args.model)
    nll = -float(household_loglike(theta, data, draws, args.model).sum())
    if not np.isfinite(nll) or not np.isclose(nll, fit['nll'], rtol=1e-9, atol=1e-4):
        raise ValueError(f'Regenerated likelihood {nll} differs from saved {fit["nll"]}; no CSV written.')
    target = Path(weekly_output) if weekly_output is not None else Path('/scratch/dtm63837/Kilts_Panel/Data/temporary') / fit_dir.name / 'weekly_inputs.csv'
    require_data_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after all checks pass, including when replacing an older export.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.weekly_inputs_', suffix='.csv', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(households=data['n_households'], weeks=len(frame), reconstructed_nll=nll,
                  saved_nll=fit['nll'], model=args.model, flavor_spec=args.flavor_spec,
                  novelty_measure=args.novelty_measure, sample=str(args.sample),
                  demand_refitted=False, price_control_refitted=not args.no_price_control)
    (fit_dir / 'weekly_inputs_regeneration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Regenerated {target}; household/week totals and saved likelihood verified.', flush=True)
    return report


# === 2. Run regeneration, using saved settings and optional overrides === #

@server_only
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, default=fit_dir)
    parser.add_argument('--sample', type=Path, default=sample, help='Override saved prepared-sample path if moved.')
    parser.add_argument('--flavors', type=Path, default=flavors, help='Override saved flavor-reference path if moved.')
    parser.add_argument('--weekly-output', type=Path, help='Temporary weekly CSV outside the code checkout.')
    args = parser.parse_args()
    if args.fit_dir is None:
        parser.error('Set fit_dir above or supply --fit-dir on the server.')
    regenerate(args.fit_dir, args.sample, args.flavors, args.weekly_output)


if __name__ == '__main__':
    main()
