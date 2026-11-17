"""Server-only binary theory analogue initialized at first observed yogurt baskets.

Uses the converged novelty_only fit. Household histories remain in server memory;
only initial-state counts, expected aggregates and figures are exported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

PROJECT = next(p for p in Path(__file__).resolve().parents
               if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(PROJECT/'Code/privacy'))
from server_data_policy import require_server, require_data_path


def run(fit_dir, output):
    require_server()
    fit_dir, output = Path(fit_dir).resolve(), Path(output).resolve()
    require_data_path(fit_dir)
    require_data_path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing results are preserved.')
    fit = json.loads((fit_dir/'fit.json').read_text())
    if (fit.get('spec') != 'novelty_only' or fit.get('habit') is not False
            or not fit.get('converged')):
        raise ValueError('Requires the converged no-habit novelty_only fit.')
    saved = dict(fit['settings'])
    if saved.get('no_price_control') is not True:
        raise ValueError('Requires the uncontrolled no-habit fit.')
    if saved.get('observation_mode') != 'observed_panel':
        raise ValueError('Requires the corrected observed-panel calendar.')
    for key in ['sample', 'flavors', 'panel']:
        if not saved.get(key):
            raise ValueError('Saved fit lacks '+key)
        saved[key] = Path(saved[key])
        require_data_path(saved[key])
    output.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
    sys.path.insert(0, str(PROJECT/'Code/final_simple'))
    sys.path.insert(0, str(PROJECT/'Code/final_simple/experiments'))
    sys.path.insert(0, str(PROJECT/'Code/final_simple/supply'))
    import numpy as np
    import pandas as pd
    from demand import estimation as base
    from demand_models import names, draws, objective
    from flavor_specs import categories
    from build_supply_review import build
    from plot_revenue_contributions import build as revenue_figures

    flavors = categories(saved.get('flavor_spec', 'three'))
    if set(flavors) != {'plain', 'berry', 'other'}:
        raise ValueError('This binary analogue currently requires the three-flavor fit.')
    # Reproduce estimate.py's exact preparation defaults without estimating again.
    settings = SimpleNamespace(**dict(saved, output=output, products_extra=None,
        max_households=None, household_dma_column='dma_cd', habit_weight=0.,
        stock_measure=base.DEFAULT_STOCK_MEASURE, stock_retention=base.DEFAULT_STOCK_RETENTION))
    initial = {}
    frame = base.prepare_data(settings, initial_state_collector=initial)
    data = base.make_arrays(frame, settings.capacity)
    if len(frame) != fit['weeks'] or data['n_households'] != fit['households']:
        raise ValueError('Reconstructed sample totals differ from the no-habit fit.')
    estimates = pd.read_csv(fit_dir/'point_estimates.csv')
    p = estimates.set_index('parameter').estimate
    parameters = names('novelty_only', flavors, data['control'])
    integration = draws(settings.draws, settings.seed, 'novelty_only', flavors)
    nll = objective(np.array([p[key] for key in parameters]), data, integration,
                    'novelty_only', flavors)[0]*data['n_households']
    if not np.isclose(nll, fit['nll'], rtol=1e-9, atol=1e-4):
        raise ValueError('Likelihood reconstruction differs from saved no-habit fit.')
    print('Saved household/week totals and likelihood reproduced.', flush=True)

    # First actual yogurt basket, before likelihood price exclusions. Restrict
    # to fitted households, but never replace an unknown first basket with a later one.
    first = initial['states']
    first = first[first.household_code.isin(frame.household_code.unique())].copy()
    if first.household_code.duplicated().any() or len(first) != fit['households']:
        raise ValueError('Expected one first observed yogurt basket per fitted household.')
    unknown = int((~first.state_observed).sum())
    first = first[first.state_observed].copy()
    if first.empty:
        raise ValueError('No households have an observable first-basket flavor state.')
    counts = []
    for flavor in ['plain', 'berry', 'other']:
        inside = first[flavor].to_numpy(float) > 0
        for h0, count in [('j', int(inside.sum())), ('0', int((~inside).sum()))]:
            counts.append(dict(flavor=flavor, h0=h0, households=count))
    exported = output/'RESULTS_ONLY'
    exported.mkdir()
    initial_counts = exported/'observed_initial_state_counts.csv'
    pd.DataFrame(counts).to_csv(initial_counts, index=False)
    audit = dict(
        initialization='first actual observed yogurt basket of each fitted household',
        h0='j if any focal-flavor package in initial basket; 0 otherwise',
        mixed_baskets='contain focal flavor => j; absence of focal flavor => 0',
        initial_unknown_rule='exclude household; never advance h0 to a later occasion',
        households_in_fit=int(fit['households']), included_households=len(first),
        excluded_unknown_initial_state=unknown,
        first_initial_date=str(first.week.min().date()),
        last_initial_date=str(first.week.max().date()),
        counterfactual_period1='next purchase week after initial basket; not necessarily 7 days later',
        counterfactual_period2='following purchase week; not necessarily 14 days after initial basket',
        time_index='consecutive purchase weeks, skipping zero-yogurt weeks and calendar gaps',
        future_choices='endogenous under optimized prices; not observed future baskets',
        type_distribution='same fitted population normal for both initial histories',
        quintiles='equal-probability population bins; household types are not inferred',
        information='uniform knows population distribution and history; informed also knows bin',
        learning=False, alpha='fixed at negative fitted population mean magnitude',
        beta='fixed at fitted inside constant plus flavor intercept',
        demand_refitted=False, reconstructed_nll=float(nll), saved_nll=float(fit['nll']),
        parameter_sha256=hashlib.sha256((fit_dir/'point_estimates.csv').read_bytes()).hexdigest())
    (exported/'initialization_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    build(exported, fit_dir/'point_estimates.csv', initial_counts)
    revenue_figures(exported)
    print('Complete. Recover only '+str(exported), flush=True)


if __name__ == '__main__':
    # Fail before loading any empirical dependencies or parsing an empirical path
    # on a non-server computer. There is no local execution override.
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.fit_dir, args.output)
