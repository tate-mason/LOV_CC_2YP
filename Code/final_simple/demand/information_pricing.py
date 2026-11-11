"""Server-only history-adaptive prices; only novelty preference is hidden.

Reuses an existing converged habit-inclusive fit. Writes aggregate results only.
No demand refit, binary price calibration, or individual policy export.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(ROOT/'Code/privacy'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server_data_policy import server_only, require_data_path, require_server
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
try:
    from . import estimation, split_pricing
    from .history import saved_stock_measure
    from .information_pricing_model import solve_environment
except ImportError:
    import estimation, split_pricing
    from history import saved_stock_measure
    from information_pricing_model import solve_environment
from flavor_specs import categories


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@server_only
def run(args):
    """Reconstruct and verify saved demand, then export aggregate policy comparisons."""
    require_data_path(args.fit_dir)
    require_data_path(args.output)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing results are preserved.')
    if args.nodes < 3 or args.nodes > 41 or args.search_points < 17:
        raise ValueError('Use 3..41 quadrature nodes and at least 17 search points.')
    if not 0 <= args.delta <= 1 or (args.max_environments is not None and args.max_environments < 1):
        raise ValueError('Check revenue discount factor and environment limit.')
    fit = json.loads((args.fit_dir/'fit.json').read_text())
    if fit['model'] != 'heterogeneous_history' or not fit.get('converged', False):
        raise ValueError('A converged heterogeneous-history fit is required.')
    settings = dict(fit['settings'])
    if settings.get('novelty_measure', 'previous_share') != 'previous_share':
        raise ValueError('This transition requires the saved previous-share novelty definition.')
    measure = saved_stock_measure(settings)
    settings['stock_measure'] = measure
    settings.setdefault('stock_retention', 1.)
    settings.setdefault('flavor_spec', 'three')
    settings.setdefault('novelty_measure', 'previous_share')
    for key in ['sample', 'flavors', 'products_extra']:
        settings[key] = Path(settings[key]) if settings.get(key) is not None else None
    for key in ['sample', 'flavors']:
        override = getattr(args, key, None)
        if override is not None:
            require_data_path(override)
            settings[key] = override
    for key in ['sample', 'flavors', 'products_extra']:
        if settings[key] is not None:
            require_data_path(settings[key])
    settings['output'] = args.output
    selected = categories(settings['flavor_spec'])
    estimation.FLAVORS[:] = selected
    split_pricing.FLAVORS[:] = selected
    parameters_table = pd.read_csv(args.fit_dir/'point_estimates.csv')
    if 'at_bound' in parameters_table and parameters_table.at_bound.astype(str).str.lower().eq('true').any():
        raise ValueError('Review fitted boundary parameters before counterfactual pricing.')
    parameters = dict(zip(parameters_table.parameter, parameters_table.estimate))
    if parameters['sd_novelty'] < 0:
        raise ValueError('Invalid fitted novelty dispersion.')
    # Server-only in-memory reconstruction; do not replace any files in fit_dir.
    frame = estimation.prepare_data(SimpleNamespace(**settings))
    control = not settings.get('no_price_control', False)
    if control:
        frame, _ = estimation.add_price_controls(frame)
    if frame.duplicated(['household_code', 'week']).any():
        raise ValueError('Duplicate household/week rows.')
    if len(frame) != fit['weeks'] or frame.household_code.nunique() != fit['households']:
        raise ValueError('Reconstructed sample totals do not match saved fit.')
    draws = split_pricing.persistent_draws(settings['draws'], settings['seed'])
    households, _, _, _, beta, nll = split_pricing.posterior(frame, parameters, draws, control)
    if not np.isclose(nll, fit['nll'], rtol=1e-9, atol=1e-4):
        raise ValueError('Reconstructed likelihood differs from saved fit; no pricing results written.')
    context = split_pricing.forecast_context(frame, households, settings['stock_retention'], measure)
    first = frame.sort_values(['household_code', 'week']).groupby('household_code', sort=True).head(1)
    first = first.set_index('household_code').loc[households.household_code]
    nodes, weights = hermgauss(args.nodes)
    gamma = parameters['mean_novelty']+np.sqrt(2)*parameters['sd_novelty']*nodes
    weights = weights/np.sqrt(np.pi)
    # Cross the fitted population novelty distribution with each fixed environment.
    # Only alpha/habit are treated as known posterior means, in BOTH regimes.
    cutoff = .6744897501960817*parameters['sd_novelty']
    labels = np.where(gamma < parameters['mean_novelty']-cutoff, 'low',
                      np.where(gamma > parameters['mean_novelty']+cutoff, 'high', 'middle'))
    scenarios = [('Fitted distribution', weights)]
    if all(np.any(labels == group) for group in ['low', 'middle', 'high']):
        for favored in ['low', 'middle', 'high']:
            target = {'low': .25, 'middle': .25, 'high': .25}
            target[favored] = .5
            alternative = weights.copy()
            for group in target:
                mask = labels == group
                alternative[mask] *= target[group]/weights[mask].sum()
            scenarios.append((favored.title()+' type more likely', alternative))
    flavors = ['plain', 'berry'] if args.flavor == 'both' else [args.flavor]
    if not set(flavors) <= set(selected):
        raise ValueError('Requested focal flavor is absent from saved fit.')
    capacity = settings['capacity']
    if not isinstance(capacity, int) or not 1 <= capacity <= 12:
        raise ValueError('Saved capacity must be an integer in 1..12.')
    args.output.mkdir(parents=True)
    reports = {}
    for flavor in flavors:
        focal = selected.index(flavor)
        eligible = np.flatnonzero(context['available'][:, focal])
        if args.max_environments is not None:
            eligible = eligible[:args.max_environments]
        if not len(eligible):
            raise ValueError('No eligible pricing environments for '+flavor)
        sums = {}
        total_environments = {'inside': 0, 'outside': 0}
        max_mass_error = 0.
        for number, index in enumerate(eligible, 1):
            state = 'inside' if first.iloc[index][flavor] > 0 else 'outside'
            total_environments[state] += 1
            posted = context['prices'][index, focal]
            for scenario, prior in scenarios:
                result = solve_environment(
                    prices=context['prices'][index], available=context['available'][index], beta=beta,
                    novelty=context['novelty'][index], stock=context['stocks'][index],
                    alpha=float(households.iloc[index].alpha_mean),
                    habit=float(households.iloc[index].habit_mean), gamma=gamma, weights=prior,
                    capacity=capacity, focal=focal, delta=args.delta, stock_measure=measure,
                    stock_retention=settings['stock_retention'],
                    occasions=int(context['stock_occasions'][index, 0]),
                    controls=parameters.get('price_control', 0.)*context['cf'][index],
                    search_points=args.search_points)
                for regime in ['full', 'limited']:
                    values = result[regime]
                    max_mass_error = max(max_mass_error,
                        float(np.max(np.abs(values['state_probability'].sum(axis=1)-1))))
                    for k, (g, weight, group) in enumerate(zip(gamma, prior, labels)):
                        key = (scenario, state, regime, k, group)
                        metrics = dict(weight=weight, posted=weight*posted,
                            p1=weight*values['p1'][k], expected_p2=weight*values['expected_p2'][k],
                            change_p1=weight*(values['p1'][k]-posted),
                            change_p2=weight*(values['expected_p2'][k]-posted),
                            pct_change_p1=weight*100*(values['p1'][k]/posted-1),
                            pct_change_p2=weight*100*(values['expected_p2'][k]/posted-1),
                            revenue1=weight*values['revenue1'][k], revenue2=weight*values['revenue2'][k],
                            revenue=weight*values['revenue'][k])
                        p2 = result['p2_full'][k] if regime == 'full' else result['p2_limited']
                        terminal_revenue = result['terminal_revenue_'+regime][k]
                        mass = values['state_probability'][k]
                        for branch, mask in [('inside', result['states'][:, focal] > 0),
                                             ('outside', result['states'][:, focal] == 0)]:
                            metrics['mass_'+branch] = weight*mass[mask].sum()
                            metrics['p2_'+branch] = weight*(mass[mask] @ p2[mask])
                            realized = (result['states'][mask, focal]*values['p1'][k]
                                        +args.delta*terminal_revenue[mask])
                            metrics['profit_'+branch] = weight*(mass[mask] @ realized)
                            metrics['ratio_'+branch] = weight*(mass[mask] @ (values['p1'][k]/p2[mask]))
                        record = sums.setdefault(key, {name: 0. for name in metrics})
                        for name, value in metrics.items():
                            record[name] += float(value)
            if number % 100 == 0 or number == len(eligible):
                print(f'{flavor}: solved {number}/{len(eligible)} environments', flush=True)
        records = [dict(scenario=scenario, initial_state=state, regime=regime, node=k, group=group,
                        gamma=float(gamma[k]), **values)
                   for (scenario, state, regime, k, group), values in sums.items()]
        node_totals = pd.DataFrame(records)
        metric_columns = [c for c in node_totals if c not in ['scenario', 'initial_state', 'regime', 'node', 'group', 'gamma']]
        destination = args.output/flavor
        destination.mkdir()
        node_totals.to_csv(destination/'type_node_aggregate.csv', index=False)
        by_type = node_totals.groupby(['scenario', 'initial_state', 'regime', 'group'], as_index=False)[metric_columns].sum()
        by_type.to_csv(destination/'price_paths_by_type.csv', index=False)
        totals = node_totals.groupby(['scenario', 'initial_state', 'regime'], as_index=False)[metric_columns].sum()
        totals.to_csv(destination/'revenue_comparison.csv', index=False)
        comparison = totals.pivot(index=['scenario', 'initial_state'], columns='regime', values='revenue')
        comparison['gain'] = comparison['full']-comparison['limited']
        comparison['gain_pct'] = 100*comparison['gain']/comparison['limited'].where(comparison['limited'] > 0)
        comparison.to_csv(destination/'information_gain.csv')
        reports[flavor] = dict(environments=len(eligible), initial_states=total_environments,
                              max_mass_error=max_mass_error)
    summary = dict(status='pricing_complete', model='multinomial habit-inclusive demand',
        demand_refitted=False, saved_fit_nll=float(fit['nll']), reconstructed_nll=float(nll),
        fit_sha256=_hash(args.fit_dir/'fit.json'),
        parameters_sha256=_hash(args.fit_dir/'point_estimates.csv'),
        solver_sha256=_hash(Path(__file__).with_name('information_pricing_model.py')),
        flavor_categories=selected, stock_measure=measure, stock_retention=settings['stock_retention'],
        novelty_measure=settings['novelty_measure'], capacity=capacity, delta=args.delta,
        hidden='novelty preference only', known='price sensitivity and habit: estimated posterior means; histories and menus',
        beliefs='fitted population normal novelty distribution; fixed in every state',
        gamma_nodes=gamma.tolist(), gamma_weights=weights.tolist(), quadrature_nodes=args.nodes,
        prior_scenarios=[dict(name=name, weights=w.tolist()) for name, w in scenarios],
        evaluation='same persistent gamma types under both policies',
        limited_objective='fixed-belief objective at each decision, matching theory; realized revenue accounts for persistent types',
        common_prices='common across novelty types conditional on known alpha, habit, history, rivals and menus',
        initial_history='first observed basket; next two consecutive calendar weeks priced',
        population='fitted population novelty crossed with observed pricing environments; not realized household revenue',
        participation='nonpurchase utility zero; competing flavor prices fixed',
        prices='continuous; first-period prices may rise/fall or be negative; terminal prices nonnegative; zero cost',
        restrictions='all novelty types optimized; no middle-group exclusion',
        search_points=args.search_points, development_limit=args.max_environments,
        results=reports)
    (args.output/'information_pricing_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    try:
        from .plot_information_pricing import plot_results
    except ImportError:
        from plot_information_pricing import plot_results
    for flavor in flavors:
        plot_results(args.output/flavor, args.output/flavor)
    summary['status'] = 'complete'
    (args.output/'information_pricing_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Complete. Outputs are aggregate-only; demand fit unchanged.', flush=True)
    return summary


def main():
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flavor', default='plain')
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--flavors', type=Path)
    parser.add_argument('--nodes', type=int, default=9)
    parser.add_argument('--search-points', type=int, default=65)
    parser.add_argument('--delta', type=float, default=1.)
    parser.add_argument('--max-environments', type=int)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
