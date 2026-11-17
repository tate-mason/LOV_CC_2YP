"""Build a standalone comparison packet from theory and aggregate fit settings."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from quintile_theory import run
from plot_pricing_comparisons import figures as pricing_figures
from plot_supply_bins import plot_bins


def build(output, estimates=None, initial_states=None):
    root = Path(__file__).resolve().parents[3]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exercises = []
    state_counts = None
    if initial_states is not None:
        state_counts = pd.read_csv(initial_states, dtype={'h0': str})
        if set(state_counts) != {'flavor', 'h0', 'households'}:
            raise ValueError('Expected aggregate flavor/h0/households counts only.')
        if (state_counts.duplicated(['flavor', 'h0']).any()
                or set(state_counts.flavor) != {'plain', 'berry', 'other'}
                or set(state_counts.h0) != {'j', '0'} or len(state_counts) != 6
                or not np.isfinite(state_counts.households).all()
                or (state_counts.households < 0).any()
                or (state_counts.households % 1 != 0).any()
                or (state_counts.groupby('flavor').households.sum() <= 0).any()):
            raise ValueError('Need two nonnegative integer state counts for each flavor.')
        if state_counts.groupby('flavor').households.sum().nunique() != 1:
            raise ValueError('All flavors must initialize the same household sample.')
        state_counts.to_csv(output/'observed_initial_state_counts.csv', index=False)
    theory = output/'01_pure_theory'
    run(theory, individual=False)
    exercises.append(('Pure theory: normal gamma, mean 0, SD 1.25', theory))
    if estimates is not None:
        estimates = Path(estimates).resolve()
        table = pd.read_csv(estimates)
        if not {'parameter', 'estimate'} <= set(table) or set(table)-{'parameter', 'estimate', 'at_bound'}:
            raise ValueError('Expected population parameter estimates only.')
        if table.parameter.duplicated().any():
            raise ValueError('Duplicate parameter estimates.')
        if 'at_bound' in table and table.at_bound.astype(str).str.lower().eq('true').any():
            raise ValueError('Review boundary estimates before producing results.')
        fit = json.loads(estimates.with_name('fit.json').read_text())
        if fit.get('spec') != 'novelty_only' or fit.get('habit') is not False or not fit.get('converged'):
            raise ValueError('Expected a converged no-habit novelty_only fit.')
        p = table.set_index('parameter').estimate.to_dict()
        if not np.isfinite(list(p.values())).all():
            raise ValueError('Nonfinite parameter estimate.')
        alpha = -float(np.exp(p['mu_log_minus_alpha']+.5*p['sd_log_minus_alpha']**2))
        for flavor in ['plain', 'berry', 'other']:
            folder = output/f'{len(exercises)+1:02d}_novelty_only_{flavor}'
            run(folder, mean=p['mean_novelty'], sd=p['sd_novelty'],
                beta=p['inside_constant']+p.get('beta_'+flavor, 0.), alpha=alpha,
                delta=1., exercise=f'No-habit estimated inputs, {flavor}', individual=False)
            provenance = dict(source=str(estimates),
                source_sha256=hashlib.sha256(estimates.read_bytes()).hexdigest(),
                spec=fit['spec'], habit=fit['habit'], converged=fit['converged'],
                parameter_estimates=p, alpha_mapping='-exp(mu_log_minus_alpha + sd_log_minus_alpha^2/2)',
                beta_mapping='inside_constant + beta_flavor; other is reference',
                demand_refitted=False, pricing='existing fixed-belief binary quintile model',
                evaluation='persistent-type history integration; matches introductory objective',
                learning=False,
                initialization='household first observed yogurt basket' if state_counts is not None
                    else 'population-input benchmark; no observed household initialization')
            (folder/'source_settings.json').write_text(json.dumps(provenance, indent=2)+'\n')
            exercises.append((f'No-habit estimates / {flavor}', folder))
    sources = [] if estimates is not None else [
        ('earlier_fit', root/'Code/final_simple/results/quantified_theory_five_bins_integrated'),
        ('observed_week_128', root/'Code/final_draft/seminar_pres/assets/quantified_theory_observed_128')]
    for fit, source in sources:
        for flavor in ['plain', 'berry', 'other']:
            folder = output/f'{len(exercises)+1:02d}_{fit}_{flavor}'
            settings = json.loads((source/flavor/'settings.json').read_text())
            if fit == 'earlier_fit':
                folder.mkdir(parents=True, exist_ok=True)
                results = pd.read_csv(source/flavor/'type_results.csv')
                results.to_csv(folder/'type_results.csv', index=False)
                plot_bins(results, folder, individual=False)
                shutil.copyfile(source/flavor/'settings.json', folder/'settings.json')
            else:
                results = run(folder, mean=settings['gamma_mean'], sd=settings['gamma_sd'],
                    beta=settings['beta'], alpha=settings['alpha'], delta=settings['delta'],
                    exercise=f'Applied {fit}, {flavor}; aggregate fitted inputs', individual=False)
            (folder/'source_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
            exercises.append((f'Applied / {fit.replace("_", " ")} / {flavor}', folder))

    # Verify accounting and exactly one common offer per history in every exercise.
    ledger = []
    population = []
    for title, folder in exercises:
        data = pd.read_csv(folder/'type_results.csv')
        np.testing.assert_allclose(data.groupby(['h0','regime','group']).weight.sum(), .2)
        np.testing.assert_allclose(data.total_welfare, data.consumer_surplus+data.revenue)
        np.testing.assert_allclose(data.expected_p2,
            data.quantity1*data.p2j+(1-data.quantity1)*data.p20)
        for _, common in data[data.regime == 'common'].groupby('h0'):
            assert all(common[key].nunique() == 1 for key in ['p1','p2j','p20'])
        for (h0, regime), part in data.groupby(['h0', 'regime']):
            metrics = ['p1', 'expected_p2', 'consumer_surplus', 'revenue', 'total_welfare']
            population.append(dict(exercise=title, h0=h0, regime=regime,
                **{key:float(np.average(part[key], weights=part.weight)) for key in metrics}))
        ledger.append(dict(exercise=title, directory=folder.name, quintiles=5,
                           population_mass_per_quintile=.2, checks='passed'))
    pd.DataFrame(ledger).to_csv(output/'exercise_index.csv', index=False)
    summary = pd.DataFrame(population)
    summary.to_csv(output/'population_comparison.csv', index=False)
    if state_counts is not None:
        # Conditional policies use the same population gamma prior for both h0.
        # Actual observed initial-state counts supply the population weights.
        overall = []
        quintiles = []
        for title, folder in exercises[1:]:
            flavor = folder.name.rsplit('_', 1)[1]
            counts = state_counts[state_counts.flavor == flavor].set_index('h0').households
            n = int(counts.sum())
            weights = counts/n
            conditional = summary[summary.exercise == title]
            for regime, part in conditional.groupby('regime'):
                overall.append(dict(flavor=flavor, regime=regime, households=n,
                    **{key:float((part.set_index('h0')[key]*weights).sum())
                       for key in ['p1', 'expected_p2', 'consumer_surplus', 'revenue', 'total_welfare']}))
            data = pd.read_csv(folder/'type_results.csv')
            data['population_weight'] = data.weight*data.h0.map(weights)
            for (regime, group), part in data.groupby(['regime', 'group']):
                quintiles.append(dict(flavor=flavor, regime=regime, group=group,
                    population_mass=float(part.population_weight.sum()),
                    revenue_contribution=float((part.population_weight*part.revenue).sum())))
        overall = pd.DataFrame(overall)
        overall.to_csv(output/'observed_initialization_comparison.csv', index=False)
        measures = ['consumer_surplus', 'revenue', 'total_welfare']
        uniform_overall = overall[overall.regime == 'common'].set_index('flavor')[measures]
        informed_overall = overall[overall.regime == 'informed'].set_index('flavor')[measures]
        if (uniform_overall == 0).any().any():
            raise ValueError('Undefined observed-initialization percentage change: zero baseline.')
        (100*(informed_overall-uniform_overall)/uniform_overall.abs()).add_suffix('_pct_change').to_csv(
            output/'observed_initialization_percentage_changes.csv')
        quintiles = pd.DataFrame(quintiles)
        totals = quintiles.groupby(['flavor', 'regime']).revenue_contribution.transform('sum')
        if (totals <= 0).any():
            raise ValueError('Revenue shares require positive total revenue.')
        quintiles['revenue_share_pct'] = 100*quintiles.revenue_contribution/totals
        quintiles.to_csv(output/'observed_initialization_revenue_contributions.csv', index=False)
    keys = ['exercise', 'h0']
    metrics = ['consumer_surplus', 'revenue', 'total_welfare']
    uniform = summary[summary.regime == 'common'].set_index(keys)[metrics]
    informed = summary[summary.regime == 'informed'].set_index(keys)[metrics]
    if (uniform == 0).any().any():
        raise ValueError('Undefined percentage changes: zero population baseline.')
    (100*(informed-uniform)/uniform.abs()).add_suffix('_pct_change').to_csv(
        output/'population_percentage_changes.csv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    applied_description = (
        'Applied exercises use the saved aggregate estimates. The corrected observed-week '
        'fit is recomputed with quintiles to match the earlier fit; no demand model is refitted.')
    if estimates is not None:
        applied_description = (
            'Applied exercises use the newly estimated novelty_only model without habit. '
            'All flavor intercepts, mean price sensitivity, and gamma distribution parameters '
            'come from that fit. No demand model is refitted here.')
    lines = ['# Supply exercise review', '',
        'Open **supply_review.pdf** for the complete packet. Each exercise has two pages: '
        'four panels of optimized prices, then CS/revenue/TS across all quintiles.', '',
        'Every quintile contains 20% of its population. The four price panels show period 1 '
        'for initially-inside and initially-outside consumers, then period 2 after purchase '
        'and nonpurchase. All prices are chosen by the model; none are observed data prices '
        'or averages across purchase histories. Uniform knows the quintile mixture but cannot '
        'identify membership; Informed knows membership with uncertainty within the quintile.', '',
        'Welfare and revenue bars show 100 × (quintile-informed − uniform) / |uniform|, '
        'using per-consumer conditional means for the same quintile and initial history. '
        'Uniform pricing is the 0% baseline; the absolute denominator preserves the '
        'direction of improvements when baseline welfare is negative. '
        'TS = CS + revenue at zero cost. Gamma persists across periods. '
        'Beliefs remain fixed within the observed bin or full population.', '',
        'Pure theory assumes normal gamma (mean 0, SD 1.25), beta 2, alpha -1, delta 1. '
        + applied_description, '',
        'The old three-point-prior and quadrature display versions are superseded here by '
        'the matching quintile versions. Each folder includes PNG/PDF figures, numerical '
        'outcomes, uniform population paths, settings and source provenance.', '']
    if estimates is not None:
        lines += ['Source: '+str(estimates), '',
            'These are the binary theory exercises populated with no-habit estimates. '
            'Pricing retains fixed beliefs; reported '
            'outcomes and the introductory objective integrate the same persistent types. '
            'Terminal offers optimize under unchanged beliefs; no learning occurs. '
            'The uniform firm knows all five quintile '
            'distributions and weights but cannot identify individual membership.', '']
        if state_counts is not None:
            lines += ['Initialization uses the first actual observed yogurt basket for each '
                'included household. Focal purchase sets h0=j; no focal purchase sets h0=0. '
                'The next two consecutive purchase weeks are counterfactual pricing periods; '
                'they need not be consecutive calendar weeks. Future '
                'purchases are generated under optimized prices, not copied from observed '
                'future choices. No observed prices are imposed on either period.', '',
                'State-conditional panels show the same policies regardless of initial-state '
                'frequencies: gamma beliefs are explicitly held identical across h0. '
                'The observed_initialization_* tables combine states using actual household '
                'counts. Quintiles remain equal-probability bins of the fitted population '
                'distribution, not household posterior-mean ranks. No learning is introduced.', '']
        else:
            lines += ['INITIALIZATION LIMIT: These are population-input benchmarks. They do not '
                'read household initial occasions. Use server/run_observed_state_theory.sh '
                'to produce the observed-initialization version.', '']
    with PdfPages(output/'supply_review.pdf') as packet:
        for title, folder in exercises:
            lines += [f'## {title}', '']
            for price_fig, stem in pricing_figures(folder, title, include_expected=False):
                for ext in ['pdf', 'png']:
                    price_fig.savefig(folder/(stem+'.'+ext), dpi=180, bbox_inches='tight')
                lines += [f'![Optimized prices by decision]({folder.name}/{stem}.png)', '']
                packet.savefig(price_fig)
                plt.close(price_fig)
            for stem, label in [('welfare_all_bins','CS, revenue and TS (change as % of |uniform baseline|)')]:
                relative = f'{folder.name}/{stem}.png'
                lines += [f'![{label}]({relative})', '']
                fig, ax = plt.subplots(figsize=(13.5, 8.5))
                ax.imshow(plt.imread(folder/(stem+'.png')))
                ax.axis('off')
                fig.suptitle(title+' — '+label, fontsize=15)
                fig.tight_layout()
                packet.savefig(fig)
                plt.close(fig)
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Created {len(exercises)} exercises, two matching figures each: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--estimates', type=Path,
                        help='Population point_estimates.csv from a converged novelty_only fit')
    parser.add_argument('--initial-states', type=Path,
                        help='Aggregate flavor/h0/households counts from the server initializer')
    args = parser.parse_args()
    build(args.output, args.estimates, args.initial_states)
