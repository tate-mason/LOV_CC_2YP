"""Import only aggregate revised-pricing results and render supply-style slide assets."""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'Code/final_simple/demand/plot_information_pricing.py').is_file())
sys.path.insert(0, str(ROOT/'Code/final_simple'))
from demand.plot_information_pricing import plot_results
HERE = Path(__file__).resolve().parent


def refresh(source, flavor='plain'):
    source = Path(source)
    summary = json.loads((source/'information_pricing_summary.json').read_text())
    if (summary.get('status') != 'complete' or summary.get('demand_refitted') is not False
            or summary.get('hidden') != 'novelty preference only'
            or summary.get('specification') != 'binary_focal_vs_not'
            or summary.get('capacity') != 1
            or summary.get('development_limit') is not None):
        raise ValueError('Need completed binary plain-versus-not results; earlier full-basket outputs cannot populate binary slides.')
    if flavor not in summary['results']:
        raise ValueError('Focal flavor absent from completed run.')
    output = HERE/'assets/binary_information_pricing'/flavor
    plot_results(source/flavor, output)
    totals = pd.read_csv(source/flavor/'revenue_comparison.csv')
    rows = [r'\begin{tabular}{llrrr}\toprule',
            r'Initial state & Prior & Household-specific & Common & Gain\\\midrule']
    baseline_gains = {}
    for scenario in dict.fromkeys(totals.scenario):
        for state in ['inside', 'outside']:
            selected = totals[(totals.scenario == scenario) & (totals.initial_state == state)].set_index('regime')
            if selected.empty:
                continue
            values = selected.revenue/selected.weight
            gain = values['full']-values['limited']
            rows.append(f'{state.title()} & {scenario} & {values["full"]:.3f} & {values["limited"]:.3f} & {gain:.3f} '+r'\\')
            if scenario == 'Fitted distribution':
                baseline_gains[state] = dict(gain=float(gain), pct=float(100*gain/values['limited']))
    rows.append(r'\bottomrule\end{tabular}')
    (output/'revenue_table.tex').write_text('\n'.join(rows)+'\n')
    text = '; '.join(f'Initially {state}: {values["pct"]:.2f}'+r'\%'
                     +f' (\\${values["gain"]:.3f} per consumer)'
                     for state, values in baseline_gains.items())
    (output/'gain_headline.tex').write_text(text+'.\n')
    result = summary['results'][flavor]
    inside, outside = (result['initial_states'][key] for key in ['inside', 'outside'])
    (output/'run_context.tex').write_text(
        f'{result["environments"]:,} observed price/history environments '
        +f'({inside:,} initially inside, {outside:,} initially outside).\n')
    allowed = ['status', 'model', 'specification', 'parameter_mapping', 'outside_index', 'habit_transition', 'estimation_capacity', 'demand_refitted', 'fit_sha256', 'parameters_sha256',
               'solver_sha256', 'stock_measure', 'stock_retention', 'novelty_measure',
               'capacity', 'delta', 'hidden', 'known', 'beliefs', 'common_prices',
               'evaluation', 'quadrature_nodes', 'search_points', 'results']
    (output/'run_provenance.json').write_text(json.dumps({k: summary[k] for k in allowed}, indent=2)+'\n')
    print('Revised pricing slide assets ready:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Recovered aggregate-only RESULTS_ONLY folder.')
    parser.add_argument('--flavor', default='plain', choices=['plain', 'berry', 'other'])
    args = parser.parse_args()
    refresh(args.source, args.flavor)
