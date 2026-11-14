"""Aggregate-only welfare tables and figures; no household or research-record I/O."""
import json
import os
import tempfile
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pricing_presentation import prior_labels

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lov-welfare-mpl'))

METRICS = ['revenue', 'profit', 'consumer_surplus', 'total_welfare',
           'quantity1', 'quantity2', 'quantity', 'undiscounted_quantity',
           'zero_outside_consumer_surplus', 'zero_outside_total_welfare']


def write_welfare_report(types, summary, output):
    """Types are theoretical support points, never individual consumer records."""
    output = Path(output)
    keys = ['scenario', 'h0']
    common = summary[summary.regime == 'common'].set_index(keys)
    informed = summary[summary.regime == 'informed'].set_index(keys)
    gains = informed[METRICS] - common[METRICS]
    for metric in ['consumer_surplus', 'total_welfare']:
        c = types[types.regime == 'common'].set_index(keys+['gamma'])
        i = types[types.regime == 'informed'].set_index(keys+['gamma'])
        change = i[metric]-c[metric]
        fraction = (i.weight*(change > 1e-9)).groupby(level=keys).sum()
        gains[metric+'_winner_share'] = fraction
    gains.reset_index().to_csv(output/'welfare_changes.csv', index=False)
    summary.to_csv(output/'welfare_comparison.csv', index=False)
    group = types.assign(**{key:types[key]*types.weight for key in METRICS})
    group = group.groupby(keys+['regime', 'group'])[METRICS+['weight']].sum()
    group[METRICS] = group[METRICS].div(group.weight, axis=0)
    group.reset_index().to_csv(output/'welfare_by_group.csv', index=False)

    plot_welfare_changes(summary, output)
    scenarios = list(summary.scenario.unique())
    lines = [r'\begin{tabular}{llrrrr}\toprule',
             r'Initial state & Distribution & $\Delta\Pi$ & $\Delta CS$ & $\Delta W$ & $\Delta Q$\\\midrule']
    for scenario in scenarios:
        for h0 in ['j','0']:
            r = gains.loc[(scenario,h0)]
            label = prior_labels([scenario])[0]
            lines.append(f'${h0}$ & {label} & {r.profit:+.4f} & {r.consumer_surplus:+.4f}'
                         f' & {r.total_welfare:+.4f} & {r.quantity:+.4f} '+r'\\')
    (output/'welfare_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
    baseline = gains.xs(scenarios[0], level='scenario')
    def direction(x):
        return 'rises' if x > 1e-9 else ('falls' if x < -1e-9 else 'is unchanged')
    for h0, stem in [('j','inside'),('0','outside')]:
        r = baseline.loc[h0]
        (output/f'welfare_headline_{stem}.tex').write_text(
            f'Consumer surplus {direction(r.consumer_surplus)} '
            f'(${r.consumer_surplus:+.3f}$); total welfare {direction(r.total_welfare)} '
            f'(${r.total_welfare:+.3f}$).\n')
    return gains


def plot_welfare_changes(summary, output):
    """Plot percentage changes relative to absolute uniform-pricing baselines."""
    output = Path(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['scenario', 'h0']
    common = summary[summary.regime == 'common'].set_index(keys)
    informed = summary[summary.regime == 'informed'].set_index(keys)
    metrics = ['profit', 'consumer_surplus', 'total_welfare']
    baseline = common[metrics]
    if (baseline.abs() <= 1e-12).any().any():
        raise ValueError('Percentage welfare changes require nonzero baselines')
    changes = 100 * (informed[metrics] - baseline) / baseline.abs()
    scenarios = list(summary.scenario.unique())
    fig, axes = plt.subplots(1,2,figsize=(10,4.1),sharey=True,layout='constrained')
    for ax, h0 in zip(axes, ['j','0']):
        data = changes.xs(h0, level='h0').reindex(scenarios)
        x = np.arange(len(data))
        for shift, key, label, color in [(-.25,'profit','Firm profit','#2E6896'),
                (0,'consumer_surplus','Consumer surplus','#B35C21'),
                (.25,'total_welfare','Total welfare','#208475')]:
            ax.bar(x+shift, data[key], width=.24, label=label, color=color)
        ax.axhline(0., color='black', linewidth=.7)
        ax.set(title='Initially '+('inside' if h0=='j' else 'outside'), xticks=x,
               xticklabels=prior_labels(scenarios),
               ylabel='Change from uniform pricing (%)')
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    from matplotlib.ticker import PercentFormatter
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].legend(fontsize=8)
    for ext in ['pdf','png']:
        fig.savefig(output/('welfare_changes.'+ext), bbox_inches='tight', dpi=170)
    plt.close(fig)


def plot_discrimination_percent(summary, output):
    """Estimated-distribution revenue and welfare gains, by initial history."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from matplotlib.patches import Patch
    data = summary[summary.scenario == 'Fitted distribution']
    common = data[data.regime == 'common'].set_index('h0')
    informed = data[data.regime == 'informed'].set_index('h0')
    metrics = ['revenue', 'profit', 'consumer_surplus', 'total_welfare']
    baseline = common[metrics].reindex(['j', '0'])
    if (baseline.abs() <= 1e-12).any().any():
        raise ValueError('Percentage changes require nonzero baselines')
    changes = 100 * (informed[metrics].reindex(['j', '0']) - baseline) / baseline.abs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    inside = ['#356BA0', '#8064A2', '#339582']
    outside = ['#C25559', '#D68C42', '#CB7DA0']
    axes[0].bar([0, 1], changes.revenue, color=[inside[0], outside[0]], width=.5)
    measures = [('profit', 'Profit'), ('consumer_surplus', 'Consumer surplus'),
                ('total_welfare', 'Total welfare')]
    for k, (key, label) in enumerate(measures):
        axes[1].bar(np.array([0, 1])+(k-1)*.24, changes[key], width=.23,
                    color=[inside[k], outside[k]], edgecolor='white', linewidth=.7)
    axes[1].legend(handles=[Patch(facecolor=palette[k], label=history+' · '+label)
        for history, palette in [('Inside', inside), ('Outside', outside)]
        for k, (_, label) in enumerate(measures)],
        loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=8, ncol=2,
        columnspacing=1, handlelength=1.3, frameon=False)
    for ax, title in zip(axes, ['Revenue', 'Welfare']):
        ax.set(title=title, xticks=[0, 1], xticklabels=['Initially inside', 'Initially outside'],
               xlim=(-.55, 1.55), ylabel='Change from uniform pricing (%)')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.axhline(0, color='black', linewidth=.7)
        ax.grid(axis='y', alpha=.2)
    for ext in ['pdf', 'png']:
        fig.savefig(Path(output)/('discrimination_percent.'+ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def theory_report(output):
    """Recover welfare directly from saved pure-theory policies; no reoptimization."""
    from binary_welfare import evaluate_policy
    output = Path(output)
    settings = json.loads((output/'settings.json').read_text())
    solutions = json.loads((output/'solutions.json').read_text())
    rows = []
    for solution in solutions:
        for k, known in enumerate(solution['known']['types']):
            for regime in ['informed','common']:
                policy = known if regime == 'informed' else solution['unknown']
                values = evaluate_policy(known['gamma'], solution['h0'], settings['alpha'],
                    settings['beta'], settings['delta'],
                    known['p1'] if regime == 'informed' else policy['expected_p1'],
                    policy['p2j'], policy['p20'], settings['c'])
                rows.append(dict(scenario=solution['scenario'], h0=solution['h0'],
                    gamma=known['gamma'], group=['low','middle','high'][k],
                    weight=solution['prior'][k], regime=regime,
                    **{key:float(value) for key,value in values.items()}))
    types = pd.DataFrame(rows)
    types.to_csv(output/'welfare_type_results.csv', index=False)
    metrics = list(values)
    summary = types.assign(**{key:types[key]*types.weight for key in metrics})
    summary = summary.groupby(['scenario','h0','regime'], sort=False)[metrics].sum().reset_index()
    for solution in solutions:
        for regime, case in [('informed','known'),('common','unknown')]:
            row = summary[(summary.scenario==solution['scenario'])
                          & (summary.h0==solution['h0']) & (summary.regime==regime)].iloc[0]
            np.testing.assert_allclose(row.profit, solution[case]['persistent_type_revenue'], atol=1e-8)
    write_welfare_report(types, summary, output)
    (output/'welfare_settings.json').write_text(json.dumps(dict(
        utility_inside='beta + alpha*p + gamma*1{h=0}',
        utility_outside='gamma*1{h=j}',
        welfare='CS + profit; common shock constant omitted; same delta for both',
        evaluation='myopic choices; persistent types; policy-specific history probabilities',
        normalization_sensitivity='zero_outside_* removes gamma*1{h=j} from both utilities',
        source='solutions.json; settings.json; pure theory only'), indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theory-output', type=Path, required=True)
    theory_report(parser.parse_args().theory_output)
