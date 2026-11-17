"""Plot expected paths and welfare from quantified-theory aggregates only."""
from pathlib import Path
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'supply-bins-mpl'))


def plot_bins(results, output, individual=True):
    """Condition on each bin and integrate each policy's own history probabilities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    metrics = ['p1', 'expected_p2', 'consumer_surplus', 'revenue', 'total_welfare']
    required = {'scenario', 'h0', 'regime', 'group', 'weight', *metrics}
    if not required <= set(results):
        raise ValueError('Expected quantified-theory aggregate outcome columns.')
    if {'household_code', 'household_id', 'week', 'upc'} & set(results):
        raise ValueError('Only theory aggregates are supported.')
    base = results[results.scenario.isin(['Fitted distribution', 'Pure theory'])].copy()
    if base.scenario.nunique() != 1:
        raise ValueError('Plot one population distribution at a time.')
    if base.empty or not base.group.str.fullmatch(r'bin_\d+').all():
        raise ValueError('Expected fitted-distribution percentile bins.')
    if not np.isfinite(base[metrics+['weight']]).all().all() or (base.weight <= 0).any():
        raise ValueError('Need finite outcomes and positive integration weights.')
    base[metrics] = base[metrics].mul(base.weight, axis=0)
    grouped = base.groupby(['h0', 'regime', 'group'])[metrics+['weight']].sum()
    grouped[metrics] = grouped[metrics].div(grouped.weight, axis=0)
    bins = sorted(base.group.unique(), key=lambda group: int(group.split('_')[1]))
    output.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output/'expected_paths_and_welfare_by_bin.csv')

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(output/(name+'.'+ext), dpi=180, bbox_inches='tight')
        plt.close(fig)

    colors = plt.get_cmap('viridis')(np.linspace(.1, .9, len(bins)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout='constrained')
    for ax, h0 in zip(axes, ['j', '0']):
        for group, color in zip(bins, colors):
            k = bins.index(group)
            label = f'{100*k/len(bins):g}–{100*(k+1)/len(bins):g}%'
            for regime, style, marker in [('informed', '-', 'o')]:
                row = grouped.loc[(h0, regime, group)]
                ax.plot([1, 2], row[['p1', 'expected_p2']], color=color,
                        linestyle=style, marker=marker,
                        label=label+' / '+('Bin-informed' if regime == 'informed' else 'Common'))
        population = base[(base.h0 == h0)&(base.regime == 'common')]
        path = population[['p1', 'expected_p2']].sum()/population.weight.sum()
        ax.plot([1, 2], path, color='#222222', linestyle='--', marker='D',
                linewidth=2, label='Uniform across population')
        ax.set(title='Initially '+('inside' if h0 == 'j' else 'outside'),
               xticks=[1, 2], xticklabels=['Period 1', 'Expected period 2'], ylabel='Price ($)')
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=7, ncol=2)
    save(fig, 'expected_price_paths_by_bin')
    uniform_paths = []
    for h0 in ['j', '0']:
        population = base[(base.h0 == h0)&(base.regime == 'common')]
        path = population[['p1', 'expected_p2']].sum()/population.weight.sum()
        uniform_paths.append(dict(h0=h0, **path.to_dict()))
    pd.DataFrame(uniform_paths).to_csv(output/'uniform_population_price_path.csv', index=False)

    welfare = [('consumer_surplus', 'CS'), ('revenue', 'Revenue'), ('total_welfare', 'TS')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for row_axes, h0 in zip(axes, ['j', '0']):
        for ax, (metric, label) in zip(row_axes, welfare):
            positions = np.arange(len(bins))
            for offset, regime, name, color in [
                    (-.19, 'common', 'Uniform across population', '#777777'),
                    (.19, 'informed', 'Quintile known', '#2E6896')]:
                values = [grouped.loc[(h0, regime, group), metric] for group in bins]
                ax.bar(positions+offset, values, width=.36, label=name, color=color)
            ax.set(title=label+' / initially '+('inside' if h0 == 'j' else 'outside'),
                   xticks=positions, xticklabels=[f'Q{k+1}' for k in positions],
                   xlabel='Gamma quintile (20% each)', ylabel='Two-period value ($)')
            ax.axhline(0, color='black', linewidth=.6)
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=7)
    save(fig, 'welfare_all_bins')
    if not individual:
        return grouped
    for k, group in enumerate(bins):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
        fig.suptitle(f'Gamma percentile bin: {100*k/len(bins):g}–{100*(k+1)/len(bins):g}%')
        for ax, h0 in zip(axes, ['j', '0']):
            for regime, label, color, style in [
                    ('informed', 'Bin known', '#2E6896', '-'),
                    ('common', 'Uniform across population', '#B35C21', '--')]:
                row = grouped.loc[(h0, regime, group)]
                if regime == 'common':
                    population = base[(base.h0 == h0)&(base.regime == 'common')]
                    row = population[metrics].sum()/population.weight.sum()
                ax.plot([1, 2], row[['p1', 'expected_p2']], marker='o',
                        color=color, linestyle=style, linewidth=2, label=label)
            ax.set(title='Initially '+('inside' if h0 == 'j' else 'outside'),
                   xticks=[1, 2], xticklabels=[r'$p_1$', r'$E[p_2]$'], ylabel='Price ($)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=.2)
        save(fig, 'expected_price_paths_'+group)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout='constrained')
        fig.suptitle(f'Gamma percentile bin: {100*k/len(bins):g}–{100*(k+1)/len(bins):g}%')
        for row_axes, h0 in zip(axes, ['j', '0']):
            for ax, (metric, label) in zip(row_axes, welfare):
                values = [grouped.loc[(h0, regime, group), metric] for regime in ['common', 'informed']]
                ax.bar(['Common', 'Bin-informed'], values, color=['#777777', '#2E6896'])
                ax.set(title=label+' / initially '+('inside' if h0 == 'j' else 'outside'),
                       ylabel='Two-period value ($)')
                ax.axhline(0, color='black', linewidth=.6)
                ax.grid(axis='y', alpha=.2)
        save(fig, 'welfare_'+group)
    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Quantified-theory output root with flavor subdirectories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for flavor in ['plain', 'berry', 'other']:
        # This filename contains theoretical quadrature nodes, never household records.
        plot_bins(pd.read_csv(args.source/flavor/'type_results.csv'), args.output/flavor)
