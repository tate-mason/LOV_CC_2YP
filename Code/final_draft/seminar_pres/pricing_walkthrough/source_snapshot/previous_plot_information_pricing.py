"""Supply-style figures from explicitly named aggregate pricing exports only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'information-pricing-mpl'))


def plot_results(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source, output = Path(source), Path(output)
    names = ['type_node_aggregate.csv', 'price_paths_by_type.csv', 'revenue_comparison.csv']
    nodes, groups, totals = [pd.read_csv(source/name) for name in names]
    for table in [nodes, groups, totals]:
        if any(c in table for c in ['household_code', 'household_id', 'week', 'upc']):
            raise ValueError('Figures accept aggregate result tables only.')
        if not {'scenario', 'initial_state', 'regime', 'weight', 'p1', 'expected_p2', 'revenue'} <= set(table):
            raise ValueError('Missing aggregate pricing columns.')
    output.mkdir(parents=True, exist_ok=True)
    base = 'Fitted distribution'
    baseline = nodes[nodes.scenario == base]
    color = {'inside': '#2E6896', 'outside': '#B35C21'}
    policy_colors = {'full': '#2E6896', 'limited': '#B35C21'}
    policy_names = {'full': 'Household-specific prices', 'limited': 'Common prices across γ'}
    state_labels = {'inside': 'Initially inside', 'outside': 'Initially outside'}
    files = []
    def save(fig, stem):
        for ext in ['pdf', 'png']:
            fig.savefig(output/(stem+'.'+ext), bbox_inches='tight', dpi=180)
            files.append(stem+'.'+ext)
        plt.close(fig)
    # Same layout as theoretical optimal-price plots: p1 by h0, p2 by h1.
    full = baseline[baseline.regime == 'full']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    for initial in ['inside', 'outside']:
        x = full[full.initial_state == initial].sort_values('gamma')
        if not x.empty:
            axes[0].plot(x.gamma, x.p1/x.weight, color=color[initial], lw=2, label=state_labels[initial])
    pooled = full.groupby('gamma', as_index=False).sum(numeric_only=True).sort_values('gamma')
    for branch in ['inside', 'outside']:
        axes[1].plot(pooled.gamma, pooled['p2_'+branch]/pooled['mass_'+branch],
                     color=color[branch], lw=2, label='Previous '+branch)
    for ax, title in zip(axes, ['Optimal period-1 prices', 'Optimal period-2 prices']):
        ax.set(xlabel=r'$\gamma$', ylabel='Mean optimal price ($)', title=title)
        ax.legend(fontsize=8);ax.grid(axis='y', alpha=.2)
    save(fig, 'optimal_prices_by_gamma')
    # Four initial/realized-history combinations, matching the supply panels.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    # Match the supply exercise's history order, colors and line styles.
    for initial, branch, c, style in [
            ('outside', 'outside', '#1f77b4', '-'),
            ('outside', 'inside', '#ff7f0e', '--'),
            ('inside', 'inside', '#2ca02c', '-.'),
            ('inside', 'outside', '#d62728', ':')]:
        x = full[full.initial_state == initial].sort_values('gamma')
        if x.empty:
            continue
        h0, h1 = ('j' if initial == 'inside' else '0'), ('j' if branch == 'inside' else '0')
        label = rf'$(h_0,h_1)=({h0},{h1})$'
        axes[0].plot(x.gamma, x['ratio_'+branch]/x['mass_'+branch], color=c, ls=style, lw=2, label=label)
        axes[1].plot(x.gamma, x['profit_'+branch]/x['mass_'+branch], color=c, ls=style, lw=2, label=label)
    for ax, title, ylabel in zip(axes, ['Price ratios by realized history', 'Profit conditional on first purchase outcome'],
                                 [r'Mean $p_1/p_2$', 'Expected two-period revenue ($)']):
        ax.set(xlabel=r'$\gamma$', ylabel=ylabel, title=title)
        ax.legend(fontsize=8);ax.grid(axis='y', alpha=.2)
    save(fig, 'price_ratios_and_profit')
    # Type-specific versus common paths in the two initial-history panels.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
    palette = {'low': '#2E6896', 'middle': '#208475', 'high': '#B35C21'}
    selected = groups[groups.scenario == base]
    for ax, initial in zip(axes, ['inside', 'outside']):
        x = selected[(selected.initial_state == initial) & (selected.regime == 'full')]
        x = x.assign(_order=x.group.map({'low': 0, 'middle': 1, 'high': 2})).sort_values('_order')
        for row in x.itertuples():
            ax.plot([1, 2], [row.p1/row.weight, row.expected_p2/row.weight], marker='o',
                    lw=2, color=palette[row.group], label=row.group.title()+' type')
        common = totals[(totals.scenario == base) & (totals.initial_state == initial) & (totals.regime == 'limited')]
        if len(common):
            r = common.iloc[0]
            ax.plot([1, 2], [r.p1/r.weight, r.expected_p2/r.weight], marker='D', linestyle='--',
                    color='#232C35', lw=2, label='Common prices across γ')
        ax.set(title=state_labels[initial], xticks=[1, 2], xticklabels=[r'$p_1$', r'$\mathbb{E}[p_2]$'], ylabel='Mean price ($)')
        ax.legend(fontsize=8);ax.grid(axis='y', alpha=.2)
    save(fig, 'information_price_paths_comparison')
    scenarios = list(dict.fromkeys(totals.scenario))
    for metric in ['revenue', 'gain']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout='constrained')
        for ax, initial in zip(axes, ['inside', 'outside']):
            x = totals[totals.initial_state == initial]
            if x.empty:
                ax.set_title(state_labels[initial]+' (no eligible environments)')
                continue
            by_case = {regime: x[x.regime == regime].set_index('scenario').reindex(scenarios)
                       for regime in ['full', 'limited']}
            if metric == 'revenue':
                for regime, offset in [('limited', -.18), ('full', .18)]:
                    frame = by_case[regime]
                    ax.bar(np.arange(len(scenarios))+offset, frame.revenue/frame.weight,
                           width=.36, color=policy_colors[regime], label=policy_names[regime])
            else:
                mean_full = by_case['full'].revenue/by_case['full'].weight
                mean_limited = by_case['limited'].revenue/by_case['limited'].weight
                ax.bar(np.arange(len(scenarios)), mean_full-mean_limited,
                       color=policy_colors['full'], label='Household-specific − common')
            ax.set(title=state_labels[initial], xticks=np.arange(len(scenarios)),
                   xticklabels=[s.replace(' distribution', '\ndistribution').replace(' type more likely', '\nmore likely') for s in scenarios])
            ax.legend(fontsize=7);ax.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Expected two-period revenue ($)' if metric == 'revenue' else 'Revenue gain per model consumer ($)')
        save(fig, 'information_revenues' if metric == 'revenue' else 'information_value')
    # The price question: how far do offers move from observed posted prices?
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
    for ax, initial in zip(axes, ['inside', 'outside']):
        for regime in ['full', 'limited']:
            x = totals[(totals.scenario == base) & (totals.initial_state == initial) & (totals.regime == regime)]
            if x.empty:
                continue
            r = x.iloc[0]
            ax.plot([1, 2], [r.change_p1/r.weight, r.change_p2/r.weight], marker='o', lw=2,
                    color=policy_colors[regime], label=policy_names[regime])
        ax.axhline(0., color='gray', lw=.8)
        ax.set(title=state_labels[initial], xticks=[1, 2], xticklabels=[r'$p_1$', r'$\mathbb{E}[p_2]$'],
               ylabel='Mean change from observed price ($)')
        ax.legend(fontsize=8);ax.grid(axis='y', alpha=.2)
    save(fig, 'price_changes')
    (output/'figure_provenance.json').write_text(json.dumps(dict(
        source_files={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
        figures=files, baseline=base,
        meaning='Means across observed environments; gamma curves connect numerical integration nodes',
        terminal='Conditional on at least one focal purchase versus no focal purchase; averages over baskets',
        ratio='Conditional mean of realized p1/p2, not ratio of population-average prices',
        profit='Expected revenue conditional on purchase branch; zero marginal cost',
        common='Common across novelty types within a known alpha/habit/history environment'), indent=2)+'\n')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot_results(args.source, args.output or args.source)


if __name__ == '__main__':
    main()
