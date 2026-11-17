"""Alternative price displays from saved theoretical integration outcomes only."""
from pathlib import Path
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'supply-bins-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def figures(folder, title, include_expected=True):
    data = pd.read_csv(folder/'type_results.csv')
    metrics = ['p1', 'p2j', 'p20', 'expected_p2', 'quantity1']
    if {'household_code', 'household_id', 'week', 'upc'} & set(data):
        raise ValueError('Only theoretical integration outcomes are supported.')
    weighted = data.copy()
    weighted[metrics] = weighted[metrics].mul(weighted.weight, axis=0)
    grouped = weighted.groupby(['h0', 'regime', 'group'])[metrics+['weight']].sum()
    grouped[metrics] = grouped[metrics].div(grouped.weight, axis=0)
    bins = sorted(data.group.unique(), key=lambda name: int(name.split('_')[1]))
    x = np.arange(len(bins))

    def panel(ax, metric, h0, label):
        for regime, name, color, marker, offset in [
                ('common', 'Uniform', '#444444', 'D', -.10),
                ('informed', 'Informed', '#2E6896', 'o', .10)]:
            values = [grouped.loc[(h0, regime, group), metric] for group in bins]
            ax.scatter(x+offset, values, color=color, marker=marker, s=45, label=name, zorder=3)
        ax.set(title=label, xticks=x, xticklabels=[f'Q{k+1}' for k in x],
               xlabel='Gamma quintile (20% each)', ylabel='Offered price ($)')
        ax.grid(axis='y', alpha=.2)
        lo, hi = ax.get_ylim()
        if lo < 0 < hi:
            ax.axhline(0, color='black', linewidth=.6)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, layout='constrained')
    fig.suptitle(title+' — Prices by decision', fontsize=15)
    panel(axes[0, 0], 'p1', 'j', 'Period 1: initially inside')
    panel(axes[0, 1], 'p1', '0', 'Period 1: initially outside')
    for metric in ['p2j', 'p20']:
        np.testing.assert_allclose(grouped.xs('j', level='h0')[metric],
                                   grouped.xs('0', level='h0')[metric])
    panel(axes[1, 0], 'p2j', 'j', 'Period 2: purchased in period 1')
    panel(axes[1, 1], 'p20', 'j', 'Period 2: did not purchase in period 1')
    fig.supxlabel('Model-optimized prices; no averaging over purchase histories. '
                  'Terminal policies use fixed beliefs.', fontsize=10)
    yield fig, 'prices_by_decision'
    if not include_expected:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout='constrained')
    fig.suptitle(title+' — Expected second-period offers', fontsize=15)
    for ax, h0 in zip(axes, ['j', '0']):
        panel(ax, 'expected_p2', h0, 'Initially '+('inside' if h0 == 'j' else 'outside'))
        ax.set_ylabel('Expected offered price ($)')
    fig.supxlabel('Both policies evaluated within the same quintile; '
                  'each uses its own purchase probabilities.', fontsize=10)
    yield fig, 'expected_prices_same_quintile'


def build(source):
    source = Path(source)
    index = pd.read_csv(source/'exercise_index.csv')
    with PdfPages(source/'pricing_display_alternatives.pdf') as packet:
        for row in index.itertuples():
            folder = source/row.directory
            for fig, stem in figures(folder, row.exercise):
                for ext in ['pdf', 'png']:
                    fig.savefig(folder/(stem+'.'+ext), dpi=180, bbox_inches='tight')
                packet.savefig(fig)
                plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    build(parser.parse_args().source)
