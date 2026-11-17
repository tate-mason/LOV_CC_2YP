"""Quintile shares of total revenue, from aggregate theory outcomes only."""
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
from matplotlib.ticker import PercentFormatter


def build(source):
    source = Path(source)
    rows = []
    with PdfPages(source/'revenue_contributions_by_quintile.pdf') as packet:
        for exercise in pd.read_csv(source/'exercise_index.csv').itertuples():
            folder = source/exercise.directory
            data = pd.read_csv(folder/'type_results.csv')
            if {'household_code', 'household_id', 'week', 'upc'} & set(data):
                raise ValueError('Only theoretical integration outcomes are supported.')
            data['weighted_revenue'] = data.weight*data.revenue
            contribution = data.groupby(['h0', 'regime', 'group']).weighted_revenue.sum()
            totals = contribution.groupby(level=['h0', 'regime']).transform('sum')
            if (totals <= 0).any():
                raise ValueError('Revenue shares require positive total revenue.')
            shares = 100*contribution/totals
            np.testing.assert_allclose(shares.groupby(level=['h0', 'regime']).sum(), 100.)
            table = pd.DataFrame(dict(revenue_contribution=contribution,
                                     total_revenue=totals, contribution_pct=shares)).reset_index()
            table.insert(0, 'exercise', exercise.exercise)
            rows.append(table)
            table.to_csv(folder/'revenue_contributions_by_quintile.csv', index=False)

            bins = sorted(data.group.unique(), key=lambda group: int(group.split('_')[1]))
            x = np.arange(len(bins))
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout='constrained')
            fig.suptitle(exercise.exercise+' — Contribution to total revenue', fontsize=15)
            for ax, h0 in zip(axes, ['j', '0']):
                for regime, name, color, offset in [
                        ('common', 'Uniform', '#555555', -.19),
                        ('informed', 'Informed', '#2E6896', .19)]:
                    values = [shares.loc[(h0, regime, group)] for group in bins]
                    bars = ax.bar(x+offset, values, width=.36, label=name, color=color)
                    ax.bar_label(bars, labels=[f'{value:.1f}%' for value in values],
                                 padding=3, fontsize=8)
                ax.set(title='Initially '+('inside' if h0 == 'j' else 'outside'),
                       xticks=x, xticklabels=[f'Q{k+1}' for k in x],
                       xlabel='Gamma quintile (20% of population each)',
                       ylabel='Share of total two-period revenue')
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
                ax.legend(fontsize=9)
            fig.supxlabel('Quintile revenue / total revenue under the same policy and initial history. '
                          'Each policy sums to 100%.', fontsize=9)
            for ext in ['png', 'pdf']:
                fig.savefig(folder/('revenue_contributions_by_quintile.'+ext),
                            dpi=180, bbox_inches='tight')
            packet.savefig(fig)
            plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(source/'revenue_contributions_by_quintile.csv', index=False)
    print(source/'revenue_contributions_by_quintile.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    build(parser.parse_args().source)
