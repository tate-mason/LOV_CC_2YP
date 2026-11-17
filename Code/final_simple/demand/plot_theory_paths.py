"""Expected price paths from saved pure-theory support-point outcomes."""
from pathlib import Path
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'theory-paths-mpl'))


def plot_paths(types, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    required = {'scenario', 'h0', 'gamma', 'regime', 'p1', 'p2j', 'p20', 'quantity1', 'expected_p2'}
    if not required <= set(types) or {'household_code', 'household_id', 'week', 'upc'} & set(types):
        raise ValueError('Expected pure-theory support-point outcomes only.')
    np.testing.assert_allclose(types.expected_p2,
                               types.quantity1*types.p2j+(1-types.quantity1)*types.p20)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, (scenario, data) in enumerate(types.groupby('scenario', sort=False), 1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout='constrained')
        fig.suptitle(scenario)
        gammas = sorted(data.gamma.unique())
        colors = plt.get_cmap('viridis')(np.linspace(.1, .9, len(gammas)))
        for ax, h0 in zip(axes, ['j', '0']):
            for gamma, color in zip(gammas, colors):
                for regime, style, marker in [('informed', '-', 'o'), ('common', '--', 'D')]:
                    selected = data[(data.h0 == h0)&(data.gamma == gamma)&(data.regime == regime)]
                    if len(selected) != 1:
                        raise ValueError('Need one outcome per prior, initial history, type and regime.')
                    row = selected.iloc[0]
                    ax.plot([1, 2], [row.p1, row.expected_p2], color=color, linestyle=style,
                            marker=marker, label=f'γ={gamma:g} / '+('Known type' if regime == 'informed' else 'Uniform'))
            ax.set(title='Initially '+('inside' if h0 == 'j' else 'outside'),
                   xticks=[1, 2], xticklabels=[r'$p_1$', r'$E[p_2]$'], ylabel='Price ($)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=.2)
        for ext in ['png', 'pdf']:
            fig.savefig(output/f'expected_price_paths_prior_{index}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    types[['scenario', 'h0', 'gamma', 'regime', 'p1', 'expected_p2']].to_csv(
        output/'expected_price_paths.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_paths(pd.read_csv(args.source/'welfare_type_results.csv'), args.output)
