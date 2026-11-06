"""Export fitted population history distributions, including from saved fits."""
from pathlib import Path
import os
import tempfile
from statistics import NormalDist
import argparse
import json
import numpy as np
import pandas as pd


def export_history_distributions(table, output, converged=True):
    p = table.set_index('parameter').estimate
    if 'mean_novelty' not in p:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provisional = not converged or table.get('at_bound', pd.Series(False)).astype(str).str.lower().eq('true').any()
    summaries, curves = [], []
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, term, label in zip(axes, ['novelty', 'own_stock'], ['Switching / novelty', 'Own-stock habit']):
        mean, sd = float(p['mean_'+term]), float(p.get('sd_'+term, 0.))
        if not np.isfinite([mean, sd]).all() or sd < 0:
            raise ValueError('Invalid distribution parameters')
        normal = NormalDist(mean, sd) if sd > 0 else None
        row = dict(coefficient=term, mean=mean, sd=sd, point_mass=sd == 0,
                   share_below_zero=normal.cdf(0) if normal else float(mean < 0), provisional=bool(provisional))
        for q in [.01, .05, .25, .5, .75, .95, .99]:
            row[f'q{int(100*q):02d}'] = normal.inv_cdf(q) if normal else mean
        summaries.append(row)
        if normal:
            x = np.linspace(mean-4*sd, mean+4*sd, 801)
            density = np.exp(-.5*((x-mean)/sd)**2)/(sd*np.sqrt(2*np.pi))
            ax.plot(x, density)
            curves.extend(dict(coefficient=term, value=float(v), density=float(d), cdf=normal.cdf(float(v)), point_mass=False)
                          for v, d in zip(x, density))
        else:
            ax.axvline(mean, label='Point mass')
            curves.append(dict(coefficient=term, value=mean, density=np.nan, cdf=1., point_mass=True))
        ax.axvline(0, color='gray', ls='--')
        ax.set(title=f'{label}\nMean={mean:.4g}; SD={sd:.4g}', xlabel='Utility coefficient', ylabel='Population density')
    pd.DataFrame(summaries).to_csv(output/'history_distribution_summary.csv', index=False)
    pd.DataFrame(curves).to_csv(output/'history_distribution_grid.csv', index=False)
    fig.suptitle('Fitted population history distributions'+(' (provisional)' if provisional else ''))
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(output/f'history_distributions.{extension}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    fit = json.loads((args.fit_dir/'fit.json').read_text())
    if fit['model'] not in ['separate_history', 'heterogeneous_history']:
        parser.error('Requires a split-history fit')
    export_history_distributions(pd.read_csv(args.fit_dir/'point_estimates.csv'),
                                 args.output or args.fit_dir, fit.get('converged', False))
