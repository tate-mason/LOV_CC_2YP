"""Rebuild slide price figures with one y scale per figure, using aggregates only."""
from pathlib import Path
import json
import ast
import re
import os
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'slide-price-scales-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
sys.path.insert(0, str(ROOT / 'Code/final_simple/demand'))
# Load the existing rendering function without importing estimation dependencies.
# The local SciPy binary is unavailable; evaluate w + log(w) = x by bisection.
def wrightomega(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.zeros_like(x), np.maximum(1., x + 1.)
    for _ in range(100):
        mid = (lo + hi) / 2
        below = mid + np.log(mid) < x
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    result = (lo + hi) / 2
    np.testing.assert_allclose(result + np.log(result), x, atol=1e-12)
    return result


def known_prices(gamma, h0, alpha, beta, delta):
    terminal_z = wrightomega(beta + np.array([-gamma, gamma]) - 1)
    terminal_value = -terminal_z / alpha
    effective = -delta * (terminal_value[0] - terminal_value[1])
    intercept = beta + (-gamma if h0 == 'j' else gamma)
    first_z = float(wrightomega(intercept + alpha * effective - 1))
    return {'expected_p1': effective - (1 + first_z) / alpha}


source = ROOT / 'Code/final_simple/demand/quantified_theory.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'figures')
namespace = dict(np=np, known_prices=known_prices, wrightomega=wrightomega,
                 prior_labels=lambda values: values)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
figures = namespace['figures']
from pricing_presentation import information_price_paths_comparison, information_figures


def main():
    # Capture only the requested price plots; do not regenerate other assets.
    captured = {}
    original_save = matplotlib.figure.Figure.savefig
    price_names = {'optimal_prices_by_gamma', 'information_price_paths_comparison'}
    names = price_names | {'information_revenues', 'information_friction_profits'}

    def capture(fig, filename, *args, **kwargs):
        path = Path(filename)
        if path.stem in names:
            captured[path] = (fig, args, kwargs)
            if path.stem == 'information_friction_profits':
                captured[path.with_name('profits' + path.suffix)] = (fig, args, kwargs)

    matplotlib.figure.Figure.savefig = capture
    try:
        theory = HERE / 'assets/theory_delta1'
        curves = pd.read_csv(theory / 'solutions.csv')  # Pure theoretical solutions.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for state, label in [('j', r'Initially inside ($x_0=j$)'), ('0', r'Initially outside ($x_0=0$)')]:
            rows = curves[curves.initial_state == state]
            axes[0].plot(rows.gamma, rows.p1, label=label, lw=2)
        rows = curves[curves.initial_state == 'j']
        for key, label in [('p2j', r'Previous inside ($x_1=j$)'), ('p20', r'Previous outside ($x_1=0$)')]:
            axes[1].plot(rows.gamma, rows[key], label=label, lw=2)
        for ax, title, ylabel in zip(axes, ['Optimal period-1 prices', 'Optimal period-2 prices'], [r'$p_{1r}^*$', r'$p_{2r}^*$']):
            ax.axhline(0, color='gray', lw=.7)
            ax.set(xlabel=r'$\gamma_r$', ylabel=ylabel, title=title)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(theory / f'optimal_prices_by_gamma.{ext}', dpi=160)
        plt.close(fig)
        folder = theory / 'gamma_uncertainty'
        information_price_paths_comparison(
            pd.read_csv(folder / 'information_friction_by_group.csv', dtype={'h0': str}),
            pd.read_csv(folder / 'information_friction_aggregate.csv', dtype={'h0': str}),
            folder, 'focal flavor')
        information_figures(pd.read_csv(folder / 'information_friction_aggregate.csv', dtype={'h0': str}), folder, delta=1.)
        for flavor in ['plain', 'berry', 'other']:
            folder = HERE / 'assets/quantified_theory_observed_128' / flavor
            settings = json.loads((folder / 'settings.json').read_text())
            results = pd.read_csv(folder / 'type_results.csv', dtype={'h0': str})
            summary = pd.read_csv(folder / 'comparison.csv', dtype={'h0': str})
            support = pd.read_csv(folder / 'gamma_distribution.csv').gamma.to_numpy()
            for row in results[results.regime == 'informed'].itertuples():
                rebuilt = known_prices(row.gamma, row.h0, settings['alpha'], settings['beta'], settings['delta'])
                np.testing.assert_allclose(rebuilt['expected_p1'], row.p1, atol=1e-10)
            figures(results, summary, settings['alpha'], settings['beta'], settings['delta'], support, folder)
    finally:
        matplotlib.figure.Figure.savefig = original_save

    scales = {}
    for path, (fig, args, kwargs) in captured.items():
        if path.stem in price_names:
            values = np.concatenate([line.get_ydata() for ax in fig.axes for line in ax.lines])
            assert np.isfinite(values).all()
            span = max(float(np.ptp(values)), 1e-6)
            ticks = MaxNLocator(nbins=7).tick_values(values.min() - .04 * span, values.max() + .04 * span)
            limits = (float(ticks[0]), float(ticks[-1]))
            for ax in fig.axes:
                ax.set_ylim(*limits)
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', labelleft=True)
                assert ax.get_ylim() == limits
            scales[str(path.relative_to(HERE))] = limits
        from matplotlib.text import Text
        for item in fig.findobj(Text):
            label = item.get_text()
            for old in ['Household-specific', 'Personalized', 'Informed']:
                label = label.replace(old, 'Type-specific')
            label = re.sub(r'\binformed\b', 'type-specific', label)
            label = label.replace('complete information', 'type-specific pricing')
            label = label.replace('type unobserved', 'uniform pricing')
            item.set_text(label)
        original_save(fig, path, *args, **kwargs)
    print(f'Rebuilt {len(captured)//2} figures (PDF and PNG); y limits shared within each figure only.')
    print(json.dumps({name: limits for name, limits in scales.items() if name.endswith('.pdf')}, indent=2))


if __name__ == '__main__':
    main()
