"""Rebuild short-deck pricing assets from named aggregates and pure theory only.

No household panels, purchase records, weekly inputs, or empirical estimation.
Run the theoretical generator first, then this script, from any directory.
"""
from pathlib import Path
import hashlib
import json
import sys
import shutil

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'Code/pricing_presentation.py').is_file())
sys.path.insert(0, str(ROOT/'Code'))
sys.path.insert(0, str(ROOT/'Code/final_simple/demand'))
import pricing_presentation as presentation
from pricing_presentation import information_figures, information_price_paths_comparison, benchmark_figure, plotting, save_figure
from gamma_calibration import known_prices, uncertainty
from scipy.special import expit
sys.path.insert(0, str(ROOT/'Code/final_simple/supply'))
from pricing_model import static_price

# Change slide terminology at render time; saved policy names and calculations stay intact.
_original_save_figure = presentation.save_figure

def save_presentation_figure(fig, output, stems):
    from matplotlib.text import Text
    replacements = {'Personalized pricing': 'Household-specific prices',
                    'Common pricing': 'Common prices',
                    'Personalized − common': 'Household-specific − common',
                    'Common price': 'Common prices'}
    for text in fig.findobj(Text):
        value = text.get_text()
        if value == 'Common price':
            value = 'Common prices'
        for old, new in replacements.items():
            if old != 'Common price':
                value = value.replace(old, new)
        text.set_text(value)
    _original_save_figure(fig, output, stems)

presentation.save_figure = save_presentation_figure

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'Results/pricing_no_disc'
OUTPUT = HERE/'assets/aligned_pricing'


def audit(table, types, settings):
    """Independently reconstruct both revenues from the documented binary model."""
    gamma = np.array(settings['estimated_gamma'])
    a, b, delta = (settings[k] for k in ['alpha', 'beta', 'delta'])
    for scenario, prior in settings['priors']:
        w = np.array(prior)
        for h in ['j', '0']:
            rows = table[(table.scenario == scenario) & (table.h0 == h)].set_index('case')
            known = [known_prices(g, h, a, b, delta) for g in gamma]
            common = uncertainty(gamma, w, h, a, b, delta)
            s1 = expit(b+a*common['expected_p1']+(-gamma if h == 'j' else gamma))
            s2j = expit(b+a*common['p2j']-gamma)
            s20 = expit(b+a*common['p20']+gamma)
            actual = w @ (s1*common['expected_p1']+delta*(
                s1*common['p2j']*s2j+(1-s1)*common['p20']*s20))
            informed = w @ np.array([r['profit'] for r in known])
            np.testing.assert_allclose(rows.loc['known', 'persistent_type_revenue'], informed, atol=1e-8)
            np.testing.assert_allclose(rows.loc['unknown', 'persistent_type_revenue'], actual, atol=1e-8)
            np.testing.assert_allclose(rows.value_of_information, informed-actual, atol=1e-8)
            assert informed+1e-8 >= actual
            group = types[(types.scenario == scenario) & (types.h0 == h)]
            np.testing.assert_allclose(group.probability @ group.known_profit, informed, atol=1e-8)
            np.testing.assert_allclose(group.probability @ group.unknown_policy_persistent_profit, actual, atol=1e-8)


def calibrated_gamma_prices(settings, types):
    """Known-type comparative statics using saved binary plain calibration only."""
    a, b, delta = (settings[k] for k in ['alpha', 'beta', 'delta'])
    support = np.asarray(settings['estimated_gamma'])
    # Cover the same theoretical range; highlight where estimated means lie.
    grid = np.unique(np.r_[np.linspace(-3., 3., 601), support])
    p2j = np.array([static_price(b-g, a) for g in grid])
    p20 = np.array([static_price(b+g, a) for g in grid])
    p1 = {h: np.array([known_prices(g, h, a, b, delta)['expected_p1'] for g in grid])
          for h in ['j', '0']}
    # Check optimal prices against exported type-specific offers, not just plot arrays.
    for name, g in zip(['low', 'medium', 'high'], support):
        for h in ['j', '0']:
            row = types[(types.type == name) & (types.h0 == h)].iloc[0]
            price = known_prices(g, h, a, b, delta)['expected_p1']
            terminal = np.array([static_price(b-g, a), static_price(b+g, a)])
            share = expit(b+a*price+(-g if h == 'j' else g))
            np.testing.assert_allclose([price, share*terminal[0]+(1-share)*terminal[1]],
                                      [row.known_p1, row.known_expected_p2], atol=1e-8)
    # Terminal FOCs and the first-period FOC must hold over the full sweep.
    sj, s0 = expit(b-grid+a*p2j), expit(b+grid+a*p20)
    np.testing.assert_allclose(1+a*p2j*(1-sj), 0., atol=1e-12)
    np.testing.assert_allclose(1+a*p20*(1-s0), 0., atol=1e-12)
    continuation = delta*(p2j*sj-p20*s0)
    for h in ['j', '0']:
        first_share = expit(b+a*p1[h]+(-grid if h == 'j' else grid))
        np.testing.assert_allclose(1+a*(1-first_share)*(p1[h]+continuation), 0., atol=1e-12)
    pd.DataFrame(dict(gamma=grid, p1_inside=p1['j'], p1_outside=p1['0'],
                      p2_after_inside=p2j, p2_after_outside=p20)).to_csv(
                          OUTPUT/'calibrated_prices_by_gamma.csv', index=False)
    plt = plotting()
    for stem, zoom in [('calibrated_prices_by_gamma', False),
                       ('calibrated_prices_by_gamma_detail', True)]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, layout='constrained')
        for ax in axes:
            ax.axvspan(support.min(), support.max(), color='#DCE6ED', alpha=.55,
                       label='Estimated group-mean range')
        for h, color, label in [('j', '#2E6896', 'Initially inside'),
                                ('0', '#B35C21', 'Initially outside')]:
            axes[0].plot(grid, p1[h], color=color, lw=2, label=label)
            values = [known_prices(g, h, a, b, delta)['expected_p1'] for g in support]
            axes[0].scatter(support, values, color=color, s=28, zorder=4)
        for values, color, label in [(p2j, '#2E6896', 'Previous inside'),
                                      (p20, '#B35C21', 'Previous outside')]:
            axes[1].plot(grid, values, color=color, lw=2, label=label)
            axes[1].scatter(support, np.interp(support, grid, values), color=color, s=28, zorder=4)
        for ax, title in zip(axes, ['Optimal period-1 prices', 'Optimal period-2 prices']):
            ax.set(xlabel=r'$\gamma$', title=title)
            ax.grid(axis='y', alpha=.2)
            if zoom:
                margin = .12*(support.max()-support.min())
                ax.set_xlim(support.min()-margin, support.max()+margin)
                ax.set_xticks(support, [f'{g:.3f}' for g in support])
                visible = (grid >= support.min()-margin) & (grid <= support.max()+margin)
                values = np.r_[p1['j'][visible], p1['0'][visible], p2j[visible], p20[visible]]
                ax.set_ylim(values.min()-.05, values.max()+.05)
            else:
                ax.set_xlim(-3., 3.)
            ax.legend(fontsize=8, loc='best')
        axes[0].set_ylabel('Optimal price ($)')
        save_figure(fig, OUTPUT, [stem])
        plt.close(fig)
    print('PASS: calibrated gamma curves reproduce all six exported type/state price paths; full-grid FOCs pass.')


def main():
    folder = SOURCE/'pricing/plain'
    filenames = ['information_friction_aggregate.csv', 'information_friction_by_group.csv',
                 'information_friction_settings.json', 'information_calibration_status.json',
                 'pricing_policy_benchmarks.csv']
    table = pd.read_csv(folder/filenames[0], dtype={'h0': str})
    types = pd.read_csv(folder/filenames[1], dtype={'h0': str})
    settings = json.loads((folder/filenames[2]).read_text())
    assert json.loads((folder/filenames[3]).read_text())['status'] == 'complete'
    assert settings['delta'] == 1.
    audit(table, types, settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibrated_gamma_prices(settings, types)
    information_figures(table, OUTPUT, delta=1.)
    information_price_paths_comparison(types, table, OUTPUT, 'plain')
    benchmarks = pd.read_csv(folder/filenames[4])
    benchmark_figure(benchmarks, OUTPUT, 'plain')
    totals = benchmarks.set_index('policy').discounted_revenue
    common, personal = totals['Common pricing'], totals['Personalized pricing']
    (OUTPUT/'coupon_headline.tex').write_text(
        f'\\${common:,.2f} to \\${personal:,.2f} (+{100*(personal/common-1):.2f}\\%)')
    # Short table uses the SAME persistent-type revenue column as every chart.
    lines = [r'\begin{tabular}{llrrr}\toprule',
             r'Initial state & Prior & Household-specific & Common & Gain\\\midrule']
    for scenario, _ in settings['priors']:
        for h in ['j', '0']:
            rows = table[(table.scenario == scenario) & (table.h0 == h)].set_index('case')
            informed, uninformed = rows.loc[['known', 'unknown'], 'persistent_type_revenue']
            lines.append(f'${h}$ & {scenario} & {informed:.4f} & {uninformed:.4f} & {informed-uninformed:.4f} '+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (OUTPUT/'information_results.tex').write_text('\n'.join(lines)+'\n')
    theory = HERE/'assets/theory_delta1/gamma_uncertainty'
    th = pd.read_csv(theory/'information_friction_aggregate.csv', dtype={'h0': str})
    equal = th[th.scenario == 'Equal types']
    gains = equal[equal.case == 'known'].set_index('h0').value_of_information
    np.testing.assert_allclose([gains['j'], gains['0']], [.055506, .264245], atol=1e-6)
    # Identical type-versus-common path format in both sections.
    th_paths = pd.read_csv(theory/'information_friction_by_group.csv', dtype={'h0': str})
    information_price_paths_comparison(th_paths, th, theory, 'focal flavor')
    information_figures(th, theory, delta=1.)
    for stem in ['profits', 'information_value', 'first_prices', 'terminal_prices']:
        for ext in ['pdf', 'png']:
            shutil.copyfile(theory/f'information_friction_{stem}.{ext}', theory/f'{stem}.{ext}')
    provenance = dict(operation='aggregate-only presentation alignment; no empirical rerun',
        empirical_source=str(SOURCE.relative_to(ROOT)), delta=1.,
        revenue_evaluation='same persistent consumer types under both policies',
        common_policy='fixed-belief policy, no Bayesian updating; evaluated under persistent types',
        empirical_stock_measure='cumulative (saved fit); average-stock specification not yet estimated',
        information_calibration_price_target_households=settings['households_in_price_targets'],
        excluded_price_target_households=settings['households_without_complete_price_targets'],
        estimated_gamma=settings['estimated_gamma'],
        gamma_sweep=[-3., 3.], gamma_sweep_interpretation='calibrated comparative statics, not estimated support',
        note='binary calibration maps novelty to symmetric state dependence; full split-history coupon exercise is separate',
        coupon_common=float(common), coupon_personal=float(personal),
        coupon_gain_pct=float(100*(personal/common-1)),
        inputs={str((folder/name).relative_to(ROOT)): hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in filenames})
    (OUTPUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('PASS: eight calibrated state/prior cells reproduce persistent-type revenues; theory comparison checked.')
    print(f'Coupon revenue: {common:.2f} -> {personal:.2f} (+{100*(personal/common-1):.2f}%).')


if __name__ == '__main__':
    main()
