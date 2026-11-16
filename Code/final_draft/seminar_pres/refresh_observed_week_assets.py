"""Refresh presentation assets from explicit aggregate-only corrected fit exports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIT = ROOT / 'fixed_est/refined_128'
QUICK = ROOT / 'fixed_est/quick_32'
DEMAND = HERE / 'assets/demand_observed_128'
PRICING = HERE / 'assets/quantified_theory_observed_128'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def table(path, header, body, columns, adjust=False):
    lines = ([r'\begin{adjustbox}{max width=\linewidth}'] if adjust else [])
    lines += [r'\begin{tabular}{' + columns + r'}\toprule', header + r'\\\midrule']
    lines += [' & '.join(row) + r' \\' for row in body]
    lines += [r'\bottomrule\end{tabular}']
    if adjust:
        lines += [r'\end{adjustbox}']
    path.write_text('\n'.join(lines) + '\n')


def run():
    DEMAND.mkdir(parents=True, exist_ok=True)
    estimates = rows(FIT / 'estimates.csv')
    p = {r['parameter']: float(r['estimate']) for r in estimates}
    fit = json.loads((FIT / 'fit_summary.json').read_text())
    if fit['observation_mode'] != 'observed_panel' or not fit['converged']:
        raise ValueError('Presentation refresh requires the converged corrected-calendar fit.')
    if any(r['at_bound'].lower() == 'true' for r in estimates):
        raise ValueError('Inspect bound estimates before refreshing presentation.')
    labels = {
        'inside_constant': r'Inside constant $\beta_0$',
        'beta_berry': 'Berry vs. other', 'beta_plain': 'Plain vs. other',
        'mu_log_minus_alpha': r'Mean log price magnitude $\mu_{\log a}$',
        'mean_novelty': r'Mean novelty $\mu_\gamma$',
        'mean_own_stock': r'Mean habit $\mu_\phi$',
        'sd_log_minus_alpha': r'SD log price magnitude $\sigma_{\log a}$',
        'sd_novelty': r'SD novelty $\sigma_\gamma$',
        'sd_own_stock': r'SD habit $\sigma_\phi$',
    }
    table(DEMAND / 'estimates.tex', 'Parameter & Estimate & Standard error',
          [[labels[r['parameter']], f"{float(r['estimate']):.4f}", f"{float(r['std_error']):.4f}"]
           for r in estimates], 'lrr', adjust=True)
    table(DEMAND / 'estimates_intervals.tex', r'Parameter & Estimate & SE & 95\% interval',
          [[labels[r['parameter']], f"{float(r['estimate']):.3f}", f"{float(r['std_error']):.3f}",
            f"[{float(r['ci_lower']):.3f}, {float(r['ci_upper']):.3f}]"] for r in estimates], 'lrrr')
    audit = fit['calendar_audit']
    zero_share = audit['retained_zero_purchase_weeks'] / fit['weeks']
    table(DEMAND / 'fit_summary.tex', 'Specification & Three flavors; no price controls', [
        ['Observation calendar', 'Observed full-shopping-panel weeks'],
        ['Habit history', 'Average prior purchase-week flavor shares'],
        ['Households', f"{fit['households']:,}"],
        ['Household-week observations', f"{fit['weeks']:,}"],
        ['Zero-purchase weeks', f"{audit['retained_zero_purchase_weeks']:,} ({100*zero_share:.2f}" + r'\%)'],
        ['Negative log likelihood', f"{fit['nll']:,.2f}"],
        ['AIC', f"{fit['aic']:,.2f}"], ['BIC', f"{fit['bic']:,.2f}"],
        ['Optimizer / inference', 'Converged / Hessian SEs']], 'lr')
    distributions = rows(FIT / 'history_distribution_summary.csv')
    table(DEMAND / 'heterogeneity.tex', 'Preference & Mean & Population SD & Share positive',
          [[{'novelty': 'Novelty', 'own_stock': 'Habit'}[r['coefficient']],
            f"{float(r['mean']):.3f}", f"{float(r['sd']):.3f}",
            f"{100*(1-float(r['share_below_zero'])):.1f}" + r'\%'] for r in distributions], 'lrrr')
    mu, sd = p['mu_log_minus_alpha'], p['sd_log_minus_alpha']
    alpha_mean = -math.exp(mu + sd*sd/2)
    alpha_sd = math.sqrt(math.expm1(sd*sd) * math.exp(2*mu+sd*sd))
    (DEMAND / 'price_moments.tex').write_text(
        f'Population mean price coefficient: $E[\\alpha_i]={alpha_mean:.3f}$.\\\\\n'
        f'Population standard deviation: $\\mathrm{{SD}}(\\alpha_i)={alpha_sd:.3f}$.\n')
    wtp = rows(FIT / 'implied_wtp.csv')
    wtp_labels = dict(inside_constant=r'Inside constant $\beta_0$', beta_berry='Berry vs. other',
                      beta_plain='Plain vs. other', novelty='Novelty', own_stock='Habit')
    table(DEMAND / 'implied_wtp.tex', 'Utility term & Mean WTP & Population SD',
          [[wtp_labels[r['term']], f"{float(r['mean_wtp']):.4f}", f"{float(r['population_sd']):.4f}"]
           for r in wtp], 'lrr', adjust=True)
    table(DEMAND / 'parameter_mapping.tex', 'Input & Corrected 128-draw estimate', [
        [r'Plain intercept $\beta_0+\beta_{\mathrm{plain}}$', f"{p['inside_constant']+p['beta_plain']:.3f}"],
        [r'Berry intercept $\beta_0+\beta_{\mathrm{berry}}$', f"{p['inside_constant']+p['beta_berry']:.3f}"],
        [r'Other intercept $\beta_0$', f"{p['inside_constant']:.3f}"],
        [r'Mean novelty $\mu_\gamma$', f"{p['mean_novelty']:.3f}"],
        [r'SD novelty $\sigma_\gamma$', f"{p['sd_novelty']:.3f}"]], 'lr')
    quick = {r['parameter']: float(r['estimate']) for r in rows(QUICK / 'point_estimates.csv')}
    sensitivity_names = ['mean_novelty', 'sd_novelty', 'mean_own_stock', 'sd_own_stock', 'sd_log_minus_alpha']
    table(DEMAND / 'draw_sensitivity.tex', 'Parameter & 32 draws & 128 draws',
          [[labels[name], f'{quick[name]:.3f}', f'{p[name]:.3f}'] for name in sensitivity_names], 'lrr')
    shutil.copyfile(FIT / 'history_distributions.pdf', DEMAND / 'history_distributions.pdf')
    inputs = ['point_estimates.csv', 'estimates.csv', 'fit_summary.json', 'implied_wtp.csv',
              'history_distribution_summary.csv', 'history_distributions.pdf']
    (DEMAND / 'provenance.json').write_text(json.dumps(dict(
        source='fixed_est/refined_128', observation_mode='observed_panel', draws=128,
        stock_measure=fit['stock_measure'],
        inputs={name: hashlib.sha256((FIT / name).read_bytes()).hexdigest() for name in inputs}), indent=2) + '\n')
    totals = rows(PRICING / 'all_flavors_comparison.csv')
    pricing_body = []
    percentages = {}
    for flavor in ['plain', 'berry', 'other']:
        summary = [r for r in totals if r['flavor'] == flavor and r['scenario'] == 'Fitted distribution']
        welfare_body = []
        for h0, history in [('j', 'Inside'), ('0', 'Outside')]:
            c = next(r for r in summary if r['h0'] == h0 and r['regime'] == 'common')
            i = next(r for r in summary if r['h0'] == h0 and r['regime'] == 'informed')
            gain = float(i['revenue']) - float(c['revenue'])
            pricing_body.append([flavor.title(), history, f"{float(i['revenue']):.4f}",
                                 f"{float(c['revenue']):.4f}", f'{gain:.4f}'])
            changes = {key: 100*(float(i[key])-float(c[key]))/abs(float(c[key]))
                       for key in ['revenue', 'profit', 'consumer_surplus', 'total_welfare', 'quantity']}
            if flavor == 'plain':
                percentages[h0] = changes
            welfare_body.append([history] + [f'{changes[key]:+.1f}' + r'\%'
                                for key in ['profit', 'consumer_surplus', 'total_welfare', 'quantity']])
        table(PRICING / flavor / 'welfare_table_percent.tex',
              r'Initial state & $\Delta\Pi$ & $\Delta CS$ & $\Delta W$ & $\Delta Q$', welfare_body, 'lrrrr')
    table(PRICING / 'all_flavors_revenue.tex', 'Focal flavor & Initial history & Type-specific & Uniform & Gain',
          pricing_body, 'llrrr')
    macros = dict(correctedPriceMean=f'{alpha_mean:.3f}',
                  correctedZeroShare=f'{100*zero_share:.2f}',
                  correctedNoveltyWTP=f"-\\${abs(float(next(r['mean_wtp'] for r in wtp if r['term']=='novelty'))):.2f}",
                  correctedHabitWTP=f"{float(next(r['mean_wtp'] for r in wtp if r['term']=='own_stock')):.2f}",
                  correctedHabitTenWTP=f"{.1*float(next(r['mean_wtp'] for r in wtp if r['term']=='own_stock')):.2f}")
    for h0, label in [('j', 'Inside'), ('0', 'Outside')]:
        for key, metric in [('revenue', 'Revenue'), ('consumer_surplus', 'CS'), ('total_welfare', 'Welfare')]:
            value = percentages[h0][key]
            macros['corrected'+label+metric+'Change'] = f'{abs(value):.1f}'
            macros['corrected'+label+metric+'Direction'] = 'rises' if value >= 0 else 'falls'
    (DEMAND / 'values.tex').write_text('\n'.join(
        '\\newcommand{\\' + name + '}{' + value + '}' for name, value in macros.items()) + '\n')
    print(json.dumps(dict(pricing_percent_changes=percentages, mean_price=alpha_mean,
                          zero_share=zero_share), indent=2))


if __name__ == '__main__':
    run()
