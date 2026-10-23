"""Rebuild presentation pricing figures from the current toy-model result CSVs.
Run from the repository root: .venv/bin/python updates/draft_2/presentation/build_pricing_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / 'toy_model' / 'results'
ASSETS = HERE / 'assets'
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def canvas(n=2):
    return plt.subplots(1, n, figsize=(12, 4.5))


def save(fig, axes, name):
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(ASSETS / f'pricing_{name}.{ext}', dpi=180)
    plt.close(fig)


def main():
    mix = pd.read_csv(RESULTS / 'population_mix.csv')
    fixed = pd.read_csv(RESULTS / 'fixed_menu_comparison.csv')
    regions = pd.read_csv(RESULTS / 'fixed_menu_regions.csv')
    gamma = pd.read_csv(RESULTS / 'fixed_menu_profit_by_gamma.csv')
    continuous = pd.read_csv(RESULTS / 'profit_by_gamma.csv')
    discounts = pd.read_csv(RESULTS / 'fixed_discount_comparison.csv')
    params = pd.read_csv(RESULTS / 'parameters.csv').iloc[0]
    fig, axes = canvas()
    axes[0].plot(mix.high_share, mix.common_profit, label='Optimized common discount')
    axes[0].plot(mix.high_share, mix.targeted_profit, label='Type-specific discounts')
    axes[1].plot(mix.high_share, mix.targeting_gain, label='Observed minus unobserved type')
    axes[1].axhline(0, color='black', linewidth=.8)
    for ax, ylabel in zip(axes, ['Expected discounted profit', 'Profit gain from observing type']):
        ax.set(xlabel='High-type population share', ylabel=ylabel)
    save(fig, axes, 'population_profits')

    fig, axes = canvas()
    for i, (key, label) in enumerate([('high','H discount'), ('low','L discount'), ('uniform_50_50','50/50 discount')]):
        axes[0].plot(fixed.probability_H, fixed[f'profit_{key}'], label=label, color=COLORS[i])
    for i, (key,label) in enumerate([('high','H minus 50/50 discount'),('low','L minus 50/50 discount')]):
        axes[1].plot(fixed.probability_H, fixed[f'profit_{key}']-fixed.profit_uniform_50_50, label=label, color=COLORS[i])
    axes[1].axhline(0,color='black',linewidth=.8)
    for boundary in regions.probability_H_min.iloc[1:]:
        for ax in axes:
            ax.axvline(boundary,color='gray',linestyle=':',label=f'Transition {boundary:.5f}')
    for ax,ylabel in zip(axes,['Expected discounted profit','Profit gain over 50/50 discount']):
        ax.set(xlabel='Probability customer is H',ylabel=ylabel)
    save(fig, axes, 'fixed_menu_transitions')

    fig, axes = canvas()
    for i,(key,label) in enumerate([('high','H discount'),('low','L discount'),('uniform_50_50','50/50 discount')]):
        axes[0].plot(gamma.gamma,gamma[f'profit_{key}'],label=label,color=COLORS[i])
    axes[1].plot(continuous.gamma,continuous.targeted_profit,label='Discount optimized for true gamma')
    axes[1].plot(continuous.gamma,continuous.uniform_50_50_profit,label='Fixed 50/50 discount',color=COLORS[2])
    for ax in axes:
        for g,label in [(.1,'L'),(2.,'H')]:
            ax.axvline(g,color='gray',linestyle=':',label=f'{label}: gamma={g:g}')
        ax.set(xlabel='True variety preference gamma',ylabel='Conditional expected profit')
    save(fig,axes,'profit_by_type')

    # True type is fixed within each panel. Reverse the H axis so error increases rightward.
    fig, axes = canvas()
    for ax,kind in zip(axes,['H','L']):
        data=mix.sort_values('high_share',ascending=(kind=='L'))
        error=1-data.high_share if kind=='H' else data.high_share
        profit=data[f'common_profit_{kind}']
        ax.plot(error,profit,label=f'Profit from a true {kind} consumer',color=COLORS[0 if kind=='H' else 1])
        ax.axvline(.5,color='gray',linestyle=':',label='Even type probabilities')
        ax.axhline(profit.iloc[0],color='black',linewidth=.8,linestyle='--',label=f'Known-{kind} optimum')
        ax.set(title=f'True type held fixed at {kind}',xlabel=f"Probability assigned to {'L' if kind=='H' else 'H'} (wrong type)",ylabel='Conditional expected profit')
    save(fig,axes,'mispricing_profits')

    fig, axes = canvas()
    for ax,kind in zip(axes,['H','L']):
        data=mix.sort_values('high_share',ascending=(kind=='L'))
        error=1-data.high_share if kind=='H' else data.high_share
        profit=data[f'common_profit_{kind}']
        ax.plot(error,profit.iloc[0]-profit,label=f'Loss from mispricing true {kind}',color=COLORS[0 if kind=='H' else 1])
        ax.axvline(.5,color='gray',linestyle=':',label='Even type probabilities')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=f'True type held fixed at {kind}',xlabel=f"Probability assigned to {'L' if kind=='H' else 'H'} (wrong type)",ylabel='Profit loss relative to known type')
    save(fig,axes,'mispricing_losses')

    fig, axes = canvas()
    for col,label in [('regular_profit','No discount'),('discount_profit','Fixed L discount'),('reoptimized_common_profit','Optimized common discount')]:
        axes[0].plot(discounts.probability_L,discounts[col],label=label)
    axes[1].plot(discounts.probability_L,discounts.discount_gain,label='Fixed L discount minus no discount')
    axes[1].axhline(0,color='black',linewidth=.8)
    for ax,ylabel in zip(axes,['Expected discounted profit','Profit gain from terminal discount']):
        ax.axvline(params.fixed_discount_cutoff_L,color='gray',linestyle=':',label=f'Break-even {params.fixed_discount_cutoff_L:.5f}')
        ax.set(xlabel='Probability customer is L',ylabel=ylabel)
    save(fig,axes,'terminal_discount')

    fig, axes = canvas()
    axes[0].plot(mix.high_share, mix.discount, label='Optimal discount on good 2')
    axes[0].plot(mix.high_share, mix.effective_good2_after_good1,
                 label='Effective price of good 2')
    axes[0].set(xlabel='Assumed probability customer is H', ylabel='Price / discount')
    outcomes = pd.read_csv(RESULTS / 'switching_outcomes.csv')
    # Compare the no-coupon and L-coupon policies for each actual type.
    for i,kind in enumerate(['H','L']):
        rows = outcomes[(outcomes.true_type == kind) & outcomes.policy.isin(['regular','L'])]
        axes[1].plot(rows.discount, rows.probability_switch_good2,
                     marker='o', color=COLORS[i], label=f'True {kind}: switching to good 2')
    axes[1].set(xlabel='Discount on good 2 after buying good 1',
                ylabel='Probability of buying good 2')
    save(fig, axes, 'discount_mechanism')

    # The displayed numbers and price table share the figure inputs.
    prices = pd.read_csv(RESULTS / 'prices_50_50.csv')
    table = [r'\begin{tabular}{llrrrr}\toprule',
             r'Policy & Period / history & $p_1$ & $p_2$ & $m_1$ & $m_2$\\\midrule']
    for policy,label in [('targeted_H','Observed H'),('targeted_L','Observed L'),('uniform_50_50','Common 50/50')]:
        for _, row in prices[prices.policy == policy].iterrows():
            history = '1 / no prior product' if row.period == 1 else f'2 / {row.history}'
            table.append(f'{label} & {history} & {row.price_good1:.3f} & {row.price_good2:.3f} & {row.margin_good1:.3f} & {row.margin_good2:.3f}' + r'\\')
        if policy != 'uniform_50_50': table.append(r'\addlinespace')
    table.append(r'\bottomrule\end{tabular}')
    (ASSETS/'pricing_current_menus.tex').write_text('\n'.join(table)+'\n')
    middle=mix.loc[(mix.high_share-.5).abs().idxmin()]
    vals={'DiscountH':params.discount_H,'DiscountL':params.discount_L,
          'DiscountCommon':params.discount_common50,
          'EffectiveL':params.regular_price_good2-params.discount_L,
          'EffectiveCommon':params.regular_price_good2-params.discount_common50,
          'ProfitCommon':middle.common_profit,'ProfitObserved':middle.targeted_profit,
          'InfoGain':middle.targeting_gain,'OfferCutoff':params.fixed_discount_cutoff_L,
          'DeltaH':params.delta_H,'DeltaL':params.delta_L,
          'CutLow':regions.probability_H_max.iloc[0],
          'CutHigh':regions.probability_H_min.iloc[-1]}
    for kind in ['H','L']:
        y=mix[f'common_profit_{kind}']
        best=y.iloc[-1] if kind=='H' else y.iloc[0]
        wrong=y.iloc[0] if kind=='H' else y.iloc[-1]
        vals[f'Loss{kind}Even']=best-middle[f'common_profit_{kind}']
        vals[f'Loss{kind}Wrong']=best-wrong
    (ASSETS/'pricing_numbers.tex').write_text('\n'.join(
        rf'\newcommand{{\{name}}}{{{value:.5f}}}' for name,value in vals.items())+'\n')

    # Keep active toy-model figure names consistent with the presentation.
    import shutil
    for source, dest in [('population_profits','profits'),
                         ('fixed_menu_transitions','fixed_menu_transitions'),
                         ('profit_by_type','fixed_menu_profits'),
                         ('terminal_discount','newborn_discount_cutoff'),
                         ('mispricing_profits','mispricing_profits'),
                         ('mispricing_losses','mispricing_losses'),
                         ('discount_mechanism','discount_mechanism')]:
        for ext in ('pdf','png'):
            shutil.copy2(ASSETS/f'pricing_{source}.{ext}', RESULTS/f'{dest}.{ext}')


if __name__ == '__main__':
    main()
