"""Build all active pricing figures and numerical slide inputs from free-price results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parents[1]/'toy_model'/'results'
ASSETS=HERE/'assets'


def plot(name,draw):
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    draw(axes)
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(ASSETS/f'pricing_{name}.{ext}',dpi=180)
    plt.close(fig)


def main():
    mix=pd.read_csv(RESULTS/'population_mix.csv');sweep=pd.read_csv(RESULTS/'gamma_H_sweep.csv')
    fixed=pd.read_csv(RESULTS/'fixed_menu_comparison.csv');gamma=pd.read_csv(RESULTS/'profit_by_gamma.csv')
    def population(a):
        a[0].plot(mix.high_share,mix.common_profit,label='Optimized common prices')
        a[0].plot(mix.high_share,mix.targeted_profit,label='Observed-type prices')
        a[1].plot(mix.high_share,mix.targeting_gain,label='Value of observing type')
        for ax in a:ax.set(xlabel='Probability assigned to H',ylabel='Discounted profit per arriving consumer')
    plot('population_profits',population)
    def menus(a):
        for key,label in [('high','H menu'),('low','L menu'),('uniform_50_50','50/50 menu')]:
            a[0].plot(fixed.probability_H,fixed[f'profit_{key}'],label=label)
        for key,label in [('high','H minus 50/50'),('low','L minus 50/50')]:
            a[1].plot(fixed.probability_H,fixed[f'profit_{key}']-fixed.profit_uniform_50_50,label=label)
        a[1].axhline(0,color='black',lw=.8)
        for ax in a:ax.set(xlabel='Probability assigned to H',ylabel='Discounted expected profit / gain')
    plot('fixed_menu_transitions',menus)
    for losses in [False,True]:
        def mispricing(a):
            for ax,kind in zip(a,['H','L']):
                y=mix[f'common_profit_{kind}'];best=y.iloc[-1] if kind=='H' else y.iloc[0]
                ax.plot(mix.high_share,best-y if losses else y,label=f'True {kind} held fixed',color='C0' if kind=='H' else 'C1')
                ax.axvline(.5,color='gray',ls=':',label='Even belief')
                ax.set(xlabel='Probability assigned to H',ylabel='Loss relative to known type' if losses else 'Expected two-period firm profit')
                if not losses:ax.axhline(best,color='black',ls='--',lw=.8,label='Known-type optimum')
        plot('mispricing_losses' if losses else 'mispricing_profits',mispricing)
    def value(a):
        a[0].set_title('Total value: common vs. observed type')
        a[1].set_title('Common prices: profit by period')
        for kind in ['H','L']:
            color='C0' if kind=='H' else 'C1'
            a[0].plot(sweep.gamma_H,sweep[f'{kind}_total'],color=color,label=f'{kind}: common-price value')
            a[0].plot(sweep.gamma_H,sweep[f'known_{kind}_total'],ls='--',color=color,label=f'{kind}: observed-type value')
            a[1].plot(sweep.gamma_H,.95*sweep[f'{kind}_period2'],color=color,label=f'{kind}: discounted period 2')
            a[1].plot(sweep.gamma_H,sweep[f'{kind}_period1'],ls=':',color=color,label=f'{kind}: period 1')
        for ax in a:ax.set(xlabel=r'$\gamma_H$ ($\gamma_L=1$, Pr(H)=0.5)',ylabel='Profit per arriving consumer')
    plot('customer_value',value)
    def induce(a):
        a[0].plot(sweep.gamma_H,sweep.known_H_static_period1,label='H: one-period maximum')
        a[0].plot(sweep.gamma_H,sweep.known_H_period1,label='H: dynamic period-1 profit')
        a[0].plot(sweep.gamma_H,.95*sweep.known_H_period2,label='H: discounted harvesting')
        a[1].plot(sweep.gamma_H,sweep.known_H_induction_sacrifice,label='H: current profit sacrificed')
        for ax in a:ax.set(xlabel=r'$\gamma_H$ ($\gamma_L=1$)',ylabel='Profit per arriving consumer')
    plot('induction_harvesting',induce)
    def distribution(a):
        for kind in ['H','L']:
            a[0].plot(sweep.gamma_H,sweep[f'{kind}_total'],label=f'{kind} atom (mass 0.5)')
        selected=[sweep.iloc[(sweep.gamma_H-g).abs().argmin()] for g in [1.5,2,5,10]]
        for row in selected:
            vals=sorted([row.H_total,row.L_total]);span=max(vals[1]-vals[0],.2)
            a[1].step([vals[0]-.1*span,*vals,vals[1]+.1*span],[0,.5,1,1],where='post',label=f'gamma H = {row.gamma_H:.2g}')
        a[0].set(xlabel=r'$\gamma_H$ ($\gamma_L=1$)',ylabel='Expected customer value')
        a[1].set(xlabel='Expected customer value',ylabel='Cumulative population probability',ylim=(-.03,1.03))
    plot('value_distribution',distribution)
    def truegamma(a):
        for key,label in [('high','Fixed H menu'),('low','Fixed L menu'),('uniform_50_50','Fixed 50/50 menu')]:
            a[0].plot(gamma.gamma,gamma[f'profit_{key}'],label=label)
        a[1].plot(gamma.gamma,gamma.targeted_profit,label='Optimized for true gamma')
        a[1].plot(gamma.gamma,gamma.uniform_50_50_profit,label='Fixed 50/50 menu')
        for ax in a:ax.set(xlabel='True gamma',ylabel='Discounted customer value')
    plot('profit_by_type',truegamma)
    prices=pd.read_csv(RESULTS/'prices_50_50.csv')
    # Show every price: three initial states, six complete terminal histories.
    table=[r'\begin{tabular}{lrrrrrr}\toprule',
           r'& \multicolumn{2}{c}{Observed H} & \multicolumn{2}{c}{Observed L} & \multicolumn{2}{c}{Common 50/50}\\',
           r'History & $p_j$ & $p_k$ & $p_j$ & $p_k$ & $p_j$ & $p_k$\\\midrule']
    for period,title,histories in [(1,'Period 1: initial state',['j','k','0']),
                                  (2,'Period 2: state transition',[f'{a} to {h}' for a in ['j','k','0'] for h in ['j','k']])]:
        table.append(r'\multicolumn{7}{l}{\textbf{'+title+r'}}\\[2pt]')
        for history in histories:
            entries=['$'+history.replace(' to ',r'\to ' )+'$']
            for policy in ['targeted_H','targeted_L','uniform_50_50']:
                row=prices[(prices.policy==policy)&(prices.period==period)&(prices.history==history)].iloc[0]
                entries.extend([f'{row.price_good1:.3f}',f'{row.price_good2:.3f}'])
            table.append(' & '.join(entries)+r'\\')
        if period==1:table.append(r'\midrule')
    table.append(r'\bottomrule\end{tabular}')
    (ASSETS/'pricing_current_menus.tex').write_text('\n'.join(table)+'\n')
    import sys
    sys.path.insert(0,str(HERE.parents[2]))
    import toy_model as tm
    types=[tm.gamma_H,tm.gamma_L]
    common=tm.solve_common(types,[.5,.5]);personal=[tm.solve_common([g],[1]) for g in types]
    rows=[r'\begin{tabular}{llrrr}\toprule',
          r'Prices & True type & Period 1 & $\delta\times$ Period 2 & Total\\\midrule']
    for label,kind,g,p in [('Common','H',types[0],common),('Common','L',types[1],common),
                           ('Observed type','H',types[0],personal[0]),('Observed type','L',types[1],personal[1])]:
        c=tm.components(p.p1,p.p2,[g])[0]
        rows.append(f'{label} & {kind} & {c[0]:.4f} & {tm.delta*c[1]:.4f} & {c[2]:.4f}'+r'\\')
    rows.append(r'\bottomrule\end{tabular}')
    (ASSETS/'pricing_period_results.tex').write_text('\n'.join(rows)+'\n')
    middle=mix.iloc[(mix.high_share-.5).abs().argmin()]
    vals={'ProfitCommon':middle.common_profit,'ProfitObserved':middle.targeted_profit,'InfoGain':middle.targeting_gain}
    for kind in ['H','L']:
        y=mix[f'common_profit_{kind}'];best=y.iloc[-1] if kind=='H' else y.iloc[0]
        vals[f'Loss{kind}Even']=best-middle[f'common_profit_{kind}'];vals[f'Loss{kind}Wrong']=best-(y.iloc[0] if kind=='H' else y.iloc[-1])
    (ASSETS/'pricing_numbers.tex').write_text('\n'.join(rf'\newcommand{{\{k}}}{{{v:.5f}}}' for k,v in vals.items())+'\n')

if __name__=='__main__':main()
