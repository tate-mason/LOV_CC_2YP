"""Generate only the requested symmetric-switching comparative-static figures."""
from pathlib import Path
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PARAMETERS = dict(beta=2., alpha=-1., delta=.92, c=0.)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'writeup_math_graphs'/'figures'
GAMMAS=np.linspace(-3,3,601)
LABELS={'j':r'Initially inside ($x_0=j$)','0':r'Initially outside ($x_0=0$)'}
STATES=[('0','0'),('0','j'),('j','j'),('j','0')]


def state_outcomes(curves):
    """Condition on (x0, x1); terminal demand remains integrated out."""
    rows=[]
    for initial, realized in STATES:
        for r in curves[initial]:
            p2=r['p2j' if realized=='j' else 'p20']
            dp2=r['dp2j' if realized=='j' else 'dp20']
            value=r['V2j' if realized=='j' else 'V20']
            rows.append(dict(
                gamma=r['gamma'], x0=initial, x1=realized,
                probability=r['s'] if realized=='j' else 1-r['s'],
                share1=r['s'], outside_share1=1-r['s'],
                share2=r['u' if realized=='j' else 'v'],
                outside_share2=1-r['u' if realized=='j' else 'v'],
                price_ratio=r['p1']/p2,
                price_ratio_derivative=(r['dp1']*p2-r['p1']*dp2)/p2**2,
                realized_profit=(r['p1']-PARAMETERS['c'])*(realized=='j')
                                +PARAMETERS['delta']*value))
    return rows


def plot_state_outcomes(rows):
    specs=[('all_state_price_ratios','price_ratio',r'$p_1^*(x_0)/p_2^*(x_1)$',
            'Price ratios by realized history'),
           ('all_state_price_ratio_derivatives','price_ratio_derivative',r'$dR_{x_0,x_1}/d\gamma$',
            'Price-ratio derivatives by realized history'),
           ('all_state_realized_profit','realized_profit',r'$\Pi^*(x_0,x_1)$',
            'Profit conditional on realized first-period choice')]
    for name,key,ylabel,title in specs:
        fig,ax=plt.subplots(figsize=(8,5))
        for (initial,realized),style in zip(STATES,['-','--','-.',':']):
            group=[r for r in rows if (r['x0'],r['x1'])==(initial,realized)]
            ax.plot([r['gamma'] for r in group],[r[key] for r in group],
                    ls=style,lw=2,label=rf'$(x_0,x_1)=({initial},{realized})$')
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel=r'$\gamma$',ylabel=ylabel,title=title)
        ax.legend();ax.grid(alpha=.2)
        if key=='realized_profit':
            fig.text(.5,.01,'First-period margin if purchased + discounted expected terminal profit',
                     ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,1))
        save(fig,name)


def plot_state_shares(rows):
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,key,title in zip(axes.flat,
            ['share1','share2','outside_share1','outside_share2'],
            ['Period 1: inside share','Period 2: inside share',
             'Period 1: outside share','Period 2: outside share']):
        for (initial,realized),style in zip(STATES,['-','--','-.',':']):
            group=[r for r in rows if (r['x0'],r['x1'])==(initial,realized)]
            ax.plot([r['gamma'] for r in group],[r[key] for r in group],
                    ls=style,lw=2,label=rf'$({initial},{realized})$')
        ax.set(xlabel=r'$\gamma$',ylabel='Share',title=title,ylim=(0,1))
        ax.grid(alpha=.2)
        ax.legend(title=r'$(x_0,x_1)$',fontsize=8,ncol=2)
    save(fig,'all_state_shares')


def save(fig,name):
    fig.tight_layout()
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-solutions',action='store_true',
                        help='Use the saved benchmark solutions instead of solving again.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.from_solutions:
        with (OUT/'solutions.csv').open(newline='') as file:
            saved=[{k:v if k=='initial_state' else float(v) for k,v in row.items()}
                   for row in csv.DictReader(file)]
        curves={state:[r for r in saved if r['initial_state']==state] for state in ['j','0']}
        for group in curves.values():
            np.testing.assert_allclose([r['gamma'] for r in group],GAMMAS,atol=1e-12)
    else:
        from model.numerical import evaluate
        curves={state:[evaluate(float(g),state,**PARAMETERS) for g in GAMMAS] for state in ['j','0']}
    outcomes=state_outcomes(curves)
    with (OUT/'all_state_outcomes.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    # Averaging conditional profit must recover the ex ante objective.
    for initial in ['0','j']:
        groups={a:[r for r in outcomes if r['x0']==initial and r['x1']==a] for a in ['0','j']}
        expected=[sum(groups[a][i]['probability']*groups[a][i]['realized_profit']
                      for a in ['0','j']) for i in range(len(GAMMAS))]
        np.testing.assert_allclose(expected,[r['profit'] for r in curves[initial]],atol=1e-12)
    plot_state_outcomes(outcomes)
    for r in outcomes:
        for period in [1,2]:
            assert 0<=r[f'share{period}']<=1
            np.testing.assert_allclose(r[f'share{period}']+r[f'outside_share{period}'],1)
    plot_state_shares(outcomes)
    with (OUT/'solutions.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(curves['j'][0]))
        writer.writeheader()
        for state in ['j','0']:
            writer.writerows(curves[state])
    specs=[('profit_by_gamma','profit',r'$\Pi^*$','Optimized two-period profit'),
           ('gamma_profit_by_gamma','dprofit',r'$d\Pi^*/d\gamma$','Optimized profit derivative'),
           ('gamma_p1_by_gamma','dp1',r'$dp_1^*/d\gamma$','Optimal period-1 price derivative'),
           ('matched_price_ratio_derivative','d_matched_ratio',r'$dR_a/d\gamma$','State-matched price-ratio derivatives')]
    for name,key,ylabel,title in specs:
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for state in ['j','0']:
            ax.plot(GAMMAS,[r[key] for r in curves[state]],label=LABELS[state],lw=2)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel=r'$\gamma$',ylabel=ylabel,title=title)
        ax.legend();ax.grid(alpha=.2)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    ax.plot(GAMMAS,[r['d_induce_ratio'] for r in curves['0']],lw=2,
            label=r'$R_{0\to j}=p_1^*(0)/p_2^*(j)$')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel=r'$\gamma$',ylabel=r'$dR_{0\to j}/d\gamma$',title='Induce-then-harvest price-ratio derivative')
    ax.legend();ax.grid(alpha=.2);save(fig,'induce_harvest_price_ratio_derivative')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for state in ['j','0']:
        axes[0].plot(GAMMAS,[r['p1'] for r in curves[state]],label=LABELS[state],lw=2)
    for state,key,label in [('j','p2j',r'Previous inside ($x_1=j$)'),('0','p20',r'Previous outside ($x_1=0$)')]:
        axes[1].plot(GAMMAS,[r[key] for r in curves['j']],label=label,lw=2)
    for ax,title,ylabel in zip(axes,['Optimal period-1 prices','Optimal period-2 prices'],[r'$p_1^*$',r'$p_2^*$']):
        ax.axhline(0,color='gray',lw=.7);ax.set(xlabel=r'$\gamma$',ylabel=ylabel,title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'optimal_prices_by_gamma')
    maximum=max(abs(r[k]) for state in curves for r in curves[state] for k in ['foc1','foc2j','foc20'])
    if maximum>1e-9:
        raise AssertionError(f'FOC residual {maximum}')
    print(f'Saved ten figures, 1,202 solved observations, and {len(outcomes):,} conditional outcomes; max FOC residual={maximum:.3g}')


if __name__=='__main__':
    main()
