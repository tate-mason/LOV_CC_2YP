"""Generate only the requested symmetric-switching comparative-static figures."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.numerical import evaluate, PARAMETERS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'writeup_math_graphs'/'figures'
GAMMAS=np.linspace(-3,3,601)
LABELS={'j':r'Initially inside ($x_0=j$)','0':r'Initially outside ($x_0=0$)'}


def save(fig,name):
    fig.tight_layout()
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    curves={state:[evaluate(float(g),state,**PARAMETERS) for g in GAMMAS] for state in ['j','0']}
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
    print(f'Saved six requested figures and 1,202 solved observations; max FOC residual={maximum:.3g}')


if __name__=='__main__':
    main()
