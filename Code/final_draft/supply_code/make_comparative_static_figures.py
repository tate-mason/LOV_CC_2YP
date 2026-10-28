"""Re-solve both FOCs at each gamma; save reproducible figures and source data."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp
from model.numerical import solve
from model.symbols import alpha, delta, c, sj1, s01, sj2, s02, z1, z2
from model.symbols import p1 as p1_symbol, p2 as p2_symbol
from comparative_statics.alpha_p1 import dp1_dalpha
from comparative_statics.alpha_p2 import dp2_dalpha
from comparative_statics.beta_p1 import dp1_dbeta
from comparative_statics.beta_p2 import dp2_dbeta
from comparative_statics.delta_p1 import dp1_ddelta
from comparative_statics.delta_p2 import dp2_ddelta
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.gamma_profit import dPi_dgamma
from comparative_statics.beta_p1p2 import ratio_derivative as beta_ratio_derivative
from comparative_statics.gamma2_p1 import d2p1_dgamma2
from comparative_statics.gamma2_p2 import d2p2_dgamma2
from comparative_statics.gamma_p2beta import dp2dbeta_dgamma
from comparative_statics.gamma_p1beta import dp1dbeta_dgamma_clean
from pricing.period1_foc import foc1_collected
from pricing.period2_foc import foc2

OUT = Path(__file__).resolve().parent / 'writeup_math_graphs' / 'figures'
PARAMETERS = dict(alpha=-2., beta=-1., delta=.7, c=1.)
CASES = {'(0,0)':(0,0), '(1,0)':(1,0), '(0,1)':(0,1)}
# Each expression uses the same endogenous shares as the solved FOCs.
DERIVATIVES = {
    'dp1_dbeta_dgamma': (dp1dbeta_dgamma_clean, r'$\mathrm{d}^2p_{j1}^*/(\mathrm{d}\gamma\,\mathrm{d}\beta)$', 'beta_p1_derivative_by_gamma'),
    'dp2_dbeta_dgamma': (dp2dbeta_dgamma, r'$\mathrm{d}^2p_{j2}^*/(\mathrm{d}\gamma\,\mathrm{d}\beta)$', 'beta_p2_derivative_by_gamma'),
    'd2p1_dgamma2': (d2p1_dgamma2, r'$\mathrm{d}^2p_{j1}^*/\mathrm{d}\gamma^2$', 'gamma2_p1_by_gamma'),
    'd2p2_dgamma2': (d2p2_dgamma2, r'$\mathrm{d}^2p_{j2}^*/\mathrm{d}\gamma^2$', 'gamma2_p2_by_gamma'),
    'dp1_dgamma': (dp1_dgamma, r'$\mathrm{d}p_{j1}^*/\mathrm{d}\gamma$', 'gamma_p1_by_gamma'),
    'dp2_dgamma': (dp2_dgamma, r'$\mathrm{d}p_{j2}^*/\mathrm{d}\gamma$', 'gamma_p2_by_gamma'),
    'dp1_dbeta': (dp1_dbeta, r'$\mathrm{d}p_{j1}^*/\mathrm{d}\beta$', 'beta_p1_by_gamma'),
    'dp2_dbeta': (dp2_dbeta, r'$\mathrm{d}p_{j2}^*/\mathrm{d}\beta$', 'beta_p2_by_gamma'),
    'dp1_dalpha': (dp1_dalpha, r'$\mathrm{d}p_{j1}^*/\mathrm{d}\alpha$', 'alpha_p1_by_gamma'),
    'dp2_dalpha': (dp2_dalpha, r'$\mathrm{d}p_{j2}^*/\mathrm{d}\alpha$', 'alpha_p2_by_gamma'),
    'dp1_ddelta': (dp1_ddelta, r'$\mathrm{d}p_{j1}^*/\mathrm{d}\delta$', 'delta_p1_by_gamma'),
    'dp2_ddelta': (dp2_ddelta, r'$\mathrm{d}p_{j2}^*/\mathrm{d}\delta$', 'delta_p2_by_gamma'),
    'profit_dgamma': (dPi_dgamma, r'$\mathrm{d}\Pi^*/\mathrm{d}\gamma$', 'gamma_profit_by_gamma'),
    'ratio_dbeta': (beta_ratio_derivative, r'$\mathrm{d}(p_{j1}^*/p_{j2}^*)/\mathrm{d}\beta$', 'beta_price_ratio_by_gamma'),
}
DERIVATIVES = {key: (expr, label.replace('p_{j1}', 'p_1').replace('p_{j2}', 'p_2'), name)
               for key, (expr,label,name) in DERIVATIVES.items()}
EVALUATORS = {
    key: sp.lambdify((alpha,delta,sj1,s01,sj2,s02,z1,z2,p1_symbol,p2_symbol), expr, 'numpy')
    for key,(expr,_,_) in DERIVATIVES.items()
}


def plot_pricing_focs():
    """Vary one price, holding the other at its optimum, at gamma=1.5."""
    g=1.5
    a,b,d,cost=(PARAMETERS[k] for k in ('alpha','beta','delta','c'))
    rows=[]
    for period,expression in ((1,foc1_collected),(2,foc2)):
        evaluate=sp.lambdify((alpha,delta,p1_symbol,p2_symbol,sj1,s01,sj2,s02,c),expression,'numpy')
        fig,ax=plt.subplots(figsize=(7,4.5))
        for label,(h1,h2) in CASES.items():
            x,y,*_=solve(**PARAMETERS,gamma=g,z1=h1,z2=h2)
            optimum=x if period==1 else y
            prices=np.linspace(0,max(2,2*optimum),201)
            xgrid=prices if period==1 else np.full_like(prices,x)
            ygrid=prices if period==2 else np.full_like(prices,y)
            s=1/(1+np.exp(-(b+g*h1+a*xgrid)))
            t=1/(1+np.exp(-(b+g*h2+a*ygrid)))
            values=evaluate(a,d,xgrid,ygrid,s,1-s,t,1-t,cost)
            line,=ax.plot(prices,values,label=label,linewidth=2)
            ax.plot(optimum,0,'o',color=line.get_color())
            rows.extend(zip([period]*len(prices),[label]*len(prices),prices,values))
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(xlabel=f'Period-{period} price',ylabel=f'Period-{period} FOC',
               title=f'Period-{period} pricing FOC; gamma=1.5')
        ax.legend(title='(z1,z2)')
        ax.grid(alpha=.2)
        fig.text(.5,.01,'alpha=-2, beta=-1, delta=0.7, c=1; other price fixed at optimum',
                 ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ('png','pdf'):
            fig.savefig(OUT/f'period{period}_foc.{ext}',dpi=180)
        plt.close(fig)
    with (OUT/'pricing_focs.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['period','history','price','foc'])
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(0,3,201)
    curves={}
    rows=[]
    for label,(z1,z2) in CASES.items():
        result=np.array([solve(**PARAMETERS,gamma=g,z1=z1,z2=z2) for g in grid])
        p1,p2,s1,s2,profit=result.T
        a,d=PARAMETERS['alpha'],PARAMETERS['delta']
        derivative=(-p2*z1*s1+z2*s2*(p1+d*(1-s1)*p2))/(a*p2**2)
        derivatives=[np.broadcast_to(evaluate(a,d,s1,1-s1,s2,1-s2,z1,z2,p1,p2),grid.shape)
                     for evaluate in EVALUATORS.values()]
        curves[label]=(p1,p2,s1,s2,profit,derivative,*derivatives)
        rows.extend(zip([label]*len(grid),grid,p1,p2,s1,s2,profit,p1/p2,derivative,*derivatives))
    with (OUT/'solutions.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['history','gamma','p1','p2','s1','s2','profit','ratio','ratio_derivative',*DERIVATIVES])
        writer.writerows(rows)
    specs=[('profit_by_gamma',4,'Optimized expected profit'),
           ('price_ratio_by_gamma',None,'Optimal price ratio p1/p2'),
           ('price_ratio_derivative_cases',5,'d(p1/p2)/d gamma'),
           ('p1_by_gamma',0,'Optimal period-1 price'),
           ('p2_by_gamma',1,'Optimal period-2 price')]
    specs.extend((name,index,ylabel) for index,(_,ylabel,name) in enumerate(DERIVATIVES.values(),start=6))
    for name,index,ylabel in specs:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for label,values in curves.items():
            y=values[0]/values[1] if index is None else values[index]
            ax.plot(grid,y,label=label,linewidth=2)
        if index is not None and index >= 5:
            ax.axhline(0,color='black',linewidth=.7)
        ax.set(xlabel='Switching preference gamma',ylabel=ylabel)
        ax.set_title('Solutions of both logit pricing FOCs')
        ax.legend(title='(z1,z2)')
        ax.grid(alpha=.2)
        fig.text(.5,.01,'alpha=-2, beta=-1, delta=0.7, c=1; shares adjust with prices',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ('png','pdf'):
            fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    plot_pricing_focs()
    print(f'Saved {len(specs)+2} figures (PNG/PDF), solutions.csv and pricing_focs.csv in {OUT}/')

if __name__=='__main__':
    main()
