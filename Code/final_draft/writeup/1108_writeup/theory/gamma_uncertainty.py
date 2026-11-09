"""Theoretical type uncertainty; no empirical records or inputs.
Run from any directory. Outputs go beside this script in ../assets/results/gamma_uncertainty.
Initial history is observed; gamma is known or evaluated using fixed type probabilities in each state.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gamma-uncertainty-mpl')
import json
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize_scalar, brentq
from scipy.special import lambertw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA=np.array([-1.25,0.,1.25]); BETA=2.; ALPHA=-1.; DELTA=.92; COST=0.
OUT=Path(__file__).resolve().parents[1]/'assets/results/gamma_uncertainty'


def maximize(fun, wide=False):
    # Mixture objectives can have multiple stationary points: refine all grid modes.
    lo,hi=(-20.,40.) if wide else (-10.,20.)
    x=np.linspace(lo,hi,121 if wide else 81)
    y=np.array([fun(p) for p in x])
    candidates=[(y[0],x[0]),(y[-1],x[-1])]
    for k in range(1,len(x)-1):
        if y[k]>=y[k-1] and y[k]>=y[k+1]:
            fit=minimize_scalar(lambda p:-fun(p),bounds=(x[k-1],x[k+1]),method='bounded',options={'xatol':1e-10})
            candidates.append((-fit.fun,fit.x))
    value,p=max(candidates)
    assert lo+1e-4<p<hi-1e-4,(p,value)
    return float(p),float(value)


def shares(p,h,gamma=GAMMA):
    return expit(BETA+ALPHA*p+(1 if h=='0' else -1)*gamma)


def terminal(weights,h,wide=False):
    # Fast bracketed derivative for this calibration. Its negative derivative is
    # checked at the solution; the independent grid maximizer audits every case.
    fun=lambda p: float(np.dot(weights,shares(p,h))*(p-COST))
    derivative=lambda p: float(np.dot(weights,shares(p,h)+(p-COST)*ALPHA*shares(p,h)*(1-shares(p,h))))
    p=brentq(derivative,0.,20.,xtol=1e-12)
    return p,fun(p)


def known(h,prior):
    rows=[]
    for k,g in enumerate(GAMMA):
        p2={};values={}
        for a in ['j','0']:
            z=float(lambertw(np.exp(BETA+(1 if a=='0' else -1)*g+ALPHA*COST-1)).real)
            p2[a]=COST-(1+z)/ALPHA;values[a]=-z/ALPHA
        effective=COST-DELTA*(values['j']-values['0'])
        z=float(lambertw(np.exp(BETA+(1 if h=='0' else -1)*g+ALPHA*effective-1)).real)
        p=effective-(1+z)/ALPHA;s=float(shares(p,h,g))
        value=s*(p-COST)+DELTA*(s*values['j']+(1-s)*values['0'])
        rows.append(dict(gamma=float(g),p1=p,p2j=p2['j'],p20=p2['0'],s1=s,profit=value))
    expected_p1=sum(prior[k]*r['p1'] for k,r in enumerate(rows))
    expected_p2=sum(prior[k]*(r['s1']*r['p2j']+(1-r['s1'])*r['p20']) for k,r in enumerate(rows))
    value=sum(prior[k]*r['profit'] for k,r in enumerate(rows))
    return dict(profit=value,expected_p1=expected_p1,expected_p2=expected_p2,types=rows)


def evaluate(p,h,prior,details=False):
    a=float(prior@shares(p,h))
    p2j,vj=terminal(prior,'j');p20,v0=terminal(prior,'0')
    value=a*(p-COST)+DELTA*(a*vj+(1-a)*v0)
    if not details:return value
    return dict(profit=value,expected_p1=p,expected_p2=a*p2j+(1-a)*p20,
                p2j=p2j,p20=p20,s1=a,belief_j=prior.tolist(),belief_0=prior.tolist())


def solve(prior,h,validate=True):
    prior=np.asarray(prior,dtype=float)
    assert np.isclose(prior.sum(),1) and (prior>=0).all()
    p,value=maximize(lambda p:evaluate(p,h,prior))
    unknown=evaluate(p,h,prior,True);full=known(h,prior)
    assert full['profit']+1e-8>=value
    assert unknown['belief_j']==prior.tolist()==unknown['belief_0']
    if validate:
        pwide,vwide=maximize(lambda p:evaluate(p,h,prior),wide=True)
        assert abs(value-vwide)<1e-7
        # Independent direct optimization audits both terminal branches.
        s=shares(p,h)
        for state,w,price in [('j',prior,unknown['p2j']),('0',prior,unknown['p20'])]:
            pp,vv=maximize(lambda x:float(w@shares(x,state)*(x-COST)),wide=True)
            assert abs(vv-float(w@shares(price,state)*(price-COST)))<1e-8
        eps=1e-4
        assert abs((evaluate(p+eps,h,prior)-evaluate(p-eps,h,prior))/(2*eps))<1e-5
    return dict(prior=prior.tolist(),h0=h,known=full,unknown=unknown,value_of_information=full['profit']-value)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    priors=[('Equal types',[1/3]*3),('Habit type more likely',[.5,.25,.25]),('Neutral type more likely',[.25,.5,.25]),('Novelty type more likely',[.25,.25,.5])]
    results=[]
    for label,prior in priors:
        for h in ['j','0']:
            r=solve(prior,h);r['scenario']=label;results.append(r)
    # If type is certain, both information regimes must coincide.
    for k in range(3):
        for h in ['j','0']:
            prior=np.eye(3)[k];r=solve(prior,h)
            assert abs(r['value_of_information'])<1e-8
    (OUT/'solutions.json').write_text(json.dumps(results,indent=2))
    (OUT/'settings.json').write_text(json.dumps(dict(gamma=GAMMA.tolist(),beta=BETA,alpha=ALPHA,delta=DELTA,c=COST,priors=priors,checks='Wider first-price grids; independent terminal maximizers; first-price finite-difference FOCs; Fixed-belief invariance; full-information bound; degenerate priors'),indent=2))
    labels=['Equal','Habit-heavy','Neutral-heavy','Novelty-heavy'];x=np.arange(4)
    for metric,name,ylabel in [('profit','profits','Expected discounted profit'),('expected_p1','first_prices','Expected first-period offered price'),('expected_p2','terminal_prices','Expected terminal offered price')]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True,layout='constrained')
        for ax,h in zip(axes,['j','0']):
            rr=[r for r in results if r['h0']==h]
            for offset,key,color,label in [(-.18,'known','#386cb0','Known type'),(.18,'unknown','#e08032','Uncertain type; fixed beliefs')]:
                ax.bar(x+offset,[r[key][metric] for r in rr],.36,color=color,label=label)
            ax.set_xticks(x,labels,rotation=20);ax.set_title('Initially '+('inside' if h=='j' else 'outside'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylabel(ylabel);handles,ls=axes[0].get_legend_handles_labels();fig.legend(handles,ls,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
        fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True,layout='constrained')
    for ax,h in zip(axes,['j','0']):
        rr=[r for r in results if r['h0']==h]
        ax.bar(x,[r['value_of_information'] for r in rr],color='#386cb0');ax.set_xticks(x,labels,rotation=20);ax.set_title('Initially '+('inside' if h=='j' else 'outside'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Known-type profit minus uncertain-type profit');fig.savefig(OUT/'information_value.pdf');fig.savefig(OUT/'information_value.png',dpi=180);plt.close(fig)
    lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Pricing with known and uncertain switching types}',r'\label{tab:gamma-uncertainty}',r'\begin{tabular}{llrrr}\toprule',r'Initial state & Prior & Known profit & Uncertain profit & Information value\\\midrule']
    for r in results:lines.append(f"${r['h0']}$ & {r['scenario']} & {r['known']['profit']:.4f} & {r['unknown']['profit']:.4f} & {r['value_of_information']:.4f} \\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(OUT/'results.tex').write_text('\n'.join(lines)+'\n')
    print('All checks passed; wrote four figures, aggregate theoretical solutions, settings, and result table.')

if __name__=='__main__':main()
