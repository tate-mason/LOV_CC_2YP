"""Theoretical type uncertainty; no empirical records or inputs.
Run from any directory. Outputs go beside this script in ../assets/results/gamma_uncertainty.
Initial history is observed; gamma is known or evaluated using fixed type probabilities in each state.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gamma-uncertainty-mpl')
import json
import argparse
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'final_simple/supply'))
from pricing_model import expit, golden_maximum, static_price
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA=np.array([-1.25,0.,1.25]); BETA=2.; ALPHA=-1.; DELTA=1.0; COST=0.
OUT=Path(__file__).resolve().parents[1]/'assets/results/gamma_uncertainty'


def maximize(fun, wide=False):
    # Mixture objectives can have multiple stationary points: refine all grid modes.
    lo,hi=(-20.,40.) if wide else (-10.,20.)
    x=np.linspace(lo,hi,121 if wide else 81)
    y=np.array([fun(p) for p in x])
    candidates=[(y[0],x[0]),(y[-1],x[-1])]
    for k in range(1,len(x)-1):
        if y[k]>=y[k-1] and y[k]>=y[k+1]:
            price,value=golden_maximum(fun,x[k-1],x[k+1])
            candidates.append((value,price))
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
    lo,hi=0.,20.
    for _ in range(100):
        mid=(lo+hi)/2
        if derivative(mid)>0: lo=mid
        else: hi=mid
    p=(lo+hi)/2
    return p,fun(p)


def known(h,prior):
    rows=[]
    for k,g in enumerate(GAMMA):
        p2={};values={}
        for a in ['j','0']:
            p2[a]=static_price(BETA+(1 if a=='0' else -1)*g,ALPHA,COST)
            values[a]=(p2[a]-COST)*float(shares(p2[a],a,g))
        effective=COST-DELTA*(values['j']-values['0'])
        p=static_price(BETA+(1 if h=='0' else -1)*g,ALPHA,effective);s=float(shares(p,h,g))
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


def persistent_revenue(policy, h, prior):
    """Evaluate a fixed-belief policy with the SAME type in both periods.

    Policy selection retains fixed beliefs; evaluation accounts for the sorting
    of persistent types into first-period purchase outcomes.
    """
    s1 = shares(policy['expected_p1'], h)
    s2j = shares(policy['p2j'], 'j')
    s20 = shares(policy['p20'], '0')
    by_type = (s1*(policy['expected_p1']-COST)
               + DELTA*(s1*s2j*(policy['p2j']-COST)
                        + (1-s1)*s20*(policy['p20']-COST)))
    return float(np.asarray(prior) @ by_type)


def solve(prior,h,validate=True):
    prior=np.asarray(prior,dtype=float)
    assert np.isclose(prior.sum(),1) and (prior>=0).all()
    p,value=maximize(lambda p:evaluate(p,h,prior))
    unknown=evaluate(p,h,prior,True);full=known(h,prior)
    unknown['fixed_belief_objective'] = value
    unknown['persistent_type_revenue'] = persistent_revenue(unknown, h, prior)
    full['persistent_type_revenue'] = full['profit']
    assert full['profit']+1e-8 >= unknown['persistent_type_revenue']
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
    return dict(prior=prior.tolist(),h0=h,known=full,unknown=unknown,value_of_information=full['profit']-unknown['persistent_type_revenue'])


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output
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
    (OUT/'settings.json').write_text(json.dumps(dict(gamma=GAMMA.tolist(),beta=BETA,alpha=ALPHA,delta=DELTA,c=COST,priors=priors,revenue_evaluation='same persistent types under both policies; common profit field remains fixed-belief objective',checks='Wider first-price grids; independent terminal maximizers; first-price finite-difference FOCs; Fixed-belief invariance; full-information bound; degenerate priors'),indent=2))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from pricing_presentation import information_figures
    import pandas as pd
    table = pd.DataFrame([dict(scenario=r['scenario'], h0=r['h0'], case=case,
        expected_p1=r[case]['expected_p1'], expected_p2=r[case]['expected_p2'],
        persistent_type_revenue=r[case]['persistent_type_revenue'], value_of_information=r['value_of_information'])
        for r in results for case in ['known', 'unknown']])
    table.to_csv(OUT/'information_friction_aggregate.csv', index=False)
    type_rows = []
    for r in results:
        for label, row in zip(['low', 'medium', 'high'], r['known']['types']):
            type_rows.append(dict(scenario=r['scenario'], h0=r['h0'], type=label,
                gamma=row['gamma'], known_p1=row['p1'],
                known_expected_p2=row['s1']*row['p2j']+(1-row['s1'])*row['p20']))
    pd.DataFrame(type_rows).to_csv(OUT/'information_friction_by_group.csv', index=False)
    information_figures(table, OUT, DELTA)
    # Existing slide filenames remain valid; all price figures show p1 -> p2.
    import shutil
    for name in ['profits', 'first_prices', 'terminal_prices', 'information_value']:
        for extension in ['pdf', 'png']:
            shutil.copyfile(OUT/f'information_friction_{name}.{extension}', OUT/f'{name}.{extension}')
    lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Pricing with known and uncertain switching types}',r'\label{tab:gamma-uncertainty}',r'\begin{tabular}{llrrr}\toprule',r'Initial state & Prior & Personalized pricing & Common pricing & Information value\\\midrule']
    for r in results:lines.append(f"${r['h0']}$ & {r['scenario']} & {r['known']['profit']:.4f} & {r['unknown']['persistent_type_revenue']:.4f} & {r['value_of_information']:.4f} \\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(OUT/'results.tex').write_text('\n'.join(lines)+'\n')
    print('All checks passed; wrote four figures, aggregate theoretical solutions, settings, and result table.')

if __name__=='__main__':main()
