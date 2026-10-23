"""Free two-period monopoly prices with observed random initial states.
Myopic logit consumers; period-one outside choices exit permanently.
Menus commit before choices, depend on full observed history, and incorporate
endogenous type selection without adding a signal or a learning intervention.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq

beta_0=np.array([2.2,2.0])
gamma_H,gamma_L=2.,1.
alpha,delta=-1.5,.95
costs=np.array([1.2,.8])
STATES=(0,1,None)  # products j, k, no previous product
INITIAL_WEIGHTS=np.ones(3)/3


def logit_probs(v):
    v=np.r_[0.,v]; e=np.exp(v-v.max()); s=e/e.sum()
    return s[1:],float(s[0])


def shares_jac(p,gamma,state):
    v=beta_0+alpha*np.asarray(p)
    if state is not None:v=v+gamma*(np.arange(2)!=state)
    s,_=logit_probs(v)
    return s,alpha*(np.diag(s)-np.outer(s,s))


def state_objective(flat,gammas,weights,state):
    p1=flat[:2];p2=flat[2:].reshape(2,2)
    value=0.;g1=np.zeros(2);g2=np.zeros((2,2))
    for gamma,weight in zip(gammas,weights):
        s1,j1=shares_jac(p1,gamma,state); w=np.zeros(2)
        for h in range(2):
            s2,j2=shares_jac(p2[h],gamma,h);m=p2[h]-costs
            w[h]=s2@m
            g2[h]+=delta*weight*s1[h]*(s2+j2.T@m)
        effective=p1-costs+delta*w
        value+=weight*(s1@effective)
        g1+=weight*(s1+j1.T@effective)
    return float(value),np.r_[g1,g2.ravel()]


@dataclass
class Policy:
    p1:np.ndarray  # initial state, product
    p2:np.ndarray  # initial state, first purchase, product
    profit:float
    type_profits:np.ndarray


def components(p1,p2,gammas):
    rows=[]
    for gamma in gammas:
        first=second=survival=0.
        for a,state in enumerate(STATES):
            s1,_=shares_jac(p1[a],gamma,state)
            w=np.array([shares_jac(p2[a,h],gamma,h)[0]@(p2[a,h]-costs) for h in range(2)])
            first+=INITIAL_WEIGHTS[a]*(s1@(p1[a]-costs))
            second+=INITIAL_WEIGHTS[a]*(s1@w)
            survival+=INITIAL_WEIGHTS[a]*s1.sum()
        rows.append([first,second,first+delta*second,survival])
    return np.asarray(rows)


def evaluate_policy(p1,p2,gammas):return components(p1,p2,gammas)[:,2]


def solve_common(gammas,weights):
    gammas=np.asarray(gammas);weights=np.asarray(weights,dtype=float)
    if (gammas.ndim!=1 or not len(gammas) or gammas.shape!=weights.shape
        or not np.all(np.isfinite(gammas)) or not np.all(np.isfinite(weights))
        or np.any(weights<0) or not np.isclose(weights.sum(),1)):
        raise ValueError('Finite gammas and nonnegative weights summing to one required')
    active=weights>0
    if np.count_nonzero(active)==1:
        # Backward solution for a single type: common effective margin M.
        gamma=float(gammas[active][0]);a=-alpha
        def one_period(effective_cost,state):
            utility=beta_0+alpha*effective_cost
            if state is not None:utility=utility+gamma*(np.arange(2)!=state)
            # a*M = 1 + sum exp(utility-a*M), using log-sum-exp stability.
            def equation(m):
                terms=np.r_[0.,utility-a*m];top=terms.max()
                return np.log(a*m)-(top+np.log(np.exp(terms-top).sum()))
            upper=max(2.,float(np.max(utility)/a+2))
            while equation(upper)<0:upper*=2
            margin=brentq(equation,1/a,upper,xtol=1e-13)
            prices=effective_cost+margin
            return prices,float(shares_jac(prices,gamma,state)[0]@(prices-costs))
        terminal=np.array([one_period(costs,h)[0] for h in range(2)])
        w=np.array([shares_jac(terminal[h],gamma,h)[0]@(terminal[h]-costs) for h in range(2)])
        p1=np.array([one_period(costs-delta*w,state)[0] for state in STATES])
        p2=np.tile(terminal,(3,1,1));values=evaluate_policy(p1,p2,gammas)
        return Policy(p1,p2,float(weights@values),values)
    solutions=[]
    for state in STATES:
        candidates=[]
        for markup in (.3,1.2,3.):
            result=minimize(lambda x:-state_objective(x,gammas,weights,state)[0],
                np.tile(costs+markup,3),jac=lambda x:-state_objective(x,gammas,weights,state)[1],
                method='BFGS',options={'gtol':1e-8,'maxiter':600})
            if np.isfinite(result.fun) and np.max(np.abs(result.jac))<2e-6:candidates.append(result)
        if not candidates:raise RuntimeError(f'No converged price solution: {state}')
        solutions.append(min(candidates,key=lambda r:r.fun).x)
    p1=np.array([x[:2] for x in solutions]);p2=np.array([x[2:].reshape(2,2) for x in solutions])
    values=evaluate_policy(p1,p2,gammas)
    return Policy(p1,p2,float(weights@values),values)


def fixed_menu_regions(values):
    values=np.asarray(values); intercept=values[:,1];slope=values[:,0]-intercept
    breaks=[0.,1.]
    for i in range(len(values)):
        for j in range(i):
            if abs(slope[i]-slope[j])>1e-12:
                q=(intercept[j]-intercept[i])/(slope[i]-slope[j])
                if 0<q<1:breaks.append(float(q))
    regions=[];breaks=sorted(set(breaks))
    for a,b in zip(breaks[:-1],breaks[1:]):
        k=int(np.argmax(intercept+(a+b)/2*slope))
        if regions and regions[-1][2]==k:regions[-1]=(regions[-1][0],b,k)
        else:regions.append((a,b,k))
    return regions


def run_analysis(output,points=61,gamma_max=10.,gamma_min=1.5,gamma_step=.5):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    types=[gamma_H,gamma_L];personal=[solve_common([g],[1]) for g in types]
    common=solve_common(types,[.5,.5]);policies=[*personal,common];labels=['high','low','uniform_50_50']
    mix=[]
    for q in np.unique(np.r_[np.linspace(0,1,points),.5]):
        p=solve_common(types,[q,1-q]);obs=q*personal[0].profit+(1-q)*personal[1].profit
        mix.append(dict(high_share=q,common_profit=p.profit,targeted_profit=obs,targeting_gain=obs-p.profit,
            common_profit_H=p.type_profits[0],common_profit_L=p.type_profits[1]))
    mix=pd.DataFrame(mix);mix.to_csv(out/'population_mix.csv',index=False)
    values=np.array([evaluate_policy(p.p1,p.p2,types) for p in policies]);regions=fixed_menu_regions(values)
    pd.DataFrame([dict(probability_H_min=a,probability_H_max=b,best_menu=labels[k]) for a,b,k in regions]).to_csv(out/'fixed_menu_regions.csv',index=False)
    q=np.linspace(0,1,1001);v=q[:,None]*values[:,0]+(1-q[:,None])*values[:,1]
    fixed=pd.DataFrame(v,columns=[f'profit_{n}' for n in labels]);fixed.insert(0,'probability_H',q)
    fixed.to_csv(out/'fixed_menu_comparison.csv',index=False)
    prices=[]
    for name,p in zip(['targeted_H','targeted_L','uniform_50_50'],policies):
        for a,state in enumerate(['j','k','0']):
            for t,h,pr in [(1,state,p.p1[a])]+[(2,f'{state} to {h}',p.p2[a,b]) for b,h in enumerate(['j','k'])]:
                prices.append(dict(policy=name,period=t,history=h,price_good1=pr[0],price_good2=pr[1],margin_good1=pr[0]-costs[0],margin_good2=pr[1]-costs[1]))
    pd.DataFrame(prices).to_csv(out/'prices_50_50.csv',index=False)
    sweep=[];by_gamma=[];distribution=[]
    for g in np.arange(gamma_min,gamma_max+gamma_step/2,gamma_step):
        pH=solve_common([g],[1]);p=solve_common([g,gamma_L],[.5,.5])
        c=components(p.p1,p.p2,[g,gamma_L]);known=components(pH.p1,pH.p2,[g])[0]
        row=dict(gamma_H=g,gamma_L=gamma_L)
        for i,kind in enumerate(['H','L']):
            for j,key in enumerate(['period1','period2','total','survival']):row[f'{kind}_{key}']=c[i,j]
            row[f'{kind}_period2_survivor']=c[i,1]/c[i,3]
            distribution.append(dict(gamma_H=g,true_type=kind,probability=.5,value=c[i,2],period2_value=delta*c[i,1]))
        row.update(known_H_total=known[2],known_L_total=personal[1].profit,
                   known_H_period1=known[0],known_H_period2=known[1])
        # A clean induction benchmark: optimize one-period prices at the same arriving state.
        static=0.
        for a,state in enumerate(STATES):
            def objective(pr):
                s,j=shares_jac(pr,g,state);m=pr-costs
                return -(s@m),-(s+j.T@m)
            r=minimize(objective,costs+1,jac=True,method='BFGS',options={'gtol':1e-9})
            static+=INITIAL_WEIGHTS[a]*(-r.fun)
        row['known_H_static_period1']=static
        row['known_H_induction_sacrifice']=static-known[0]
        sweep.append(row)
        by_gamma.append(dict(gamma=g,targeted_profit=pH.profit,uniform_50_50_profit=evaluate_policy(common.p1,common.p2,[g])[0],**{f'profit_{n}':evaluate_policy(pol.p1,pol.p2,[g])[0] for n,pol in zip(labels,policies)}))
    pd.DataFrame(sweep).to_csv(out/'gamma_H_sweep.csv',index=False)
    pd.DataFrame(distribution).to_csv(out/'value_distribution.csv',index=False)
    pd.DataFrame(by_gamma).to_csv(out/'profit_by_gamma.csv',index=False)
    pd.DataFrame(by_gamma).to_csv(out/'fixed_menu_profit_by_gamma.csv',index=False)
    middle=mix.iloc[(mix.high_share-.5).abs().argmin()]
    report=f'''# Free prices with random initial histories

Initial state is j, k, or 0, each with probability 1/3 independently of type. Initial 0 means no prior product and no switching bonus. Initial state is observed. Period-one outside choices exit permanently.
Prices for both goods in both periods are freely optimized, with no coupon restriction, price floor, or price ceiling. Period-two menus can depend on both initial state and period-one purchase. Consumers are myopic. All menus are committed before choices; endogenous selection of types into purchase histories enters the objective. First purchases can now be informative about type.

Parameters: beta=(2.2,2), costs=(1.2,0.8), alpha=-1.5, delta=.95. Good 2 costs 0.8 versus 1.2 for good 1; both prices remain freely optimized.
50/50 common profit: {middle.common_profit:.8f}; observed-type profit: {middle.targeted_profit:.8f}; information gain: {middle.targeting_gain:.8f}.

The gamma_H sweep uses gamma_H={gamma_min:g} to {gamma_max:g} in steps of {gamma_step:g}, holds gamma_L={gamma_L:g} and Pr(H)=0.5 fixed and reoptimizes all common prices for every gamma_H. It reports each true type's period-one profit, undiscounted period-two profit per arriving consumer, total discounted value, survival probability, and period-two profit per survivor. The known-type comparison separately optimizes each type. Induction sacrifice compares known-type dynamic period-one profit with its optimized one-period benchmark; it is not a causal effect of a coupon.

The value distribution has two atoms with masses 0.5, rather than a continuous density. value_distribution.csv records these atoms. A continuous value density would require a continuous distribution over gamma. The distribution is of expected customer values, not realized transaction profits.

Known-type prices use backward interior equations. Heterogeneous common-price optimization uses analytic gradients and three unconstrained BFGS starts per initial state, accepting stationary solutions with maximum gradient below 2e-6. Multistart checks provide numerical evidence, not a global theorem. With known type, interior terminal margins are equal across products at each history; a higher-margin destination is no longer maintained.
'''
    (out/'results.md').write_text(report);print(report)
    return mix

if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output-dir',default='updates/toy_model/results')
    parser.add_argument('--points',type=int,default=61);parser.add_argument('--gamma-max',type=float,default=10.)
    parser.add_argument('--gamma-min',type=float,default=1.5);parser.add_argument('--gamma-step',type=float,default=.5)
    args=parser.parse_args()
    if (args.points<3 or not np.all(np.isfinite([args.gamma_min,args.gamma_max,args.gamma_step]))
        or args.gamma_step<=0 or args.gamma_min<=gamma_L or args.gamma_max<args.gamma_min
        or not np.isclose((args.gamma_max-args.gamma_min)/args.gamma_step,round((args.gamma_max-args.gamma_min)/args.gamma_step))):
        parser.error('Need points >=3, gamma-min > gamma_L, positive step, and endpoints on the grid')
    run_analysis(args.output_dir,args.points,args.gamma_max,args.gamma_min,args.gamma_step)
