"""Two-period logit toy model: fixed regular prices and a targeted discount.

Effective price = regular price - discount. Only good 2 can be discounted,
only in period two, and only after a purchase of good 1. Consumers start with
no prior purchase; period-one outside choices exit. Type probabilities are
fixed, with no learning. All comparisons hold regular prices/costs fixed.
Run .venv/bin/python toy_model.py to regenerate the active results and figures.
The former unrestricted-price model is retained in updates/toy_model/archive_unrestricted/.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

beta_0 = np.array([2.2, 2.0])
gamma_H, gamma_L = 2.0, 0.1
alpha, delta = -1.5, 0.95
costs = np.array([1.2, 0.2])
regular_prices = np.array([2.0, 2.0])
PRICE_MIN = 1.0
DISCOUNT_MAX = float(regular_prices[1] - PRICE_MIN)
J = 2


def logit_probs(utility):
    full = np.r_[0.0, utility]
    e = np.exp(full - full.max())
    probs = e / e.sum()
    return probs[1:], float(probs[0])


def effective_prices(discount=0.0):
    if not np.isfinite(discount) or not 0 <= discount <= DISCOUNT_MAX:
        raise ValueError(f'Discount must be in [0, {DISCOUNT_MAX}]')
    return regular_prices - np.array([0.0, discount])


def shares(discount, gamma, history):
    # No coupon before the first purchase or after already buying good 2.
    prices = effective_prices(discount if history == 0 else 0.0)
    utility = beta_0 + alpha * prices
    if history is not None:
        utility = utility + gamma * (np.arange(J) != history)
    return logit_probs(utility)


def terminal_profit(discount, gamma, history=0):
    prices = effective_prices(discount if history == 0 else 0.0)
    s, _ = shares(discount, gamma, history)
    return float(s @ (prices - costs))


def terminal_discount_gradient(discount, gamma):
    """Discount cost plus outside demand and higher-margin diversion effects."""
    s, s0 = shares(discount, gamma, 0)
    m = effective_prices(discount) - costs
    return float(s[1] * (-1 - alpha * (s0*m[1] + s[0]*(m[1]-m[0]))))


def evaluate_discount(discount, gammas):
    """Expected two-period profit per initial consumer, conditional on true type."""
    first, _ = logit_probs(beta_0 + alpha * regular_prices)
    first_profit = first @ (regular_prices - costs)
    return np.array([
        first_profit + delta * sum(first[h] * terminal_profit(discount, g, h)
                                   for h in range(J))
        for g in gammas
    ])


@dataclass
class Policy:
    discount: float
    profit: float
    type_profits: np.ndarray

    @property
    def p1(self):
        return regular_prices.copy()

    @property
    def p2(self):
        return np.array([effective_prices(self.discount), regular_prices.copy()])


def solve_common(gammas, weights):
    gammas, weights = np.asarray(gammas), np.asarray(weights, dtype=float)
    if (gammas.ndim != 1 or weights.shape != gammas.shape or len(gammas) == 0
            or not np.all(np.isfinite(gammas)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError('Finite type values and nonnegative weights summing to one required')

    def gradient(d):
        return sum(w*terminal_discount_gradient(d, g) for g, w in zip(gammas, weights))

    # Check both boundaries and each stationary point located on a fine grid.
    # These baseline profit derivatives have a unique zero when interior.
    grid = np.linspace(0, DISCOUNT_MAX, 129)
    grads = [gradient(d) for d in grid]
    candidates = [0., DISCOUNT_MAX]
    for a, b, ga, gb in zip(grid[:-1], grid[1:], grads[:-1], grads[1:]):
        if ga == 0:
            candidates.append(a)
        if ga*gb < 0:
            candidates.append(brentq(gradient, a, b, xtol=1e-13))
    d = max(candidates, key=lambda d: weights @ evaluate_discount(d, gammas))
    values = evaluate_discount(d, gammas)
    return Policy(float(d), float(weights @ values), values)


def fixed_menu_regions(values):
    """Upper-envelope regions for fixed discounts; columns are true H and L."""
    values = np.asarray(values)
    intercepts = values[:, 1]
    slopes = values[:, 0] - intercepts
    breaks = [0., 1.]
    for a in range(len(values)):
        for b in range(a):
            diff = slopes[a] - slopes[b]
            if abs(diff) > 1e-12:
                q = (intercepts[b]-intercepts[a])/diff
                if 0 < q < 1:
                    breaks.append(float(q))
    breaks = sorted(set(breaks))
    regions = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        k = int(np.argmax(intercepts+(a+b)/2*slopes))
        if regions and regions[-1][2] == k:
            regions[-1] = (regions[-1][0], b, k)
        else:
            regions.append((a, b, k))
    return regions


def run_analysis(output, points=101, gamma_max=3.0):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    types = [gamma_H, gamma_L]
    personal = [solve_common([g], [1.]) for g in types]
    common50 = solve_common(types, [.5, .5])
    labels = ['high', 'low', 'uniform_50_50']
    policies = [*personal, common50]
    fixed_values = np.array([evaluate_discount(p.discount, types) for p in policies])
    pd.DataFrame([dict(menu=n, discount=p.discount, profit_true_H=v[0], profit_true_L=v[1])
                  for n, p, v in zip(labels, policies, fixed_values)]).to_csv(out/'fixed_menu_type_profits.csv', index=False)
    regions = fixed_menu_regions(fixed_values)
    pd.DataFrame([dict(probability_H_min=a, probability_H_max=b, best_menu=labels[k])
                  for a,b,k in regions]).to_csv(out/'fixed_menu_regions.csv', index=False)
    rows=[]
    for q in np.unique(np.r_[np.linspace(0,1,points), .5]):
        p=solve_common(types,[q,1-q])
        perfect=q*personal[0].profit+(1-q)*personal[1].profit
        rows.append(dict(high_share=q, discount=p.discount, common_profit=p.profit,
                         targeted_profit=perfect,targeting_gain=perfect-p.profit,
                         common_profit_H=p.type_profits[0],common_profit_L=p.type_profits[1],
                         effective_good2_after_good1=regular_prices[1]-p.discount))
    mix=pd.DataFrame(rows); mix.to_csv(out/'population_mix.csv',index=False)
    q=np.unique(np.r_[np.linspace(0,1,1001),[r[0] for r in regions]])
    profits=q[:,None]*fixed_values[:,0]+(1-q[:,None])*fixed_values[:,1]
    frame=pd.DataFrame(profits,columns=[f'profit_{n}' for n in labels])
    frame.insert(0,'probability_H',q)
    frame['best_menu']=np.array(labels)[profits.argmax(axis=1)]
    frame.to_csv(out/'fixed_menu_comparison.csv',index=False)
    prices=[]
    for name,p in [('uniform_50_50',common50),('targeted_H',personal[0]),('targeted_L',personal[1])]:
        for period,history,pr in [(1,'no prior product',p.p1),(2,'good1',p.p2[0]),(2,'good2',p.p2[1])]:
            prices.append(dict(policy=name,period=period,history=history,
                               price_good1=pr[0],price_good2=pr[1],
                               margin_good1=pr[0]-costs[0],margin_good2=pr[1]-costs[1]))
    pd.DataFrame(prices).to_csv(out/'prices_50_50.csv',index=False)
    gamma_rows=[]; fixed_gamma=[]
    for g in np.unique(np.r_[np.linspace(0,gamma_max,points),types]):
        p=solve_common([g],[1.])
        fixed_value=evaluate_discount(common50.discount,[g])[0]
        gamma_rows.append(dict(gamma=g,discount=p.discount,targeted_profit=p.profit,
                               uniform_50_50_profit=fixed_value,gain_for_type=p.profit-fixed_value))
        fixed_gamma.append(dict(gamma=g,**{f'profit_{n}':evaluate_discount(p.discount,[g])[0]
                                          for n,p in zip(labels,policies)}))
    pd.DataFrame(gamma_rows).to_csv(out/'profit_by_gamma.csv',index=False)
    pd.DataFrame(fixed_gamma).to_csv(out/'fixed_menu_profit_by_gamma.csv',index=False)
    switches=[]
    for name,p in [('regular',Policy(0,0,np.array([]))),('H',personal[0]),('L',personal[1]),('common_50_50',common50)]:
        for type_name,g in zip(['H','L'],types):
            s,s0=shares(p.discount,g,0)
            switches.append(dict(policy=name,true_type=type_name,discount=p.discount,
                                 probability_good1=s[0],probability_switch_good2=s[1],
                                 probability_outside=s0,conditional_switch=s[1]/sum(s),
                                 terminal_profit=terminal_profit(p.discount,g)))
    pd.DataFrame(switches).to_csv(out/'switching_outcomes.csv',index=False)
    regular_values=evaluate_discount(0,types)
    low_values=evaluate_discount(personal[1].discount,types)
    gains=low_values-regular_values
    cutoff=-gains[0]/(gains[1]-gains[0])
    discount_rows=[]
    for u in np.linspace(0,1,points):
        p=solve_common(types,[1-u,u])
        r=np.array([1-u,u])@regular_values; d=np.array([1-u,u])@low_values
        discount_rows.append(dict(probability_L=u,regular_profit=r,discount_profit=d,
                                  discount_gain=d-r,reoptimized_common_profit=p.profit))
    pd.DataFrame(discount_rows).to_csv(out/'fixed_discount_comparison.csv',index=False)
    first,s0=logit_probs(beta_0+alpha*regular_prices)
    parameters=dict(regular_price_good1=regular_prices[0],regular_price_good2=regular_prices[1],
                    cost_good1=costs[0],cost_good2=costs[1],gamma_H=gamma_H,gamma_L=gamma_L,
                    alpha=alpha,delta=delta,discount_max=DISCOUNT_MAX,first_share_good1=first[0],
                    first_share_good2=first[1],first_share_outside=s0,discount_H=personal[0].discount,
                    discount_L=personal[1].discount,discount_common50=common50.discount,
                    fixed_discount_cutoff_L=cutoff,delta_H=gains[0],delta_L=gains[1])
    pd.DataFrame([parameters]).to_csv(out/'parameters.csv',index=False)
    lines=['# Fixed regular prices and targeted discounts', '',
           'Effective price = regular price - discount. Only good 2 can be discounted, in period two after buying good 1. All other prices remain at the fixed regular price.', '',
           'Illustrative regular prices (2.00, 2.00), costs (1.20, 0.20), price coefficient -1.5, gamma_H=2, gamma_L=0.1, delta=0.95. The cost gap and fixed-price restriction replace the previous unrestricted-price calibration. Regular prices are maintained assumptions, not claimed to be unrestricted monopoly optima.', '',
           'No prior purchase, myopic logit consumers, permanent exit after period-one outside. First purchases do not reveal type. Known type probabilities; no Bayesian learning or classifier. Profits are per initial consumer.', '',
           f'H discount: {personal[0].discount:.8f}; L discount: {personal[1].discount:.8f}; 50/50 common discount: {common50.discount:.8f}.', '',
           f'50/50 common profit: {common50.profit:.8f}; observed-type profit: {(personal[0].profit+personal[1].profit)/2:.8f}; information gain: {mix.loc[np.isclose(mix.high_share,.5),"targeting_gain"].iloc[0]:.8f}.', '',
           f'Offering the fixed L-optimal coupon instead of no coupon breaks even at Pr(L)={cutoff:.8f}. Continuous discount optimization is a different problem.', '',
           'Fixed discount regions (q=Pr(H)):']
    lines += [f'- {a:.8f} < q < {b:.8f}: {labels[k]} discount.' for a,b,k in regions]
    lines += ['', 'The discount increases demand for good 2 and reduces its unit margin. In this calibration the L-optimal discount leaves good 2 with a larger margin than good 1. H has zero optimal discount; logit switching is nevertheless probabilistic, not guaranteed.', '',
              'Only the good-1 history is eligible for a coupon. This isolates switching toward good 2; after good 2 a coupon would instead induce retention. Population profits include both first-period purchase histories.', '',
              'Solver checks boundaries and all stationary roots bracketed by a 129-point derivative grid. Tests compare solutions with independent bounded maximization and dense feasible grids. Concavity of conditional mispricing profits is not imposed.', '']
    (out/'results.md').write_text('\n'.join(lines))
    (out/'fixed_menu_results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return mix


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='updates/toy_model/results')
    parser.add_argument('--points',type=int,default=101)
    parser.add_argument('--gamma-max',type=float,default=3.)
    args=parser.parse_args()
    if args.points<3 or args.gamma_max<0:
        parser.error('Need points >=3 and gamma-max >=0')
    run_analysis(args.output_dir,args.points,args.gamma_max)
