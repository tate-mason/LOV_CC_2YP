"""Theoretical supply. Edit Parameters, then run python3 supply/supply_simulation.py.

One product versus outside; fixed known type. Solve p2(h1) first, then p1(h0).
Includes known state, hidden initial state, and ex ante price commitment.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys
import tempfile
from pprint import pformat

import numpy as np
import pandas as pd

# Use a writable plotting cache, including on a shared research server.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "final_simple_matplotlib"))

import math
import hashlib
import csv
from scipy.special import expit as scipy_expit
from scipy.optimize import brentq
import sympy as sp
# Support both running this file directly and importing it for verification.
if __package__:
    from . import supply_sympy as theory
else:
    import supply_sympy as theory



# === Parameters: edit these values === #

output = Path(__file__).resolve().parents[1] / "results" / "supply"
beta = 2.0
alpha = -1.0                         # Signed price coefficient: must be negative.
delta = 1.0                         # Preserve the theory's existing calibration.
cost = 0.0
gamma_min = -3.0
gamma_max = 3.0
gamma_points = 61
priors = [0.25, 0.5, 0.75]           # Probability the observed initial choice was inside.
# Additional figures from the symbolic derivatives (the original 601-point grid).
derivative_gamma_min = -3.0
derivative_gamma_max = 3.0
derivative_gamma_points = 601
skip_derivatives = False             # True runs only the three supply cases.
from_solutions = False               # True reads your saved derivative solutions.
no_appendix = False                 # True skips the benchmark-only TeX appendix.
run_self_tests = False                    # True runs theoretical checks, not the full grid.
CASES = ["known_state", "hidden_initial_state", "ex_ante_commitment"]
CASE_LABELS = dict(zip(CASES, ["Known state (base)", "Hidden initial state", "Ex ante commitment"]))
STATE_LABELS = {"j": r"Initially inside ($x_0=j$)", "0": r"Initially outside ($x_0=0$)"}
STATES = [("0", "0"), ("0", "j"), ("j", "j"), ("j", "0")]
PARAMETERS = dict(beta=beta, alpha=alpha, delta=delta, c=cost)
GAMMAS = np.linspace(derivative_gamma_min, derivative_gamma_max, derivative_gamma_points)
OUT = output / "derivatives" / "figures"





# === 1. Choice probabilities and numerical maximization === #



def expit(x):
    x = np.asarray(x)
    small = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+small), small/(1+small))

def golden_maximum(function, lo, hi):
    ratio = (math.sqrt(5)-1)/2
    left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fl, fr = function(left), function(right)
    for _ in range(100):
        if hi-lo <= 1e-10*(1+abs(left)+abs(right)):
            break
        if fl > fr:
            hi, right, fr = right, left, fl
            left = hi-ratio*(hi-lo)
            fl = function(left)
        else:
            lo, left, fl = left, right, fr
            right = lo+ratio*(hi-lo)
            fr = function(right)
    return (left, fl) if fl > fr else (right, fr)

def scalar_maximum(function, bounds, points=161):
    """Search every grid-local maximum, including boundaries (mixtures may multimode)."""
    grid = np.linspace(*bounds, points)
    # Supply objectives support array evaluation; profiles are scalar functions.
    try:
        values = np.asarray(function(grid))
        if values.shape != grid.shape:
            raise ValueError("Scalar objective")
    except (ValueError, TypeError):
        values = np.array([function(x) for x in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, points - 1):
        if values[i] >= max(values[i - 1], values[i + 1]) and values[i] > min(values[i - 1], values[i + 1]):
            price, value = golden_maximum(function, grid[i - 1], grid[i + 1])
            candidates.append((value, price))
    value, price = max(candidates)
    return float(price), float(value)

# Solve the one-period logit pricing FOC. The dynamic problem uses the same equation with
# a continuation-adjusted effective cost.
def static_price(intercept, alpha, effective_cost=0.):
    """Signed alpha<0; monotone transformed FOC, including introductory subsidies."""
    def foc(p):
        return alpha*(p-effective_cost)+1+np.exp(np.clip(intercept+alpha*p, -745, 700))
    lo, hi = effective_cost, effective_cost+max(2., 2/abs(alpha))
    while foc(hi) > 0:
        hi = lo+2*(hi-lo)
    for _ in range(100):
        mid = (lo+hi)/2
        if foc(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2



# === 2. Backward induction and the three supply cases === #



# First solve terminal prices. Then solve initial prices for known history, hidden initial
# history, and commitment to both prices.
def supply_solution(gamma, prior=.5, beta=2., alpha=-1., delta=1.0, cost=0.):
    """Personalized pricing; hidden h0, then either observe h1 or commit to p1,p2."""
    if not (np.isfinite([gamma, prior, beta, alpha, delta, cost]).all()
            and alpha < 0 and 0 <= prior <= 1 and 0 <= delta <= 1):
        raise ValueError("Need finite inputs, alpha<0, prior/delta in [0,1].")
    # Index 0 is previously inside j; index 1 is previously outside 0.
    intercepts = beta + np.array([-gamma, gamma])
    weights = np.array([prior, 1 - prior])
    p2 = np.array([static_price(x, alpha, cost) for x in intercepts])
    s2 = expit(intercepts + alpha * p2)
    W2 = (p2 - cost) * s2
    adjustment = delta * (W2[0] - W2[1])

    def share1(p):
        return expit(intercepts + alpha * np.asarray(p)[..., None]) @ weights

    def adaptive_value(p):
        s = share1(p)
        return s * (p - cost) + delta * (s * W2[0] + (1 - s) * W2[1])

    scale = (20 + abs(beta) + abs(gamma) + abs(alpha * cost)) / abs(alpha)
    bounds = (cost - scale, cost + scale)
    pooled_p1, adaptive = scalar_maximum(adaptive_value, bounds)
    # Rechecking a wider domain catches a binding numerical search boundary.
    wide_p1, wide_value = scalar_maximum(adaptive_value, (cost - 2*scale, cost + 2*scale), 321)
    if wide_value > adaptive + 1e-8:
        pooled_p1, adaptive = wide_p1, wide_value
    known_p1 = np.array([static_price(x, alpha, cost - adjustment) for x in intercepts])
    known_values = []
    for x, price in zip(intercepts, known_p1):
        s = expit(x + alpha * price)
        known_values.append(s*(price-cost) + delta*(s*W2[0]+(1-s)*W2[1]))

    def committed_profile(second, return_price=False):
        terminal = expit(intercepts + alpha * second)
        Wj, W0 = terminal*(second-cost)
        def objective(first):
            s = share1(first)
            return s*(first-cost)+delta*(s*Wj+(1-s)*W0)
        price, value = scalar_maximum(objective, (cost-2*scale, cost+2*scale))
        return (price, value) if return_price else value

    # Profile the joint problem, optimizing p1 anew at EACH candidate p2.
    # Refining every grid-local mode avoids assuming a globally unique FOC.
    if delta == 0:
        cp1, cp2, committed = pooled_p1, static_price(beta, alpha, cost), adaptive
    else:
        cp2, committed = scalar_maximum(committed_profile, (cost, cost+scale))
        cp1, committed = committed_profile(cp2, True)
        # Independent wider and denser profile scan.
        check_p2, check_value = scalar_maximum(committed_profile, (cost, cost+2*scale), 321)
        if check_value > committed+1e-9:
            cp2 = check_p2
            cp1, committed = committed_profile(cp2, True)
        if min(cp2-cost, cost+2*scale-cp2) < scale*1e-5:
            raise RuntimeError("Committed optimum is near a search bound; inspect calibration.")
    known = float(weights @ known_values)
    if not known + 1e-7 >= adaptive >= committed - 1e-7:
        raise AssertionError("Information-value ordering failed.")
    s = share1(pooled_p1)
    derivative = float(weights @ (alpha*expit(intercepts+alpha*pooled_p1)
                                  * expit(-intercepts-alpha*pooled_p1)))
    committed_first_share = share1(cp1)
    committed_first_derivative = float(weights @ (alpha*expit(intercepts+alpha*cp1)*expit(-intercepts-alpha*cp1)))
    terminal = expit(intercepts+alpha*cp2)
    terminal_weights = np.array([committed_first_share, 1-committed_first_share])
    committed_foc1 = committed_first_share+committed_first_derivative*(cp1-cost+delta*(cp2-cost)*(terminal[0]-terminal[1]))
    committed_foc2 = float(terminal_weights @ (terminal+alpha*(cp2-cost)*terminal*(1-terminal))) if delta > 0 else 0.
    return dict(gamma=gamma, prior_inside=prior, beta=beta, alpha=alpha, delta=delta, cost=cost,
                known_p1_inside=known_p1[0], known_p1_outside=known_p1[1],
                known_value=known, hidden_p1=pooled_p1, adaptive_p2_inside=p2[0],
                adaptive_p2_outside=p2[1], hidden_value=adaptive,
                committed_p1=cp1, committed_p2=cp2, committed_value=committed,
                initial_information_value=known-adaptive,
                second_period_flexibility_value=adaptive-committed,
                committed_foc1=committed_foc1, committed_foc2=committed_foc2,
                hidden_foc=s+derivative*(pooled_p1-cost+adjustment))



# === 3. Revenue, profit, prices, and probabilities for every history === #



# Keep all four (h0,h1) paths. A path price is an offered price; expected revenue weights
# it by purchase probabilities.
def outcome_tables(result):
    """All four realized histories per case, plus correctly weighted ex ante objects."""
    g, beta, alpha, delta, cost, pi = [result[k] for k in ["gamma", "beta", "alpha", "delta", "cost", "prior_inside"]]
    paths, summary = [], []
    for case in CASES:
        case_paths = []
        for h0, weight in [("j", pi), ("0", 1-pi)]:
            if case == "known_state":
                p1 = result["known_p1_inside" if h0 == "j" else "known_p1_outside"]
            else:
                p1 = result["hidden_p1" if case == "hidden_initial_state" else "committed_p1"]
            s1 = float(expit(beta+(-1 if h0 == "j" else 1)*g+alpha*p1))
            conditional_value = 0.
            local = []
            for h1, transition in [("j", s1), ("0", 1-s1)]:
                p2 = result["committed_p2"] if case == "ex_ante_commitment" else result["adaptive_p2_inside" if h1 == "j" else "adaptive_p2_outside"]
                s2 = float(expit(beta+(-1 if h1 == "j" else 1)*g+alpha*p2))
                # Realize the first choice; integrate the terminal taste shock.
                conditional_profit = (p1-cost)*(h1 == "j")+delta*(p2-cost)*s2
                conditional_value += transition*conditional_profit
                row = dict(gamma=g, prior_inside=pi, case=case, h0=h0, h1=h1,
                           initial_probability=weight, transition_probability=transition,
                           path_probability=weight*transition, p1=p1, p2=p2,
                           price_ratio=p1/p2 if abs(p2) > 1e-12 else np.nan,
                           first_purchase_probability=s1, second_purchase_probability=s2,
                           conditional_profit=conditional_profit)
                local.append(row)
            for row in local:
                row["optimal_profit_given_h0"] = conditional_value
            case_paths.extend(local)
        probability = np.array([r["path_probability"] for r in case_paths])
        if not np.isclose(probability.sum(), 1.):
            raise AssertionError("Path probabilities do not sum to one.")
        mean_p1, mean_p2, profit = [float(probability @ np.array([r[k] for r in case_paths]))
                                    for k in ["p1", "p2", "conditional_profit"]]
        expected_ratio = float(probability @ np.array([r["price_ratio"] for r in case_paths]))
        expected_key = {"known_state": "known_value", "hidden_initial_state": "hidden_value", "ex_ante_commitment": "committed_value"}[case]
        if not np.isclose(profit, result[expected_key], atol=1e-8):
            raise AssertionError("Path profits do not reproduce optimized objective.")
        summary.append(dict(gamma=g, prior_inside=pi, case=case, expected_p1=mean_p1,
                            expected_p2=mean_p2, ratio_of_expected_prices=mean_p1/mean_p2 if abs(mean_p2) > 1e-12 else np.nan,
                            expected_price_ratio=expected_ratio, optimal_profit=profit))
        paths.extend(case_paths)
    return paths, summary



# === 4. Figures === #



def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def make_case_figures(paths, summary, output):
    plt = plotting()
    for prior in summary.prior_inside.unique():
        stem = f"prior_{prior:g}".replace(".", "p")
        subset = paths[paths.prior_inside == prior]
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        for column, case in enumerate(CASES):
            data = subset[subset.case == case]
            if case == "known_state":
                for h0 in ["0", "j"]:
                    initial = data[(data.h0 == h0) & (data.h1 == "j")]
                    axes[0, column].plot(initial.gamma, initial.p1, label=f"p1(h0={h0})")
            else:
                initial = data[(data.h0 == "0") & (data.h1 == "j")]
                axes[0, column].plot(initial.gamma, initial.p1, label="p1")
            if case == "ex_ante_commitment":
                axes[0, column].plot(initial.gamma, initial.p2, label="p2")
            else:
                for h1 in ["0", "j"]:
                    terminal = data[(data.h0 == "0") & (data.h1 == h1)]
                    axes[0, column].plot(terminal.gamma, terminal.p2, "--", label=f"p2(h1={h1})")
            for (h0, h1), history in data.groupby(["h0", "h1"]):
                # Duplicate ratios in restricted cases need only one curve.
                if case != "known_state" and h0 != "0":
                    continue
                if case == "ex_ante_commitment" and h1 != "j":
                    continue
                axes[1, column].plot(history.gamma, history.price_ratio, label=f"h0={h0}, h1={h1}")
            for h0 in ["0", "j"]:
                initial = data[(data.h0 == h0) & (data.h1 == "j")]
                axes[2, column].plot(initial.gamma, initial.optimal_profit_given_h0, label=f"E[profit|h0={h0}]")
            expected = summary[(summary.prior_inside == prior) & (summary.case == case)]
            axes[2, column].plot(expected.gamma, expected.optimal_profit, "k--", label="Ex ante E[profit]")
            axes[0, column].set_title(CASE_LABELS[case])
        for row, ylabel in enumerate(["Optimal prices", "Path-specific p1 / p2", "Expected two-period profit"]):
            for ax in axes[row]:
                ax.set(xlabel="gamma", ylabel=ylabel)
                ax.axhline(0, color="gray", linewidth=.5)
                ax.legend(fontsize=7)
        fig.suptitle(f"Supply cases: Pr(h0=j)={prior:g}")
        fig.tight_layout()
        fig.savefig(output / f"supply_cases_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_cases_{stem}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for case, group in summary[summary.prior_inside == prior].groupby("case", sort=False):
            axes[0].plot(group.gamma, group.optimal_profit, label=CASE_LABELS[case])
            axes[1].plot(group.gamma, group.ratio_of_expected_prices, label=CASE_LABELS[case])
            axes[2].plot(group.gamma, group.expected_price_ratio, label=CASE_LABELS[case])
        for ax, title in zip(axes, ["Optimal expected profit", "E[p1] / E[p2]", "E[p1 / p2]"]):
            ax.set(xlabel="gamma", title=title)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"supply_comparison_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_comparison_{stem}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, case in zip(axes, CASES):
            data = subset[subset.case == case]
            for (h0, h1), history in data.groupby(["h0", "h1"]):
                ax.plot(history.gamma, history.conditional_profit, label=f"h0={h0}, h1={h1}")
            ax.set(xlabel="gamma", ylabel="Path-conditional discounted profit", title=CASE_LABELS[case])
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"supply_history_profit_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_history_profit_{stem}.pdf")
        plt.close(fig)





# === 5. Evaluate the symbolic supply formulas numerically === #

_KEYS = ("dp1", "d2p1", "dprofit", "matched_j", "matched_0", "induce_harvest", "dp2j", "dp20")

_ARGS = (theory.alpha,theory.beta,theory.gamma,theory.delta,theory.c,theory.q,theory.sj1,theory.u,theory.v,theory.p1,theory.p2j,theory.p20,theory.Vj,theory.V0)

_EXPRESSIONS = (theory.dp1_dgamma,theory.d2p1_dgamma2,theory.dPi_dgamma,theory.matched_j,theory.matched_0,theory.induce_harvest,
                theory.price2_derivative(theory.gamma,theory.u,-1,theory.p2j),theory.price2_derivative(theory.gamma,theory.v,1,theory.p20))

_EVALUATE = sp.lambdify(_ARGS,_EXPRESSIONS,"numpy")

def inside_share(price,gamma,state,beta=2.,alpha=-1.):
    return float(scipy_expit(beta+theory.signed_state(state)*gamma+alpha*price))

def optimal_price(intercept,alpha,effective_cost):
    # Multiplying the FOC by alpha gives a strictly decreasing function.
    f = lambda p: alpha*(p-effective_cost)+1+np.exp(np.clip(intercept+alpha*p,-745,700))
    lo = effective_cost
    width = max(2.,2/abs(alpha))
    hi = lo+width
    while f(hi) > 0:
        width *= 2
        hi = lo+width
    return float(brentq(f,lo,hi,xtol=1e-12))

# Solve both terminal histories, solve the initial price, then evaluate the formulas from
# supply_sympy at those optimized prices.
def evaluate(gamma,initial_state,beta=2.,alpha=-1.,delta=1.0,c=0.):
    q = theory.signed_state(initial_state)
    if not (np.isfinite([gamma,beta,alpha,delta,c]).all() and alpha<0 and 0<=delta<=1):
        raise ValueError("Require finite parameters, alpha<0, and 0<=delta<=1 (delta=0 is the static counterfactual).")
    p2j,p20 = [optimal_price(beta+theory.signed_state(state)*gamma,alpha,c) for state in ["j","0"]]
    u,v = inside_share(p2j,gamma,"j",beta,alpha),inside_share(p20,gamma,"0",beta,alpha)
    Vj,V0 = (p2j-c)*u,(p20-c)*v
    DV = Vj-V0
    p1 = optimal_price(beta+q*gamma,alpha,c-delta*DV)
    s = inside_share(p1,gamma,initial_state,beta,alpha)
    profit = s*(p1-c)+delta*(s*Vj+(1-s)*V0)
    values = _EVALUATE(alpha,beta,gamma,delta,c,q,s,u,v,p1,p2j,p20,Vj,V0)
    result = dict(gamma=float(gamma),initial_state=initial_state,p1=p1,p2j=p2j,p20=p20,s=s,u=u,v=v,
                  V2j=Vj,V20=V0,DeltaV=DV,profit=profit,
                  static_component=c-1/(alpha*(1-s)),continuation_adjustment=-delta*DV,
                  static_counterfactual=optimal_price(beta+q*gamma,alpha,c))
    result.update(dict(zip(_KEYS,map(float,values))))
    result['matched_j' if initial_state=='0' else 'matched_0'] = float('nan')
    if initial_state=='j':
        result['induce_harvest'] = float('nan')
    result['d_matched_ratio'] = result['matched_j'] if initial_state=='j' else result['matched_0']
    result['d_induce_ratio'] = result['induce_harvest'] if initial_state=='0' else float('nan')
    result['foc1'] = s+alpha*s*(1-s)*(p1-c+delta*DV)
    result['foc2j'] = u+alpha*u*(1-u)*(p2j-c)
    result['foc20'] = v+alpha*v*(1-v)*(p20-c)
    return result

def solve(*, initial_state="0", **parameters):
    if 'z1' in parameters or 'z2' in parameters:
        raise ValueError("Old 0/1 histories are superseded: use initial_state='j' or '0'; both continuation branches are solved.")
    gamma = parameters.pop('gamma',0.)
    r = evaluate(gamma,initial_state,**parameters)
    return r['p1'],r['p2j'],r['p20'],r['s'],r['u'],r['v'],r['profit']

# === 6. Figures and all realized-history outcomes === #



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
        ax.legend()
        ax.grid(alpha=.2)
        if key=='realized_profit':
            fig.text(.5,.01,'Two-period profit by purchase history',
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


def make_derivative_figures():
    global plt
    plt = plotting()
    OUT.mkdir(parents=True,exist_ok=True)
    if from_solutions:
        with (OUT/'solutions.csv').open(newline='') as file:
            saved=[{k:v if k=='initial_state' else float(v) for k,v in row.items()}
                   for row in csv.DictReader(file)]
        curves={state:[r for r in saved if r['initial_state']==state] for state in ['j','0']}
        for group in curves.values():
            np.testing.assert_allclose([r['gamma'] for r in group],GAMMAS,atol=1e-12)
    else:
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
            ax.plot(GAMMAS,[r[key] for r in curves[state]],label=STATE_LABELS[state],lw=2)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel=r'$\gamma$',ylabel=ylabel,title=title)
        ax.legend()
        ax.grid(alpha=.2)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    ax.plot(GAMMAS,[r['d_induce_ratio'] for r in curves['0']],lw=2,
            label=r'$R_{0\to j}=p_1^*(0)/p_2^*(j)$')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel=r'$\gamma$',ylabel=r'$dR_{0\to j}/d\gamma$',title='Induce-then-harvest price-ratio derivative')
    ax.legend()
    ax.grid(alpha=.2)
    save(fig,'induce_harvest_price_ratio_derivative')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for state in ['j','0']:
        axes[0].plot(GAMMAS,[r['p1'] for r in curves[state]],label=STATE_LABELS[state],lw=2)
    for state,key,label in [('j','p2j',r'Previous inside ($x_1=j$)'),('0','p20',r'Previous outside ($x_1=0$)')]:
        axes[1].plot(GAMMAS,[r[key] for r in curves['j']],label=label,lw=2)
    for ax,title,ylabel in zip(axes,['Optimal period-1 prices','Optimal period-2 prices'],[r'$p_1^*$',r'$p_2^*$']):
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel=r'$\gamma$',ylabel=ylabel,title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save(fig,'optimal_prices_by_gamma')
    maximum=max(abs(r[k]) for state in curves for r in curves[state] for k in ['foc1','foc2j','foc20'])
    if maximum>1e-9:
        raise AssertionError(f'FOC residual {maximum}')
    print(f'Saved ten figures, {2*len(GAMMAS):,} solved observations, and {len(outcomes):,} conditional outcomes; max FOC residual={maximum:.3g}')





# === 7. Checks and run records === #



def self_test():
    for gamma, prior, delta in [(0., .5, .92), (-2., .25, .92), (1.5, .75, .92),
                                 (-1., 0., .92), (1., 1., .92), (1., .5, 0.)]:
        solution = supply_solution(gamma, prior, delta=delta)
        paths, summary = outcome_tables(solution)
        assert len(paths) == 12 and len(summary) == 3
        for key in ["hidden_foc", "committed_foc1", "committed_foc2"]:
            assert abs(solution[key]) < 1e-6
        if gamma == 0:
            assert np.ptp([row["optimal_profit"] for row in summary]) < 1e-7
            assert all(np.isclose(row["price_ratio"], 1., atol=1e-6) for row in paths)
        if prior in (0., 1.):
            assert np.isclose(solution["known_value"], solution["hidden_value"], atol=1e-7)
    print("Supply checks passed: all cases, four histories, probabilities, profit reconstruction, FOCs, degenerate priors, gamma=0, delta=0.")

def write_run_notes(output, information):
    """Write the same run information in plain text; you never edit a JSON file."""
    output.mkdir(parents=True, exist_ok=True)
    text = "RUN INFORMATION\n\n"
    text += "This is an output record, not a settings file.\n"
    text += "Change the Python parameters at the top of the script to change the run.\n\n"
    text += pformat(information, sort_dicts=False, width=90)
    (output / "run_notes.txt").write_text(text + "\n")




# === 8. Optional command-line overrides === #



def read_command_line(settings):
    """Optional server flags. Normal use: edit the parameters at the top."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "supply_outputs")
    parser.add_argument("--gamma-min", type=float, default=-3.)
    parser.add_argument("--gamma-max", type=float, default=3.)
    parser.add_argument("--gamma-points", type=int, default=61)
    parser.add_argument("--priors", type=float, nargs="+", default=[.25, .5, .75])
    parser.add_argument("--beta", type=float, default=2.)
    parser.add_argument("--alpha", type=float, default=-1., help="Signed price coefficient, negative.")
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--cost", type=float, default=0.)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--derivative-gamma-min", type=float)
    parser.add_argument("--derivative-gamma-max", type=float)
    parser.add_argument("--derivative-gamma-points", type=int)
    parser.add_argument("--skip-derivatives", action="store_true")
    parser.add_argument("--from-solutions", action="store_true")
    parser.add_argument("--no-appendix", action="store_true")
    parser.set_defaults(**vars(settings))
    return parser.parse_args()




# === 9. Run the supply exercise, step by step === #



def run_supply(settings):
    """The actual research steps, in the order they run."""
    global PARAMETERS, GAMMAS, OUT, from_solutions
    PARAMETERS = dict(beta=settings.beta, alpha=settings.alpha, delta=settings.delta, c=settings.cost)
    GAMMAS = np.linspace(settings.derivative_gamma_min, settings.derivative_gamma_max, settings.derivative_gamma_points)
    OUT = settings.output / "derivatives" / "figures"
    from_solutions = settings.from_solutions
    if settings.self_test:
        self_test()
        return
    if settings.gamma_points < 2 or settings.gamma_max <= settings.gamma_min:
        raise ValueError("Need >=2 gamma points and gamma-max > gamma-min.")
    if not settings.skip_derivatives:
        if settings.derivative_gamma_points < 2 or settings.derivative_gamma_max <= settings.derivative_gamma_min:
            raise ValueError("The derivative grid needs at least two points and increasing endpoints.")
        if not settings.no_appendix:
            benchmark_parameters = dict(beta=2.0, alpha=-1.0, delta=1.0, c=0.0)
            benchmark_grid = np.linspace(-3.0, 3.0, 601)
            if PARAMETERS != benchmark_parameters or not np.array_equal(GAMMAS, benchmark_grid):
                raise ValueError("The written appendix describes the original benchmark. Use no_appendix=True for a custom calibration/grid.")
    settings.output.mkdir(parents=True, exist_ok=True)
    solutions, paths, summaries = [], [], []
    for prior in settings.priors:
        for i, gamma in enumerate(np.linspace(settings.gamma_min, settings.gamma_max, settings.gamma_points)):
            result = supply_solution(gamma, prior, settings.beta, settings.alpha, settings.delta, settings.cost)
            history_rows, summary_rows = outcome_tables(result)
            solutions.append(result)
            paths.extend(history_rows)
            summaries.extend(summary_rows)
            if i % 10 == 0 or i == settings.gamma_points-1:
                print(f"Prior {prior:g}: solved {i+1}/{settings.gamma_points} gamma values", flush=True)
    raw, histories, expected = pd.DataFrame(solutions), pd.DataFrame(paths), pd.DataFrame(summaries)
    raw.to_csv(settings.output / "supply_solutions.csv", index=False)
    histories.to_csv(settings.output / "supply_history_outcomes.csv", index=False)
    expected.to_csv(settings.output / "supply_expected_outcomes.csv", index=False)
    residual = float(raw[["hidden_foc", "committed_foc1", "committed_foc2"]].abs().to_numpy().max())
    if residual > 1e-6:
        raise AssertionError(f"FOC residual too large: {residual}")
    make_case_figures(histories, expected, settings.output)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(settings).items()},
                    maximum_foc_residual=residual, calibrations=len(raw), cases=CASES,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (settings.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_run_notes(settings.output, manifest)
    if not settings.skip_derivatives:
        make_derivative_figures()
        if not settings.no_appendix:
            theory.make_latex_appendix(OUT.parent, PARAMETERS, GAMMAS, evaluate)
    print(f"All three supply cases completed. CSVs and PNG/PDF figures: {settings.output}")

if __name__ == "__main__":
    # Collect the ordinary Python variables above into one settings record.
    # settings.beta means exactly the variable beta from the Parameters section.
    settings = SimpleNamespace(
        output=output,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_points=gamma_points,
        priors=priors,
        beta=beta,
        alpha=alpha,
        delta=delta,
        cost=cost,
        derivative_gamma_min=derivative_gamma_min,
        derivative_gamma_max=derivative_gamma_max,
        derivative_gamma_points=derivative_gamma_points,
        skip_derivatives=skip_derivatives,
        from_solutions=from_solutions,
        no_appendix=no_appendix,
        self_test=run_self_tests,
    )
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_supply(settings)
