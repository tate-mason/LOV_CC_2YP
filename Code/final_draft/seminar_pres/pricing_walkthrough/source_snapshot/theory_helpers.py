"""Shared binary supply calculations for the separate information exercises.

These preserve the baseline equations in supply_simulation.py. Only NumPy and
pandas are needed for calculations; matplotlib is loaded when drawing figures.
"""
from pathlib import Path
import math
import os
import tempfile
import numpy as np
import pandas as pd

CASES = ["known_state", "hidden_initial_state", "ex_ante_commitment"]
LABELS = dict(zip(CASES, ["Known state (base)", "Hidden initial state", "Ex ante commitment"]))

# === 1. Choice probabilities and numerical price search === #
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

def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

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
