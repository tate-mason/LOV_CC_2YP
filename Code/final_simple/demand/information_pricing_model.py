"""Pure two-period price model: only novelty is hidden, habit and price taste known.

Inputs are mathematical utility/history arrays. This module performs no empirical
I/O and creates no household records. The server runner supplies fitted contexts.
"""
from math import lgamma
import numpy as np
from scipy.special import expit, logsumexp, wrightomega

try:
    from .history import update_history
except ImportError:
    from history import update_history


def baskets(capacity, inside):
    def generate(total, slots):
        if slots == 1:
            yield (total,)
        else:
            for n in range(total+1):
                for rest in generate(total-n, slots-1):
                    yield (n,)+rest
    return np.array(list(generate(capacity, inside+1)), dtype=int)


def _scalar_maximum(value, derivative, bound, points):
    """Screen stationary points on a dimensionless grid and refine every maximum."""
    for _ in range(8):
        count = max(points, int(8*bound)+1)
        if count > 32769:
            raise ValueError('Price search too wide; inspect the fitted utility scale.')
        grid = np.linspace(-bound, bound, count)
        slopes = derivative(grid)
        if slopes[0] > 0 and slopes[-1] < 0:
            break
        bound *= 2
    else:
        raise ArithmeticError('Could not bracket first-period price optimum.')
    left = grid[:-1][(slopes[:-1] >= 0) & (slopes[1:] <= 0)]
    right = grid[1:][(slopes[:-1] >= 0) & (slopes[1:] <= 0)]
    for _ in range(48):
        mid = (left+right)/2
        positive = derivative(mid) > 0
        left, right = np.where(positive, mid, left), np.where(positive, right, mid)
    candidates = np.r_[grid, (left+right)/2]
    values = value(candidates)
    best = int(np.argmax(values))
    if candidates[best] in [grid[0], grid[-1]]:
        raise ArithmeticError('First-period optimum touches search boundary.')
    return float(candidates[best]), float(values[best])


def _terminal_common(intercepts, weights, alpha, capacity, points):
    """Fixed-belief terminal optimum for every basket, shared across novelty types.

    Optimize in t=alpha*p. Every mixture optimum lies between the component
    optima. Grid-screen all local maxima and bisect all positive-to-negative FOCs.
    """
    component = 1+wrightomega(intercepts-1)
    lo, hi = component.min(axis=0), component.max(axis=0)
    count = max(points, int(8*np.max(hi-lo))+1)
    if count > 32769:
        raise ValueError('Terminal mixture too wide for the price search.')
    grid = lo[None, :]+np.linspace(0, 1, count)[:, None]*(hi-lo)[None, :]
    def objective(t, states):
        q = expit(intercepts[:, states]-t[None, :])
        return capacity/alpha*t*(weights[:, None]*q).sum(axis=0)
    def slope(t, states):
        q = expit(intercepts[:, states]-t[None, :])
        return (weights[:, None]*q*(1-t[None, :]*(1-q))).sum(axis=0)
    q = expit(intercepts[:, None, :]-grid[None, :, :])
    d = (weights[:, None, None]*q*(1-grid[None, :, :]*(1-q))).sum(axis=0)
    values = capacity/alpha*grid*(weights[:, None, None]*q).sum(axis=0)
    selected = values.argmax(axis=0)
    states_all = np.arange(len(lo))
    price_t = grid[selected, states_all].copy()
    best = values[selected, states_all].copy()
    rows, states = np.where((d[:-1] >= 0) & (d[1:] <= 0))
    left, right = grid[rows, states], grid[rows+1, states]
    for _ in range(48):
        mid = (left+right)/2
        positive = slope(mid, states) > 0
        left, right = np.where(positive, mid, left), np.where(positive, right, mid)
    roots = (left+right)/2
    root_values = objective(roots, states)
    for state, root, value in zip(states, roots, root_values):
        if value > best[state]:
            best[state], price_t[state] = value, root
    np.testing.assert_allclose(slope(price_t, states_all), 0., atol=2e-9)
    return price_t/alpha, best


def solve_environment(prices, available, beta, novelty, stock, alpha, habit,
                      gamma, weights, capacity=7, focal=1, delta=1.,
                      stock_measure='cumulative', stock_retention=1., occasions=1,
                      controls=None, search_points=65):
    """Solve one known-price/habit/history environment across persistent gamma types.

    Known regimes share exactly the same utility and history transition. Complete
    information knows gamma; limited information uses fixed weights at each state.
    The latter first-price objective is the theory's fixed-belief objective.
    Both policies are evaluated using the SAME persistent type in both periods.
    Common means common across gamma, conditional on known alpha, habit, history,
    rival prices and menus; it does not mean one dollar price across households.
    All types are priced; there is no middle-group price restriction. First prices
    can be negative; terminal prices are nonnegative at zero marginal cost.
    """
    arrays = [np.asarray(x, float) for x in [prices, beta, novelty, stock, gamma, weights]]
    prices, beta, novelty, stock, gamma, weights = arrays
    available = np.asarray(available, bool)
    j = len(prices)
    if (any(x.shape != (j,) for x in [beta, novelty, stock, available])
            or gamma.ndim != 1 or weights.shape != gamma.shape or len(gamma) < 1
            or not all(np.isfinite(x).all() for x in arrays)
            or not np.isfinite([alpha, habit, delta]).all() or alpha <= 0
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)
            or (prices < 0).any() or not 0 <= delta <= 1
            or int(capacity) != capacity or not 1 <= capacity <= 12
            or not 0 <= focal < j or not available[focal] or search_points < 17):
        raise ValueError('Invalid pricing dimensions, probabilities, menu or parameters.')
    capacity = int(capacity)
    controls = np.zeros(j) if controls is None else np.asarray(controls, float)
    if controls.shape != (j,) or not np.isfinite(controls).all():
        raise ValueError('Invalid fixed utility controls.')
    # Enumerate only available alternatives, retaining full shares for history.
    active = np.flatnonzero(available)
    local_focal = int(np.flatnonzero(active == focal)[0])
    choices = np.r_[active, j]  # outside/nonpurchase has fixed utility zero
    states_local = baskets(capacity, len(active))
    states = np.zeros((len(states_local), j+1), dtype=int)
    states[:, choices] = states_local
    inside_total = states[:, :j].sum(axis=1, keepdims=True)
    shares = np.divide(states[:, :j], inside_total,
                       out=np.zeros_like(states[:, :j], dtype=float), where=inside_total > 0)
    novelty2 = shares.sum(axis=1, keepdims=True)-shares
    stock2 = update_history(stock, shares, occasions, measure=stock_measure, retention=stock_retention)
    first = beta+gamma[:, None]*novelty+habit*stock+controls-alpha*prices
    terminal = (beta[None, None, :]+gamma[:, None, None]*novelty2[None, :, :]
                +habit*stock2[None, :, :]+controls[None, None, :]-alpha*prices)
    first[:, focal] += alpha*prices[focal]
    terminal[:, :, focal] += alpha*prices[focal]
    nonfocal = active[active != focal]
    competitor1 = np.column_stack([first[:, nonfocal], np.zeros(len(gamma))])
    intercept1 = first[:, focal]-logsumexp(competitor1, axis=1)
    competitor2 = np.concatenate([terminal[:, :, nonfocal],
                                  np.zeros((*terminal.shape[:2], 1))], axis=-1)
    intercept2 = terminal[:, :, focal]-logsumexp(competitor2, axis=-1)
    p2_full = (1+wrightomega(intercept2-1))/alpha
    q2_full = expit(intercept2-alpha*p2_full)
    r2_full = capacity*p2_full*q2_full
    p2_limited, perceived2 = _terminal_common(intercept2, weights, alpha, capacity, search_points)
    r2_limited = capacity*p2_limited[None, :]*expit(intercept2-alpha*p2_limited)
    # Given focal count n, relative probabilities of all other alternatives do
    # not depend on the focal price. Aggregate continuation by n to optimize p1.
    n = states[:, focal]
    others = np.r_[nonfocal, j]
    log_cond = competitor1-logsumexp(competitor1, axis=1, keepdims=True)
    remaining_counts = states[:, others]
    coefficient = np.array([lgamma(capacity-int(k)+1)-sum(lgamma(int(c)+1) for c in row)
                            for k, row in zip(n, remaining_counts)])
    conditional = np.exp(coefficient[None, :]+log_cond @ remaining_counts.T)
    c_full = np.zeros((len(gamma), capacity+1))
    c_perceived = np.zeros_like(c_full)
    for k in range(capacity+1):
        mask = n == k
        np.testing.assert_allclose(conditional[:, mask].sum(axis=1), 1., atol=1e-12)
        c_full[:, k] = (conditional[:, mask]*r2_full[:, mask]).sum(axis=1)
        c_perceived[:, k] = conditional[:, mask] @ perceived2[mask]
    counts = np.arange(capacity+1)
    binomial_coef = np.array([lgamma(capacity+1)-lgamma(int(k)+1)-lgamma(capacity-int(k)+1)
                             for k in counts])
    def binomial(t, intercept):
        t = np.atleast_1d(t)
        logq = -np.logaddexp(0., t-intercept)
        logone = -np.logaddexp(0., intercept-t)
        mass = np.exp(binomial_coef+logq[:, None]*counts+logone[:, None]*(capacity-counts))
        return expit(intercept-t), mass
    def functions(intercepts, continuation, w):
        def value(t):
            result = np.zeros(np.atleast_1d(t).shape)
            for z, c, weight in zip(intercepts, continuation, w):
                q, mass = binomial(t, z)
                result += weight*(capacity*q*np.atleast_1d(t)/alpha+delta*(mass @ c))
            return result
        def slope(t):
            result = np.zeros(np.atleast_1d(t).shape)
            for z, c, weight in zip(intercepts, continuation, w):
                q, mass = binomial(t, z)
                continuation_slope = (mass*(capacity*q[:, None]-counts)) @ c
                result += weight*(capacity/alpha*q*(1-np.atleast_1d(t)*(1-q))
                                  +delta*continuation_slope)
            return result
        return value, slope
    bound = max(10., float(np.max(np.abs(intercept1)))+5.,
                float(np.max(alpha*r2_full))/capacity+5.)
    p1_full = np.empty(len(gamma))
    for k in range(len(gamma)):
        val, derivative = functions(intercept1[k:k+1], c_full[k:k+1], np.ones(1))
        p1_full[k] = _scalar_maximum(val, derivative, bound, search_points)[0]/alpha
    val, derivative = functions(intercept1, c_perceived, weights)
    limited_t, limited_objective = _scalar_maximum(val, derivative, bound, search_points)
    p1_limited = np.full(len(gamma), limited_t/alpha)
    def evaluate(p1, p2, r2):
        mass = np.empty((len(gamma), len(states)))
        r1, revenue2, expected2 = [], [], []
        for k in range(len(gamma)):
            q, bin_mass = binomial(np.array([alpha*p1[k]]), intercept1[k])
            mass[k] = conditional[k]*bin_mass[0, n]
            r1.append(capacity*q[0]*p1[k])
            revenue2.append(mass[k] @ r2[k])
            expected2.append(mass[k] @ p2[k])
        np.testing.assert_allclose(mass.sum(axis=1), 1., atol=1e-10)
        return dict(p1=p1, expected_p2=np.array(expected2), revenue1=np.array(r1),
                    revenue2=np.array(revenue2), revenue=np.array(r1)+delta*np.array(revenue2),
                    state_probability=mass)
    full = evaluate(p1_full, p2_full, r2_full)
    limited = evaluate(p1_limited, np.broadcast_to(p2_limited, p2_full.shape), r2_limited)
    # Every limited-information price path is feasible for each informed type.
    if (full['revenue']+1e-7 < limited['revenue']).any():
        raise AssertionError('Known-type optimization fails to dominate common policy.')
    return dict(full=full, limited=limited, gamma=gamma, weights=weights, states=states,
                p2_full=p2_full, p2_limited=p2_limited,
                terminal_revenue_full=r2_full, terminal_revenue_limited=r2_limited,
                limited_beliefs=np.broadcast_to(weights[:, None], (len(gamma), len(states))).copy(),
                limited_objective=limited_objective,
                full_revenue=float(weights @ full['revenue']),
                limited_revenue=float(weights @ limited['revenue']),
                gain=float(weights @ (full['revenue']-limited['revenue'])))
