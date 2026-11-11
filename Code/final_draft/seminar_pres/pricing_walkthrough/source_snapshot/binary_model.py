"""Two binary choices: focal flavor j versus the composite not-j option.

No baskets or rival-specific future histories. Existing fitted coefficients are
plugged into an explicitly specified binary extension of the supply model.
This is a counterfactual parameter mapping, not a refit or an exact dynamic
aggregation of the fitted multinomial model. No empirical I/O occurs here.
"""
import numpy as np
from scipy.special import expit, logsumexp, wrightomega
try:
    from .information_pricing_model import _terminal_common
except ImportError:
    from information_pricing_model import _terminal_common


def solve_binary(prices, available, beta, stock, alpha, habit, gamma, weights,
                 initial_inside, focal=1, delta=1., controls=None, search_points=65):
    """Known alpha/habit, known initial binary history, gamma known or hidden.

    Theory-style relative utility in BOTH periods:
      v_r(p,h,H) = b + gamma_r*(1{h=0}-1{h=j}) + habit*H - alpha*p.
    b = beta_j + fixed control_j - C, where C is the logsumexp of available
    rivals' baseline utilities (including their initial habit stocks) and zero
    for nonpurchase. C, rival prices, and rival stocks stay fixed for two periods.
    gamma is excluded from C because its switching effect is represented by the
    explicit binary history term. Own habit updates H' = H + 1{focal chosen}.
    Initial H comes from the saved cumulative-stock fit. Future periods each have
    ONE choice opportunity, not seven weekly purchase counts.

    Limited information keeps gamma weights fixed at both decisions. Both
    policies are evaluated with the same persistent type through both periods.
    """
    prices, beta, stock, gamma, weights = [np.asarray(x, float) for x in
                                         [prices, beta, stock, gamma, weights]]
    available = np.asarray(available, bool)
    j = len(prices)
    if (any(x.shape != (j,) for x in [beta, stock, available])
            or gamma.ndim != 1 or weights.shape != gamma.shape or not len(gamma)
            or not all(np.isfinite(x).all() for x in [prices, beta, stock, gamma, weights])
            or not np.isfinite([alpha, habit, delta]).all() or alpha <= 0
            or not 0 <= delta <= 1 or (weights < 0).any()
            or not np.isclose(weights.sum(), 1.) or (prices < 0).any()
            or not 0 <= focal < j or not available[focal]
            or initial_inside not in [False, True] or search_points < 17):
        raise ValueError('Invalid binary-pricing arrays, weights, menu or parameters.')
    controls = np.zeros(j) if controls is None else np.asarray(controls, float)
    if controls.shape != (j,) or not np.isfinite(controls).all():
        raise ValueError('Invalid fixed utility controls.')
    rivals = np.flatnonzero(available & (np.arange(j) != focal))
    rival_base = beta[rivals]-alpha*prices[rivals]+habit*stock[rivals]+controls[rivals]
    outside_index = float(logsumexp(np.r_[rival_base, 0.]))
    intercept = float(beta[focal]+controls[focal]-outside_index)
    own_stock = float(stock[focal])
    history_sign = -1. if initial_inside else 1.
    intercept1 = intercept+gamma*history_sign+habit*own_stock
    # Exactly two successor histories, ordered [focal, not focal].
    intercept2 = np.column_stack([intercept-gamma+habit*(own_stock+1.),
                                  intercept+gamma+habit*own_stock])
    p2_full = (1+wrightomega(intercept2-1))/alpha
    r2_full = p2_full*expit(intercept2-alpha*p2_full)
    # Effective-cost transformation solves the informed first price analytically.
    effective_cost = -delta*(r2_full[:, 0]-r2_full[:, 1])
    p1_full = effective_cost+(1+wrightomega(intercept1-alpha*effective_cost-1))/alpha
    # The same fixed gamma weights enter each common terminal optimization.
    p2_limited, perceived2 = _terminal_common(intercept2, weights, alpha, 1, search_points)
    adjustment = delta*(perceived2[0]-perceived2[1])
    # Subjective value = Q(p)*(p+adjustment)+delta*perceived2[1].
    relative_price, relative_value = _terminal_common(
        intercept1[:, None]+alpha*adjustment, weights, alpha, 1, search_points)
    p1_limited = np.full(len(gamma), relative_price[0]-adjustment)
    limited_objective = float(relative_value[0]+delta*perceived2[1])
    r2_limited = p2_limited[None, :]*expit(intercept2-alpha*p2_limited)
    def evaluate(p1, p2, r2):
        s1 = expit(intercept1-alpha*p1)
        mass = np.column_stack([s1, 1-s1])
        revenue1 = p1*s1
        revenue2 = (mass*r2).sum(axis=1)
        return dict(p1=p1, expected_p2=(mass*p2).sum(axis=1),
                    revenue1=revenue1, revenue2=revenue2,
                    revenue=revenue1+delta*revenue2, state_probability=mass)
    full = evaluate(p1_full, p2_full, r2_full)
    limited = evaluate(p1_limited, np.broadcast_to(p2_limited, p2_full.shape), r2_limited)
    if (full['revenue']+1e-8 < limited['revenue']).any():
        raise AssertionError('Binary informed optimum does not dominate common policy.')
    for result in [full, limited]:
        np.testing.assert_allclose(result['state_probability'].sum(axis=1), 1., atol=1e-12)
    return dict(full=full, limited=limited, gamma=gamma, weights=weights,
                states=np.array([[1, 0], [0, 1]], dtype=int), focal_index=0,
                p2_full=p2_full, p2_limited=p2_limited,
                terminal_revenue_full=r2_full, terminal_revenue_limited=r2_limited,
                limited_beliefs=np.broadcast_to(weights[:, None], (len(gamma), 2)).copy(),
                limited_objective=limited_objective,
                full_revenue=float(weights @ full['revenue']),
                limited_revenue=float(weights @ limited['revenue']),
                gain=float(weights @ (full['revenue']-limited['revenue'])),
                outside_index=outside_index, binary_intercept=intercept,
                first_utility_intercept=intercept1, terminal_utility_intercept=intercept2)
