"""Pure category-switching pricing with explicit common/separate flavor prices."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demand'))
from category_switching import logsumexp
from quintile_theory import solve_distribution


def probabilities(prices, gamma, h0, alpha, beta):
    gamma = np.asarray(gamma)
    u = np.asarray(beta)+alpha*np.asarray(prices)+(-gamma if h0 == 'j' else gamma)[..., None]
    u = np.concatenate([u, np.zeros((*u.shape[:-1], 1))], axis=-1)
    return np.exp(u-logsumexp(u, axis=-1, keepdims=True))


def outcomes(gamma, h0, alpha, beta, delta, p1, p2j, p20):
    gamma = np.asarray(gamma)
    probability1 = probabilities(p1, gamma, h0, alpha, beta)
    probabilityj = probabilities(p2j, gamma, 'j', alpha, beta)
    probability0 = probabilities(p20, gamma, '0', alpha, beta)
    q1 = probability1[..., :3].sum(axis=-1)
    any_yogurt = q1
    revenue1 = probability1[..., :3]@np.asarray(p1)
    rj = probabilityj[..., :3]@np.asarray(p2j)
    r0 = probability0[..., :3]@np.asarray(p20)
    revenue = revenue1+delta*(any_yogurt*rj+(1-any_yogurt)*r0)
    def surplus(prices, history):
        vj = np.broadcast_to(np.asarray(beta)+alpha*np.asarray(prices), (*gamma.shape, 3)).copy()
        if history == '0':
            vj += gamma[..., None]
        v0 = gamma if history == 'j' else np.zeros_like(gamma)
        return logsumexp(np.concatenate([vj, v0[..., None]], axis=-1), axis=-1)/(-alpha)
    cs1 = surplus(p1, h0)
    cs2 = any_yogurt*surplus(p2j, 'j')+(1-any_yogurt)*surplus(p20, '0')
    cs = cs1+delta*cs2
    return dict(revenue=revenue, consumer_surplus=cs, total_welfare=cs+revenue,
                revenue1=revenue1, revenue2=any_yogurt*rj+(1-any_yogurt)*r0,
                consumer_surplus1=cs1, consumer_surplus2=cs2,
                quantity1=q1, purchase_week1=any_yogurt,
                quantity2=any_yogurt*probabilityj[..., :3].sum(axis=-1)
                    +(1-any_yogurt)*probability0[..., :3].sum(axis=-1))


def solve(gamma, weights, h0, alpha, beta, delta=1., mode='common_price'):
    """Exact common-markup reduction, with separate-price FOC certification.

    One owner, common fixed alpha, equal zero costs, and category-only transitions
    imply equal prices across flavors even when three independent prices are
    permitted. For any inclusive value, a common price maximizes conditional
    revenue (the difference is a KL divergence). Thus the scalar reduction is
    also the global solution of the unrestricted three-price problem.
    """
    if mode not in ['common_price', 'separate_flavor_prices']:
        raise ValueError('Unknown pricing mode.')
    beta = np.asarray(beta, float)
    if beta.shape != (3,) or not np.isfinite(beta).all():
        raise ValueError('Need three finite flavor intercepts.')
    gamma, weights = np.asarray(gamma,float), np.asarray(weights,float)
    if (gamma.ndim != 1 or gamma.shape != weights.shape or not len(gamma)
            or not np.isfinite(gamma).all() or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isclose(weights.sum(),1)
            or alpha >= 0 or not np.isfinite(alpha) or h0 not in ['j','0']
            or not 0 <= delta <= 1):
        raise ValueError('Invalid pricing distribution or parameters.')
    effective_beta = float(logsumexp(beta))
    result=solve_distribution(gamma,weights,h0,alpha,effective_beta,delta)
    for key in ['p1', 'p2j', 'p20']:
        result[key] = np.full(3, result[key])
    full = outcomes(gamma, h0, alpha, beta, delta,
                    result['p1'], result['p2j'], result['p20'])
    np.testing.assert_allclose(result['objective'], np.asarray(weights)@full['revenue'], atol=1e-10)
    # All three unconstrained first-price derivatives must vanish, not just
    # their sum along a common-price direction.
    epsilon = 1e-5
    derivatives = []
    for j in range(3):
        bump = np.zeros(3); bump[j] = epsilon
        plus = outcomes(gamma, h0, alpha, beta, delta,
                        result['p1']+bump, result['p2j'], result['p20'])
        minus = outcomes(gamma, h0, alpha, beta, delta,
                         result['p1']-bump, result['p2j'], result['p20'])
        derivatives.append(float(np.asarray(weights)@(plus['revenue']-minus['revenue']))/(2*epsilon))
    result['flavor_first_order_residual'] = max(abs(np.asarray(derivatives)))
    result['mode'] = mode
    result['effective_beta'] = effective_beta
    return result
