"""Pure two-period welfare accounting for the presentation's binary theory.

No empirical I/O. Signed price coefficient alpha < 0; myopic logit choices.
CS is expected experienced utility in dollars, with a common shock constant
omitted. Preserve V0(h)=gamma*1{h=j}: normalizing it away changes dynamic welfare.
"""
import numpy as np


def expit(x):
    """Stable logistic probability using NumPy alone."""
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def evaluate_policy(gamma, h0, alpha, beta, delta, p1, p2j, p20, cost=0.):
    """Type-specific outcomes, integrating shocks and each policy's own histories.

    Report zero-outside CS separately as a normalization sensitivity. It has
    identical choices but a different cardinal utility assumption across states.
    Quantity is expected binary purchases, not physical package volume.
    """
    gamma, p1, p2j, p20 = np.broadcast_arrays(
        *[np.asarray(x, float) for x in [gamma, p1, p2j, p20]])
    if (h0 not in ['j', '0'] or alpha >= 0 or not 0 <= delta <= 1
            or not np.isfinite([alpha, beta, delta, cost]).all()
            or not all(np.isfinite(x).all() for x in [gamma, p1, p2j, p20])):
        raise ValueError('Need finite inputs, alpha < 0, delta in [0,1], and binary h0.')
    v1 = beta + alpha*p1 + (-gamma if h0 == 'j' else gamma)
    v2j, v20 = beta + alpha*p2j - gamma, beta + alpha*p20 + gamma
    q1, q2j, q20 = expit(v1), expit(v2j), expit(v20)
    q2 = q1*q2j + (1-q1)*q20
    revenue1, revenue2 = p1*q1, q1*p2j*q2j + (1-q1)*p20*q20
    relative_cs1 = np.logaddexp(0., v1)/(-alpha)
    relative_cs2 = (q1*np.logaddexp(0., v2j)
                   + (1-q1)*np.logaddexp(0., v20))/(-alpha)
    cs1 = relative_cs1 + (gamma/(-alpha) if h0 == 'j' else 0.)
    cs2 = relative_cs2 + q1*gamma/(-alpha)
    quantity = q1 + delta*q2
    revenue = revenue1 + delta*revenue2
    profit = revenue - cost*quantity
    cs = cs1 + delta*cs2
    return dict(p1=p1, p2j=p2j, p20=p20, expected_p2=q1*p2j+(1-q1)*p20,
                quantity1=q1, quantity2=q2, quantity=quantity,
                undiscounted_quantity=q1+q2, revenue1=revenue1, revenue2=revenue2,
                revenue=revenue, profit=profit, consumer_surplus1=cs1,
                consumer_surplus2=cs2, consumer_surplus=cs, total_welfare=cs+profit,
                zero_outside_consumer_surplus=relative_cs1+delta*relative_cs2,
                zero_outside_total_welfare=relative_cs1+delta*relative_cs2+profit)
