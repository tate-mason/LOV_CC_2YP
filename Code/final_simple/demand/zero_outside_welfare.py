"""Alternative welfare primitives: history changes WTP, outside baseline stays zero.

This is a welfare specification, not a change in choice probabilities or a refit.
The original utility-level implementation remains available in binary_welfare.py.
"""
import numpy as np
from binary_welfare import evaluate_policy


def evaluate_zero_outside(gamma,h0,alpha,beta,delta,p1,p2j,p20,cost=0.):
    original=evaluate_policy(gamma,h0,alpha,beta,delta,p1,p2j,p20,cost)
    g,first,pj,p0=np.broadcast_arrays(*[np.asarray(x,float) for x in [gamma,p1,p2j,p20]])
    a=-alpha
    cs1=np.logaddexp(0.,beta+alpha*first+(-g if h0=='j' else g))/a
    q1=original['quantity1']
    cs2=(q1*np.logaddexp(0.,beta+alpha*pj-g)
          +(1-q1)*np.logaddexp(0.,beta+alpha*p0+g))/a
    cs=cs1+delta*cs2
    result=dict(original)
    for metric in ['consumer_surplus1','consumer_surplus2','consumer_surplus','total_welfare']:
        result['original_'+metric]=original[metric]
    result.update(consumer_surplus1=cs1,consumer_surplus2=cs2,consumer_surplus=cs,
                  total_welfare=cs+original['profit'])
    np.testing.assert_allclose(cs,original['zero_outside_consumer_surplus'],atol=1e-12)
    np.testing.assert_allclose(result['total_welfare'],original['zero_outside_total_welfare'],atol=1e-12)
    return result
