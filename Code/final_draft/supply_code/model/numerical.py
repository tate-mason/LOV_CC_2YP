"""Unconstrained backward induction; derivatives evaluated from SymPy."""
import numpy as np
import sympy as sp
from scipy.special import expit
from scipy.optimize import brentq
from model import symbols as S
from model.utility import signed_state
from comparative_statics import canonical as C

PARAMETERS = dict(beta=2., alpha=-1., delta=.92, c=0.)
_KEYS = ("dp1", "d2p1", "dprofit", "matched_j", "matched_0", "induce_harvest", "dp2j", "dp20")
_ARGS = (S.alpha,S.beta,S.gamma,S.delta,S.c,S.q,S.sj1,S.u,S.v,S.p1,S.p2j,S.p20,S.Vj,S.V0)
_EXPRESSIONS = (C.dp1_dgamma,C.d2p1_dgamma2,C.dPi_dgamma,C.matched_j,C.matched_0,C.induce_harvest,
                C.price2_derivative(S.gamma,S.u,-1,S.p2j),C.price2_derivative(S.gamma,S.v,1,S.p20))
_EVALUATE = sp.lambdify(_ARGS,_EXPRESSIONS,"numpy")


def inside_share(price,gamma,state,beta=2.,alpha=-1.):
    return float(expit(beta+signed_state(state)*gamma+alpha*price))


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


def evaluate(gamma,initial_state,beta=2.,alpha=-1.,delta=.92,c=0.):
    q = signed_state(initial_state)
    if not (np.isfinite([gamma,beta,alpha,delta,c]).all() and alpha<0 and 0<=delta<1):
        raise ValueError("Require finite parameters, alpha<0, and 0<=delta<1 (delta=0 is the static counterfactual).")
    p2j,p20 = [optimal_price(beta+signed_state(state)*gamma,alpha,c) for state in ["j","0"]]
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
