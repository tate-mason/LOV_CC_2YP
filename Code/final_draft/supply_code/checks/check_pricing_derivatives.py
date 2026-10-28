"""Implicit differentiation of explicit logit pricing equations."""
import sympy as sp
from model.symbols import *
from model.shares import sj1_expr, sj2_expr
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.beta_p1 import dp1_dbeta
from comparative_statics.beta_p2 import dp2_dbeta
from comparative_statics.alpha_p1 import dp1_dalpha
from comparative_statics.delta_p1 import dp1_ddelta
from comparative_statics.gamma_profit import dPi_dgamma
from model.profit import Pi

F = sp.Matrix([p1-c+1/(alpha*(1-sj1_expr))-delta/alpha*sj2_expr/(1-sj2_expr),
               p2-c+1/(alpha*(1-sj2_expr))])
J = F.jacobian([p1,p2])
shares = {sj1:sj1_expr, s01:1-sj1_expr, sj2:sj2_expr, s02:1-sj2_expr}
for parameter, expected in [
    (gamma,[dp1_dgamma,dp2_dgamma]), (beta,[dp1_dbeta,dp2_dbeta]),
    (alpha,[dp1_dalpha,(1-alpha*p2*sj2_expr)/alpha**2]),
    (delta,[dp1_ddelta,0])]:
    residual = J * sp.Matrix([sp.sympify(t).subs(shares) for t in expected]) + F.diff(parameter)
    for actual in residual:
        assert sp.simplify(actual) == 0, (parameter, sp.simplify(actual))
# Evaluate the envelope derivative on the FOC manifold, shares held symbolic.
envelope = sj1*s01*z1*(p1-c+delta*sj2*(p2-c)) + delta*sj1*sj2*s02*z2*(p2-c)
from pricing.period2_price import p2_star
from pricing.period1_price import p1_star
assert sp.simplify(envelope.subs(p2,p2_star).subs(p1,p1_star)-dPi_dgamma) == 0
print('Implicit price derivatives and profit envelope: PASS')
