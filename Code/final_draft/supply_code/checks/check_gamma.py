import sympy as sp
from model.symbols import *
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.gamma_p1p2 import ratio_derivative
assert sp.simplify(ratio_derivative-(p2*dp1_dgamma-p1*dp2_dgamma)/p2**2) == 0
a,d = sp.symbols('a d', positive=True)
expr = ratio_derivative.subs({alpha:-a,delta:d})
assert expr.subs({z1:0,z2:0}) == 0
assert sp.simplify(expr.subs({z1:1,z2:0})).is_positive
assert sp.simplify(expr.subs({z1:0,z2:1})).is_negative
print('Gamma quotient rule and three history signs: PASS')
