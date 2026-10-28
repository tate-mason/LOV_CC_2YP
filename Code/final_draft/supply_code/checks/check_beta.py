import sympy as sp
from model.symbols import *
from comparative_statics.beta_p1 import dp1_dbeta
from comparative_statics.beta_p2 import dp2_dbeta
from comparative_statics.beta_p1p2 import ratio_derivative
assert sp.simplify(ratio_derivative-(p2*dp1_dbeta-p1*dp2_dbeta)/p2**2) == 0
print('Beta quotient rule: PASS (implicit derivatives checked separately)')
