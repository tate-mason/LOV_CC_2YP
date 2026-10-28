import sympy as sp

from model.symbols import *
from model.shares import sj2_expr

pi2_expr = sj1 * sj2_expr * (p2 - c)
foc2_calculated = sp.diff(pi2_expr, p2)
dsj2_dp = alpha * sj2_expr * (1 - sj2_expr)

foc2_expected = sj1 * (sj2_expr + (p2 - c) * dsj2_dp)

assert sp.simplify(foc2_calculated - foc2_expected) == 0

print("Period 2 FOC: PASS")
