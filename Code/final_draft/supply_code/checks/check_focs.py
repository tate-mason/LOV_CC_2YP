import sympy as sp
from model.symbols import *
from model.shares import sj1_expr, sj2_expr
from pricing.period1_foc import foc1_raw, foc1_collected
from pricing.period2_foc import foc2
from pricing.period1_price import p1_star
from pricing.period2_price import p2_star

shares = {sj1: sj1_expr, s01: 1-sj1_expr, sj2: sj2_expr, s02: 1-sj2_expr}
profit = sj1_expr * (p1-c + delta*sj2_expr*(p2-c))
# Independent decision prices: U2 does not depend on p1. The p2 envelope
# condition also justifies this derivative when p2 is optimized first.
assert sp.simplify(foc1_raw-foc1_collected) == 0
assert sp.simplify(sp.diff(profit,p1)-foc1_raw.subs(shares)) == 0
assert sp.simplify(sp.diff(profit,p2)-delta*sj1_expr*foc2.subs(shares)) == 0
assert sp.simplify(foc2.subs(p2,p2_star)) == 0
assert sp.simplify(foc1_raw.subs(p2,p2_star).subs(p1,p1_star)) == 0
print('Both FOCs and price equations: PASS')
