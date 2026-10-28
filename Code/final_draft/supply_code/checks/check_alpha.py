import sympy as sp

from model.symbols import *

P1_alpha = sp.Symbol("P1_alpha")

dsj1_dalpha_manual = sj1 * s01 * (p1 + alpha * P1_alpha)

dsj1_dalpha_direct = p1 * sj1 * s01
dsj1_dp1 = alpha * sj1 * s01

dsj1_dalpha_check = dsj1_dalpha_direct + dsj1_dp1 * P1_alpha

assert sp.simplify(dsj1_dalpha_manual - dsj1_dalpha_check) == 0

print("Alpha share derivative: PASS")
