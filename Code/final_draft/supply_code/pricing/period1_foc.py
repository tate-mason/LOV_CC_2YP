import sympy as sp
from model.symbols import *
from model.shares import dsj1_dp1

foc1_raw = sj1 + (p1 - c) * dsj1_dp1 + delta * sj2 * (p2 - c) * dsj1_dp1
foc1_collected = sj1 * dsj1_dp1 * (p1 - c + delta * sj2 * (p2 - c))
foc1_eq = sp.Eq(foc1_collected, 0)
