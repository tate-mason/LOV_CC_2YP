import sympy as sp
from model.symbols import *
from model.profit import pi2
from model.shares import dsj2_dp2

foc2 = sp.expand(sj2 + (p2 - c) * dsj2_dp2)
foc2_eq = sp.Eq(foc2, 0)
