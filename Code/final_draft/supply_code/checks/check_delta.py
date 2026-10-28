import sympy as sp
from model.symbols import *

V1 = beta + gamma * z1 + alpha * p1
s01_expr = 1 / (1 + sp.exp(V1))

R2 = sp.symbols("R2", positive=True)

F = p1 - c + 1 / (alpha * s01_expr) - delta * R2 / alpha

dp1_ddelta = sp.simplify(-sp.diff(F, delta) / sp.diff(F, p1))

expected = s01_expr * R2 / alpha

assert sp.simplify(dp1_ddelta - expected) == 0

print("Delta pricing derivative: PASS")
sp.pprint(dp1_ddelta)
