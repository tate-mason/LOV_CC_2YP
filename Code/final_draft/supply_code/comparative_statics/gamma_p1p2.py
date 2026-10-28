import sympy as sp
from model.symbols import *
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma

ratio_derivative_raw = (p2 * dp1_dgamma - p1 * dp2_dgamma) / p2**2

ratio_derivative = sp.factor(sp.simplify(ratio_derivative_raw))

ratio_derivative_expected = (-p2 * z1 * sj1 + z2 * sj2 * (p1 + delta * s01 * p2)) / (
    alpha * p2**2
)

assert sp.simplify(ratio_derivative - ratio_derivative_expected) == 0
