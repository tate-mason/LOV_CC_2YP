import sympy as sp
from model.symbols import *
from model.utility import U1, U2

# Period 1
sj1_expr = sp.exp(U1) / (1 + sp.exp(U1))
s01_expr = 1 / (1 + sp.exp(U1))

# Period 2
sj2_expr = sp.exp(U2) / (1 + sp.exp(U2))
s02_expr = 1 / (1 + sp.exp(U2))

# Share Derivatives

dsj1_dp1 = alpha * sj1 * s01
ds01_dp1 = -alpha * sj1 * s01
dsj1_dgamma = z1 * sj1 * s01

dsj2_dp2 = alpha * sj2 * s02
ds02_dp2 = -alpha * sj2 * s02
dsj2_dgamma_direct = z2 * sj2 * s02
# Coefficient of the period-2 utility change, not a total derivative.
dsj2_dutility = sj2 * s02
