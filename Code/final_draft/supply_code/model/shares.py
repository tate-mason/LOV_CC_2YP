import sympy as sp
from model.symbols import *
from model.utility import U1, U2
sj1_expr = 1/(1+sp.exp(-U1))
sj2_expr = 1/(1+sp.exp(-U2))
s01_expr, s02_expr = 1-sj1_expr, 1-sj2_expr
dsj1_dp1, dsj2_dp2 = alpha*sj1*s01, alpha*sj2*s02
ds01_dp1, ds02_dp2 = -dsj1_dp1, -dsj2_dp2
dsj1_dgamma = q*sj1*s01
dsj2_dgamma_direct = q2*sj2*s02
dsj2_dutility = sj2*s02
