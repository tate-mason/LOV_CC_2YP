import sympy as sp

from model.symbols import *
from comparative_statics.beta_p1 import dp1_dbeta

dsj1_dbeta_direct = sj1 * s01
dsj2_dbeta_direct = sj2 * s02

dsj1_dbeta_total = dsj1_dbeta_direct + alpha * sj1 * s01 * sp.Symbol("P1_beta")

P1_beta = sp.Symbol("P1_beta")

pricing_equation = p1 - c + 1 / (alpha * s01) - delta / alpha * sj2 / s02
