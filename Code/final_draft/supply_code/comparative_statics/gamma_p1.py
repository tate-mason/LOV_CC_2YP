from model.symbols import *
from model.shares import *

dp1_dgamma = (-z1 * sj1 + delta * z2 * sj2 * s01) / alpha
dsj1_dgamma_total = dsj1_dgamma + dsj1_dp1 * dp1_dgamma
dsj1_dgamma_total = sp.simplify(dsj1_dgamma_total)
