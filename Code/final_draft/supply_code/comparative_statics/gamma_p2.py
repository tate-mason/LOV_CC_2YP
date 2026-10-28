from model.symbols import *

dp2_dgamma = -(z2 * sj2) / alpha
dsj2_dgamma_total = sj2 * s02 * (z2 + alpha * dp2_dgamma)
dsj2_dgamma_total = sp.simplify(dsj2_dgamma_total)
