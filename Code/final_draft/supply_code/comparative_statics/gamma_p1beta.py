from model.symbols import *
from model.shares import *
from comparative_statics.gamma_profit import dPi_dgamma
from comparative_statics.gamma_p1 import dsj1_dgamma_total
from comparative_statics.gamma_p2 import dsj2_dgamma_total

dp1dbeta_dgamma_raw = (
    delta * (sj2 * s01 * dsj2_dgamma_total * ds01_dp1 - sj1 * dsj1_dgamma_total)
) / alpha

dp1dbeta_dgamma_clean = sp.factor(sp.simplify(dp1dbeta_dgamma_raw))
