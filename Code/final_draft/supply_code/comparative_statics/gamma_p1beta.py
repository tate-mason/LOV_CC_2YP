from model.symbols import *
from comparative_statics.gamma_p1 import dsj1_dgamma_total
from comparative_statics.gamma_p2 import dsj2_dgamma_total

# Differentiate dp1/dbeta, including both direct and price-mediated share changes.
dp1dbeta_dgamma_raw = (
    delta * (s01 * dsj2_dgamma_total - sj2 * dsj1_dgamma_total)
    - dsj1_dgamma_total
) / alpha
dp1dbeta_dgamma_clean = sp.factor(dp1dbeta_dgamma_raw)
