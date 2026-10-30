from model.symbols import beta, gamma
from comparative_statics.canonical import price1_derivative, total_derivative
dp1dbeta_dgamma_raw = total_derivative(price1_derivative(beta), gamma)
dp1dbeta_dgamma_clean = dp1dbeta_dgamma_raw
