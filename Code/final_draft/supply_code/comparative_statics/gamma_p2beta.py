"""Gamma derivative of the period-2 beta comparative static."""
from model.symbols import *

dp2dbeta_dgamma = -z2 * sj2 * s02**2 / alpha
