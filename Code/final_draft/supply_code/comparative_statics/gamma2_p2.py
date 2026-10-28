"""Total second gamma derivative of the optimal period-2 price."""
from model.symbols import *

d2p2_dgamma2 = -z2**2 * sj2 * s02**2 / alpha
