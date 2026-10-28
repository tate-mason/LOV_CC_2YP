"""Total second gamma derivative along both optimal pricing equations."""
from model.symbols import *

d2p1_dgamma2 = (
    delta * z2**2 * sj2 * s01 * s02**2
    - sj1 * s01**2 * (z1 + delta * z2 * sj2)**2
) / alpha
