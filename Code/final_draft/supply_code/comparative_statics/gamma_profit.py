from model.symbols import *

# Envelope derivative of Pi = sj1 * ((p1-c) + delta*sj2*(p2-c)).
dPi_dgamma = -sj1 * (z1 + delta * z2 * sj2) / alpha
