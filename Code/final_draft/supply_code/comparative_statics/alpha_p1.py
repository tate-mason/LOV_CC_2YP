from model.symbols import *

# Both optimal prices respond; the period-2 envelope condition applies.
dp1_dalpha = 1 / alpha**2 - p1 * sj1 / alpha + delta * s01 * sj2 * p2 / alpha
