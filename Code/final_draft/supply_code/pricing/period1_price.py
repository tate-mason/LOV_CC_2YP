import sympy as sp
from model.symbols import *

p1_star = c - (1 / (alpha * s01)) + (delta / alpha)(sj2 / s02)
p1_star_eq = sp.Eq(p1_star_sym, p1_star)
