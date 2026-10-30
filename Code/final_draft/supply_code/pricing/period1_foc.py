import sympy as sp
from model.symbols import *
from model.profit import DeltaV
foc1_raw = sj1 + alpha*sj1*s01*(p1-c) + delta*alpha*sj1*s01*DeltaV
foc1_collected = sj1 + alpha*sj1*s01*(p1-c+delta*DeltaV)
foc1_eq = sp.Eq(foc1_collected, 0)
