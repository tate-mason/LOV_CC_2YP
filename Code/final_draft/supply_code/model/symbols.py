"""Symbols for a fixed type and signed realized histories."""
import sympy as sp
alpha, beta, gamma, delta, c = sp.symbols("alpha beta gamma delta c", real=True)
p1, p2, p2j, p20 = sp.symbols("p_1 p_2 p_{2j} p_{20}", real=True)
q, q2 = sp.symbols("q q_2", real=True)
sj1, sj2, u, v = sp.symbols("s s_2 u v", positive=True)
s01, s02 = 1-sj1, 1-sj2
Vj, V0 = sp.symbols("V_{2j} V_{20}", real=True)
h0, h1 = sp.symbols("h_0 h_1")
p1_star_sym, p2_star_sym = sp.symbols("p_1^* p_2^*")
# Legacy names are signed coefficients now, never the former 0/1 indicators.
z1, z2 = q, q2
