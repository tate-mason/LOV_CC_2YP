import sympy as sp

# Parameters

alpha, beta, gamma, delta = sp.symbols("alpha beta gamma delta", real=True)

# Cost

c = sp.symbols("c", real=True)

# Prices

p1, p2 = sp.symbols("p1 p2", positive=True)
p1_star_sym = sp.Symbol("p_{rj1}^*")
p2_star_sym = sp.Symbol("p_{rj2}^*")

# Shares

sj1, sj2 = sp.symbols("s_j1 s_j2", positive=True)

s01, s02 = sp.symbols("s_01 s_02", positive=True)

# Switching

z1, z2 = sp.symbols("z_1, z_2")

# Histories

h0, h1 = sp.symbols("h_0 h_1")
