
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Illustrative parameters
# -------------------------
alpha = -2.0
delta = 0.7

# For the simple comparative-static illustrations below,
# shares are held fixed to make the sign logic transparent.
s1 = 0.35
s2 = 0.40
s01 = 0.65

gamma_grid = np.linspace(0, 3, 200)

# ============================================================
# FIGURE 1: CONDITIONAL PROFIT BY FEASIBLE HISTORY
# dPi/dgamma = -(1/alpha) [ z1*s1 + delta*z2*s2 ]
# ============================================================

base_profit = 0.50

profit_cases = {
    "(0,0) customer stays": (0,0),
    "(0,1) customer leaves": (0,1),
    "(1,0) outsider joins": (1,0),
}

plt.figure(figsize=(9,6))

for label, (z1,z2) in profit_cases.items():
    slope = -(1/alpha)*(z1*s1 + delta*z2*s2)
    profit = base_profit + slope*gamma_grid
    plt.plot(gamma_grid, profit, linewidth=2.5, label=f"{label}, slope={slope:.3f}")

plt.xlabel(r"Switching preference $\gamma$")
plt.ylabel(r"Illustrative optimized profit $\Pi^*(h)$")
plt.title(r"Conditional Profit as $\gamma$ Changes")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("profit_by_gamma.png", dpi=180)
plt.close()

# ============================================================
# FIGURE 2: PRICE-RATIO COMPARATIVE STATIC
#
# d/dgamma (p1/p2)
# =
# [-p2*z1*s1 + z2*s2*(p1 + delta*s01*p2)]
# / [alpha*p2^2]
#
# To make a picture, we need actual p1(gamma), p2(gamma).
# Here we build them from the comparative-static slopes:
#
# dp1/dgamma = (-z1*s1 + delta*z2*s01*s2)/alpha
# dp2/dgamma = -(z2*s2)/alpha
#
# and choose starting prices p1(0), p2(0).
# ============================================================

p1_0 = 1.20
p2_0 = 1.00

ratio_cases = {
    "(0,0) customer stays": (0,0),
    "(0,1) customer leaves": (0,1),
    "(1,0) outsider joins": (1,0),
}

plt.figure(figsize=(9,6))

for label, (z1,z2) in ratio_cases.items():
    dp1 = (-z1*s1 + delta*z2*s01*s2) / alpha
    dp2 = -(z2*s2) / alpha

    p1 = p1_0 + dp1*gamma_grid
    p2 = p2_0 + dp2*gamma_grid

    ratio = p1/p2

    plt.plot(gamma_grid, ratio, linewidth=2.5, label=label)

plt.xlabel(r"Switching preference $\gamma$")
plt.ylabel(r"Optimal price ratio $p_{rj1}^*/p_{rj2}^*$")
plt.title(r"Optimal Price Ratio as $\gamma$ Changes")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("price_ratio_by_gamma.png", dpi=180)
plt.close()

# ============================================================
# FIGURE 3: DIRECTLY PLOT THE SIGN OF THE RATIO DERIVATIVE
# at chosen fixed p1,p2 and shares.
#
# This isolates the comparative-static formula itself.
# ============================================================

p1 = 1.20
p2 = 1.00

def dratio_dgamma(z1,z2):
    numerator = -p2*z1*s1 + z2*s2*(p1 + delta*s01*p2)
    denominator = alpha*(p2**2)
    return numerator/denominator

labels = []
values = []

for label, (z1,z2) in ratio_cases.items():
    labels.append(label)
    values.append(dratio_dgamma(z1,z2))

plt.figure(figsize=(9,6))
plt.bar(labels, values)
plt.axhline(0, linewidth=1)
plt.ylabel(r"$\partial (p_{rj1}^*/p_{rj2}^*)/\partial\gamma$")
plt.title(r"Sign of the Price-Ratio Comparative Static")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("price_ratio_derivative_cases.png", dpi=180)
plt.close()

print("Saved:")
print("  profit_by_gamma.png")
print("  price_ratio_by_gamma.png")
print("  price_ratio_derivative_cases.png")

print("\nCase-by-case ratio derivative:")
for label, (z1,z2) in ratio_cases.items():
    print(label, dratio_dgamma(z1,z2))
