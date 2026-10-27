
import sympy as sp

# Symbols
alpha, delta = sp.symbols("alpha delta", nonzero=True)
z1, z2 = sp.symbols("z1 z2")
s1, s2, s01 = sp.symbols("s1 s2 s01", positive=True)
p1, p2 = sp.symbols("p1 p2", positive=True)

# Comparative statics already derived
dp1_dgamma = (-z1*s1 + delta*z2*s01*s2) / alpha
dp2_dgamma = -(z2*s2) / alpha

# Quotient rule for p1/p2
ratio_derivative = sp.simplify(
    (p2*dp1_dgamma - p1*dp2_dgamma) / p2**2
)

ratio_derivative_factored = sp.factor(ratio_derivative)

print("dp1/dgamma =")
sp.pprint(dp1_dgamma)

print("\ndp2/dgamma =")
sp.pprint(dp2_dgamma)

print("\nd/dgamma(p1/p2) =")
sp.pprint(ratio_derivative)

print("\nFactored =")
sp.pprint(ratio_derivative_factored)

# Expected compact result:
expected = (
    -p2*z1*s1
    + z2*s2*(p1 + delta*s01*p2)
) / (alpha*p2**2)

print("\nDifference from expected compact expression:")
sp.pprint(sp.simplify(ratio_derivative - expected))

# Case-by-case signs, assuming alpha < 0 and positive p1,p2,s1,s2,s01,delta
case_00 = sp.simplify(expected.subs({z1: 0, z2: 0}))
case_01 = sp.simplify(expected.subs({z1: 0, z2: 1}))
case_10 = sp.simplify(expected.subs({z1: 1, z2: 0}))
case_11 = sp.simplify(expected.subs({z1: 1, z2: 1}))

print("\nCases:")
print("(0,0):")
sp.pprint(case_00)

print("(0,1):")
sp.pprint(case_01)
print("With alpha<0 and positive prices/shares, this is NEGATIVE.")

print("(1,0):")
sp.pprint(case_10)
print("With alpha<0 and positive prices/shares, this is POSITIVE.")

print("(1,1):")
sp.pprint(case_11)
print("Sign is conditional, though this history is infeasible in your realized-path setup.")

# Profit derivative
profit_derivative = -(z1*s1 + delta*z2*s2) / alpha

print("\nProfit derivative:")
sp.pprint(profit_derivative)

print("\nProfit cases:")
for name, subs in {
    "(0,0)": {z1:0,z2:0},
    "(0,1)": {z1:0,z2:1},
    "(1,0)": {z1:1,z2:0},
    "(1,1)": {z1:1,z2:1},
}.items():
    expr = sp.simplify(profit_derivative.subs(subs))
    print(name)
    sp.pprint(expr)
