import sympy as sp

from model.symbols import *
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.gamma_p1p2 import ratio_derivative

print("dp1/dgamma")
sp.pprint(dp1_dgamma)
print("\ndp2/dgamma")
sp.pprint(dp2_dgamma)
print("\nd(p1/p2)/dgamma")
sp.pprint(ratio_derivative)

cases = {
    "(0.0)": {z1: 0, z2: 0},
    "(1,0)": {z1: 1, z2: 0},
    "(0,1)": {z1: 0, z2: 1},
}

for name, subs in cases.items():
    print(f"\nCase {name}")
    sp.pprint(sp.simplify(ratio_derivative.subs(subs)))
