"""Display the checked modular formulas; run_all.py performs validation."""
import sympy as sp
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.gamma_p1p2 import ratio_derivative
from comparative_statics.gamma_profit import dPi_dgamma
from model.symbols import z1,z2

if __name__ == '__main__':
    for name, expr in [('dp1/dgamma',dp1_dgamma),('dp2/dgamma',dp2_dgamma),
                       ('d(p1/p2)/dgamma',ratio_derivative),('dPi/dgamma',dPi_dgamma)]:
        print(name)
        sp.pprint(expr)
        for history in [(0,0),(1,0),(0,1)]:
            print(history,sp.simplify(expr.subs(dict(zip((z1,z2),history)))))
