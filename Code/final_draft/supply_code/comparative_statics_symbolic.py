"""Print verified canonical gamma derivatives."""
from comparative_statics.canonical import *
if __name__ == "__main__":
    for label,expr in [("dp1/dgamma",dp1_dgamma),("d2p1/dgamma2",d2p1_dgamma2),
                       ("dprofit/dgamma",dPi_dgamma),("ratio derivative",ratio_derivative)]:
        print(label, "=", expr)
