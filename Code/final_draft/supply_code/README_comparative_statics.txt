
Files included:

1. comparative_statics_symbolic.py
   - Uses SymPy.
   - Verifies:
       dp1/dgamma
       dp2/dgamma
       d/dgamma(p1/p2)
       dPi/dgamma
   - Prints all feasible case expressions and sign interpretations.

2. make_comparative_static_figures.py
   - Uses NumPy + Matplotlib.
   - Recreates:
       profit_by_gamma.png
       price_ratio_by_gamma.png
       price_ratio_derivative_cases.png

Important sign correction:
For (z1,z2)=(0,1),

d/dgamma(p1/p2)
=
s2 (p1 + delta s01 p2) / [alpha p2^2].

With alpha<0 and all prices/shares positive, this is NEGATIVE.

For (z1,z2)=(1,0),

d/dgamma(p1/p2)
=
-p2 s1 / [alpha p2^2]
=
-s1 / [alpha p2] > 0
when alpha<0.

The earlier generated graphic that labeled (0,1) as positive had the sign reversed.
