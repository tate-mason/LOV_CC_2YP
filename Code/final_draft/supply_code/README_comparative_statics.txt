Validation: python3 run_all.py
Symbolic display: python3 comparative_statics_symbolic.py
Figures: python3 make_comparative_static_figures.py

See MODEL_CONVENTIONS.md for assumptions, price dependencies, corrections,
validation coverage, reproducibility, and suggested figure extensions.

Figures are saved as PNG and PDF under figures/, alongside solutions.csv.
Both pricing FOCs are re-solved at each gamma; all shares adjust endogenously.

For alpha<0, positive prices and interior shares:
d(p1/p2)/dgamma is zero for (0,0), positive for (1,0), negative for (0,1).
The profit envelope is -s_j1*(z1 + delta*z2*s_j2)/alpha.
