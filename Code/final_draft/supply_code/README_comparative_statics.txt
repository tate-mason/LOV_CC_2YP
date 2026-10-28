Validation: python3 run_all.py
Symbolic display: python3 comparative_statics_symbolic.py
Figures: python3 make_comparative_static_figures.py

See MODEL_CONVENTIONS.md for assumptions, price dependencies, corrections,
validation coverage, reproducibility, and suggested figure extensions.

Figures are saved as PNG and PDF under writeup_math_graphs/figures/, alongside solutions.csv.
Both pricing FOCs are re-solved at each gamma; all shares adjust endogenously.
The mixed derivative d(dp_j1/dbeta)/dgamma is plotted in
writeup_math_graphs/figures/beta_p1_derivative_by_gamma.png (and .pdf), with values in
the dp1_dbeta_dgamma column of solutions.csv.
The second gamma derivative is in writeup_math_graphs/figures/gamma2_p1_by_gamma.png (and .pdf).
Period-2 counterparts are gamma2_p2_by_gamma and beta_p2_derivative_by_gamma.
The figure set also covers both price levels, both prices' derivatives in
alpha/beta/gamma/delta, ratio derivatives in beta/gamma, and the profit
derivative in gamma. These curves use gamma in [0,3] on the horizontal axis.
period1_foc and period2_foc plot each pricing FOC against its own price at
gamma=1.5, holding the other price at its optimum; dots mark the solutions.
Their source data are in pricing_focs.csv. All figures have PNG/PDF versions.

LaTeX document with all figures and equations exported directly from SymPy:
  python3 make_latex_document.py
  latexmk -cd -pdf -interaction=nonstopmode -halt-on-error writeup_math_graphs/comparative_statics_figures.tex
The main source and sympy_equations.tex are in writeup_math_graphs/. Regenerating the document overwrites both .tex files.
The writeup defines all notation, suppresses the fixed product index j,
uses p_t and s_t, and writes outside-option probabilities as 1-s_t.
Both normalized and unnormalized pricing FOCs are displayed explicitly.

For alpha<0, positive prices and interior shares:
d(p1/p2)/dgamma is zero for (0,0), positive for (1,0), negative for (0,1).
The profit envelope is -s_j1*(z1 + delta*z2*s_j2)/alpha.
