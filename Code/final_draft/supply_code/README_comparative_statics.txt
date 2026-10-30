Symmetric switching, October 30 handoff

Run python3 run_all.py to verify the model and regenerate six requested figures,
source data, symbolic equations, and the LaTeX appendix.
Then compile:
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error writeup_math_graphs/comparative_statics_figures.tex

See MODEL_CONVENTIONS.md for assumptions, conditional signs, exact ratio histories,
public interfaces, verification, and superseded artifacts. Earlier entry-only
formulas and 0/1-history plots are historical and are not part of this report.
