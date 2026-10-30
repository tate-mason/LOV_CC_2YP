# Canonical symmetric-switching model

This document supersedes the October 28 entry-only conventions. The October 30
handoff is implemented in the existing model/, pricing/, comparative_statics/,
checks/, and export/ structure.

For each fixed observed type, raw utilities are beta + alpha*p + gamma*1(previous=0)
inside and gamma*1(previous=j) outside. Subtracting outside utility yields
beta + alpha*p + q*gamma, with q(j)=-1 and q(0)=+1. Both period-2 states are solved.
Period-1 profit is s*(p1-c) + delta*(s*V2(j)+(1-s)*V2(0)). Branch values are constant
with respect to p1 conditional on the realized state, but change with gamma.

Assumptions: alpha<0, 0<delta<1, fixed cost and type, interior logit shares,
unconstrained real prices. The separately solved delta=0 case is a static
counterfactual. The dynamic price's static FOC component is not that counterfactual.

All production comparative statics originate in comparative_statics/canonical.py.
The numerical solver evaluates those SymPy expressions rather than duplicating
formulas in plotting scripts. model/numerical.py exposes evaluate(gamma, initial_state,
...), where initial_state is 'j' or '0'. solve() returns p1,p2j,p20,s,u,v,profit.
The old z1,z2 numeric interface raises a migration error; symbolic z aliases now
mean signed q, not the former 0/1 coefficients.

The handoff's structural_switching_reference.py is retained unchanged as an
independent verification oracle, not a production model or plotting entry point.
Full benchmark agreement, profit-derived symbolic FOCs, envelope identities,
implicit derivatives, finite differences of reoptimized solutions, numerical
profit maximization, and subsidy/static regressions are checked by run_all.py.
Both alpha/beta/delta derivatives and mixed beta-gamma derivatives are migrated;
only the requested gamma plots are generated.

## Reproduce

    python3 run_all.py
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error writeup_math_graphs/comparative_statics_figures.tex

Benchmark: beta=2, alpha=-1, delta=.92, c=0; 601 gamma points in [-3,3], each
initial state. Six figures: matched ratio derivatives together; induce/harvest
ratio derivative separately; profit; profit derivative; both periods' price
levels; dp1/dgamma. Files are PNG/PDF with the full solutions.csv.

The appendix proves the conditional signs. No common sign is asserted across
all initial histories. A matched ratio is not an ex-ante ratio; induce/harvest
conditions on acquisition. No type mixture or pooled pricing variant is added.

## Superseded results

The old entry-only utilities, one-branch profit, 0/1-history figures, positive-price
restrictions, and old gamma/profit/ratio formulas are superseded. In particular,
at gamma=0 continuation values coincide, so the new dynamic p1 equals the true
static price. The old profit incorrectly rewarded continuation only after an
inside purchase. At the new benchmark, gamma=-3 allows p1(0)<0. Profit's gamma
derivative changes sign for initially inside consumers. Old global ratio-sign
claims no longer apply. Calibration and history definitions also changed, so
old and new curves cannot be compared as a change of calibration alone.

Older figure files not among the six requested outputs are retained as historical
artifacts. They are not regenerated or included in the current PDF. The old
pricing_focs.csv is also historical; current FOC residuals are in solutions.csv.
DERIVATIVE_CHANGES.md and RUN_ALL_DIAGNOSIS.md contain historical diagnoses only.
