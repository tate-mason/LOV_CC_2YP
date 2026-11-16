# Short presentation: current specification

Demand results use `estimates_1111/three/no_control_fit`: the November 11
three-flavor uncontrolled fit with average habit history. The presentation
assets are in `assets/demand_1111_three/`.

The pricing exercise quantifies the original binary theory with the new
estimated gamma distribution, population mean price sensitivity and flavor
intercepts. It compares common and informed pricing, holding habit at its zero
reference. All three focal flavors have results; plain is the main example.
All empirical comparisons use the estimated gamma distribution without alternative
low/middle/high probability scenarios. Display groups summarize price paths only.
Uniform prices depend on history but not gamma; discriminatory prices depend on both.
There is no observed-price benchmark or household context reconstruction.
See `Code/final_simple/demand/QUANTIFIED_THEORY.md` for the calculation command.

Both theory and quantified pricing now compare consumer surplus, profit, total
welfare, and expected binary purchases. Main slides show percentage changes from uniform pricing, dividing each change
by the absolute uniform level of that measure. Empirical revenue and welfare
appear together with matching inside/outside x-axes and distinct history colors; the appendix gives accounting
formulas and prior-by-history
tables. Welfare retains the original outside switching utility
`V0 = gamma*1{h=j}` and integrates over each policy's own future histories.
It is expected experienced welfare under myopic choice, conditional on the
parameter mapping and fixed-belief policy restriction. Aggregate CSVs also save
the alternative zero-outside utility convention as a sensitivity calculation.

## Appendix

Both deck sources read `seminar_pres/appendix_current.tex` (compile from the
seminar presentation directory). The appendix is organized into data/demand,
theory details, quantified-theory details, and next steps.

New demand material includes fit statistics, all parameter estimates with
standard errors and confidence intervals, distribution sign shares, price
coefficient transformation, WTP interpretation, average history and likelihood.
The pricing backup includes parameter mappings, prior definitions, berry and
other price paths, and the all-flavor revenue comparison.

Superseded price-control comparisons, full-demand pricing, and earlier
price-moment calibrations are excluded. Their assets remain on disk. Sources
from before this cleanup are saved in `seminar_pres/appendix_backup/`.

Compile the canonical deck from `Code/final_draft/seminar_pres`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
```

The second source is `Code/final_draft/writeup/shortened_presentation/pres_short.tex`.
Compile it from the same directory so shared assets and appendix resolve.
