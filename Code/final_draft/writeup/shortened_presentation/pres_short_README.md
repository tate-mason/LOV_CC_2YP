# Short presentation: current specification

Demand results use `estimates_1111/three/no_control_fit`: the November 11
three-flavor uncontrolled fit with average habit history. The presentation
assets are in `assets/demand_1111_three/`.

The pricing exercise quantifies the original binary theory with the new
estimated gamma distribution, population mean price sensitivity and flavor
intercepts. It compares common and informed pricing, holding habit at its zero
reference. All three focal flavors have results; plain is the main example.
There is no observed-price benchmark or household context reconstruction.
See `Code/final_simple/demand/QUANTIFIED_THEORY.md` for the calculation command.

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
