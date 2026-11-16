# Short presentation: current specification

Demand results use `fixed_est/refined_128`: the corrected observed-shopping-week
three-flavor uncontrolled fit with average purchase-share habit history. It
includes 15,752 households and 2,034,618 weeks, of which 1,792,368 have zero yogurt
purchases. The presentation assets are in `assets/demand_observed_128/`.

Pricing assets are in `assets/quantified_theory_observed_128/`. Rebuild them and
the aggregate-only demand tables from the repository root with:

```bash
python3 Code/final_simple/demand/quantified_theory.py \
  --estimates fixed_est/refined_128/point_estimates.csv \
  --output Code/final_draft/seminar_pres/assets/quantified_theory_observed_128
python3 Code/final_draft/seminar_pres/refresh_observed_week_assets.py
python3 Code/final_draft/seminar_pres/rebuild_price_scales.py
```

For plain, the corrected plug-in exercise gives revenue changes of +1.9% / +3.4%,
consumer-surplus changes of -7.3% / -11.4%, and total-welfare changes of +0.6% /
-5.9%, for initially inside / outside consumers. Percentage denominators use
absolute uniform-pricing baselines. The 32-to-128-draw sensitivity is reported
in the appendix; convergence alone does not establish integration stability.
Original descriptive purchase-sample tables are labeled separately from demand.
The old November 11 assets remain in their existing directories.
Panels within each price figure share y-axis limits; separate figures choose
their own ranges. Welfare panels may use independent y-axis ranges.

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
