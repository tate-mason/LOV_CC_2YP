# Committee update, November 1, 2026

`committee_update.tex` contains equations, derivations, notation tables, saved
estimates and figures. `committee_update.pdf` is the compiled version.
No estimation or supply optimization is run by this document build.

Sources:
- `shell_files/res_hess` in the repository parent: heterogeneous demand fit,
  capacity 7, 32 draws, 15,105 single-person households, 281,695 weeks;
  observed-information Hessian SEs and normal intervals.
- `final_draft/supply_code/writeup_math_graphs/comparative_statics_figures.tex`
  and its `figures/solutions.csv`: October 30 symmetric-switching supply model.

Price sensitivity is positive in this update: utility contains `-alpha*p`.
This replaces the source supply document's negative signed price coefficient.
Demand parameter `mu_log_minus_alpha` becomes mu_alpha; its numeric value does
not change. Supply continuation profits use W, distinguishing them from utility V.
Supply calibration is separate from demand estimates.

Rebuild from this directory:

```bash
python3 -B build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error committee_update.tex
```

`--results /path/to/res_hess` overrides the saved-results directory.
`--hhi /path/to/flavor_hhi_histogram.pdf` or `--hhi /path/to/yogurt_household_flavors.csv`
adds the new detailed-flavor HHI figure. The builder automatically finds these
filenames in `final_draft/data_code`. This must be the unaggregated-flavor output
from the main summary folder, rather than the three-category output from
`aggregated_flavors/`. No historical HHI figure is substituted.

Assets include snapshots of the estimates, covariance, first stages and supply
solutions. The builder checks Hessian SEs against the covariance diagonal and
normal intervals against the estimates and SEs. Bootstrap text is retained as
comments in the TeX source.

Summary-statistics tables transcribe the saved run log
`/Users/tate/LOV_CC_2YP/shell_files/summCols_49641717.txt`: full-panel and
matched-yogurt household summaries, detailed/grouped purchase dynamics, and
grouped package shares and transition counts. The descriptive matched sample
includes all household sizes and precedes the demand estimation filters. These tables are in the TeX source
and are retained when `build_assets.py` refreshes estimation assets.
