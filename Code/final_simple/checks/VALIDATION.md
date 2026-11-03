# Verification of the simplification

The refactor was checked using generated data and calculation references.
No empirical CSV inputs were opened, no empirical estimates were recomputed,
and the original estimation/supply scripts were not replaced.

Completed checks:

- 36 synthetic checks pass with pandas/Polars CSV readers blocked. These cover
  histories and timing, prepared-Parquet sample construction, unavailable
  alternatives, persistent-type likelihoods, all seven analytic gradients,
  mixture nesting, price controls, optimization, household bootstrap refits,
  Hessian inference, WTP, empirical backward induction, and symbolic/numerical
  supply identities.
- Direct comparison with `final_draft` matches likelihoods and gradients for all
  28 combinations of seven estimation specifications, capacities 0/7, and price
  controls disabled/enabled.
- All empirical-pricing output columns match the prior batched solver to
  numerical tolerance under both type treatments, capacities 1/2/7, and revenue
  discount factors 0/.95, including missing competitors and ineligibility.
- All three theoretical supply cases and their outcome tables match at nine
  gamma/prior calibrations.
- The symbolic supply checks include the full 601-point grid for each of the
  two initial states: 1,202 numerical solutions, checked against the independent
  reference calculation, plus finite differences and independent optimization.
- The actual supply entry point ran with a short case grid and the complete
  derivative grid. It wrote all ten derivative figures and the symbolic appendix
  TeX source. A separate custom-grid run also wrote all figures with the
  benchmark-specific appendix disabled.
- All three numeric scripts expose their optional command-line help. Running
  pricing without selected empirical input paths stops before accessing data.

Two legacy checks reopen their own temporary synthetic CSVs. Their original
versions remain available but are excluded from `run_checks.py`. Equivalent
checks of bootstrap output and mixture metadata now capture DataFrames in memory
instead, and are included in the 36 passing checks.

Intentional organizational differences:

- Price-control, bootstrap, Hessian, and WTP functions live in `estimation.py`.
- The empirical price solver uses an explicit household loop. Output row order
  groups households together instead of the old batch/policy order; comparisons
  match rows by household and policy. `batch_size` controls progress intervals
  in the simple version; it no longer changes temporary household batch arrays.
  No empirical runtime or large-sample memory benchmark was performed.
- Mathematical equations are in `supply_sympy.py`; numeric solving and graphs are
  in `supply_simulation.py`. The latter evaluates the former's formulas.
- The run settings are ordinary editable Python variables. Optional command-line
  overrides are preserved, and readable `run_notes.txt` accompanies the automatic
  machine-readable records.
- Mathematical assumptions, price constraints, supported models, and inference
  choices are not silently substituted. In particular, the empirical stock model
  is not replaced with the binary theoretical switching model.

Reproduce the no-CSV checks:

```bash
python3 checks/run_checks.py
python3 checks/compare_original.py /path/to/final_draft
```

The comparison script needs the original code only for verification. The research scripts and their ordinary synthetic tests run independently of it.

After completing the equivalent refactor, the requested presentation change was
applied to both versions: empirical pricing now defaults to revenue discount
factor 0.95, and the revenue panel shows one bar per type for R1 + delta*R2.
The theoretical supply calibration remains 0.92.


## Data addition and demand/supply/data layout

The project now has `data`, `demand`, and `supply` folders. Imports, direct-script
entry points, package entry points, default demand/supply result paths, and the
checks were updated for that layout. The default data build prints its available
steps and accesses no data until steps are explicitly selected.

49 synthetic checks now pass, including 13 data checks. The new checks cover:

- Raw-reader options using an in-memory substitute, without opening text files.
- Column projections, longest-prefix matching, and preserving original files.
- Panel trip/household/product joins and retail version/store/flavor joins.
- Annual retail schema alignment and concatenation.
- Saturday dates, including Sunday and year boundaries.
- Yogurt module filtering and exact year/week/UPC/store matching.
- A complete synthetic Parquet merge/week/retail/matching run.
- Single-person and price filters before the two-week rule.
- Detailed and grouped HHI, switching, mixed baskets, and partition invariance.
- Flavor recovery retaining existing entries and selecting earliest valid labels.
- Preserving all-household retail coverage in the prepared demand input.
- Rendering HHI and other summary figures from generated data.

`compare_data_original.py` matches the original panel/retail assignment
expressions and all 40 exported summary tables plus the prepared purchase sample,
using both one and three household partitions. Old scripts with top-level raw
I/O are not imported: only their table calculation ASTs are evaluated on generated
tables. Old summary CSV intermediates are supplied from memory export captures,
so no CSV file is opened. The demand/supply original-vs-simple comparisons
continue to pass after the directory moves.

Data implementation changes are documented in `data/README.md`: the retail
merge/master-match path is made consistent, temporary year files are private to
the run, and summary calculations use Parquet checkpoints alongside the retained
CSV/LaTeX outputs. No empirical dataset was rebuilt or model re-estimated.
