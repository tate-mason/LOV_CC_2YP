# File-level teaching notes

## data_code/cleaning_merging/panel_convert.py

This script runs at module level: importing it can perform data work. First
the imports make filesystem and Polars tools available; path strings select
raw/server locations. Integer switches `annual` and `master` are truth-tested
to decide which conversion loops run. `range(2006, 2016)` includes 2006 through
2015, excluding the upper endpoint. Dictionaries translate source table names
to output stems. Nested loops construct one TSV/Parquet pair per year/table,
and `exists` prevents trying to read missing files.

After conversion, wildcard scans discover Parquet files. `columns_to_keep`
is a dictionary of table-specific projection lists, sometimes containing Polars
alias expressions. `split("_")[0]` extracts a table prefix before selecting its
projection. Scanning and sinking keep large tables out of ordinary Python lists;
eagerly reading every annual table would need much more memory. Lowercasing
column names standardizes downstream joins but does not reconcile every schema
difference or verify raw-codebook meaning. No CLI parser protects the hard-coded
paths; inspect them before running.

## data_code/cleaning_merging/panel_concat.py

This stacks annual cleaned panel tables into combined files. `concat_list` is
an ordinary list of three table names. In the loop, an f-string builds a wildcard
such as `cleaned_trips_*.parquet`. A lazy scan over those files followed by
`sink_parquet` writes one combined table. It does not add rows by matching keys:
concatenation stacks records, whereas a join combines columns through matches.
There is no deduplication or explicit year reconciliation here. A stale matching
file is therefore part of the input unless the directory is audited.

## data_code/cleaning_merging/panel_merge.py

Purchase lines anchor the merge. Join-key casts align integer dtypes before an
inner trip join, then left joins attach annual household and UPC product
metadata. Method chaining expresses a sequence of relational operations whose
result is sunk to `merged_panel.parquet`. An inner join drops unmatched purchase
lines; a left join retains them but may add missing metadata. Duplicate right
keys can multiply rows in either case.

The second half scans the output and prints diagnostics. `collect().item()`
extracts a scalar from a one-cell query result; `to_dict(as_series=False)` makes
a dictionary of Python lists for null-count reporting. `count[0]` picks its only
row. File byte size divided by `1024**3` expresses GiB. The empty-output check
raises, but a nonempty result is not proof of correct joins. Explicit key
validation would be a stronger alternative to relying only on total row ratios.

## data_code/cleaning_merging/panel_week.py

This adds a Saturday-ending integer key to the merged panel. The nested
expression parses purchase-date strings, extracts weekdays, computes a duration,
adds it, formats `%Y%m%d`, and casts to Int64. Read inside out. With Polars'
Monday=1 through Sunday=7 convention, `6-(weekday % 7)` gives Monday five days,
Saturday zero, and Sunday six days. It agrees with the downstream modulo-seven
formula over those weekday values.

The query writes a separate temporary file before `os.replace` replaces the
original. Writing directly to the source path while scanning it could corrupt
the read; staging then replacement avoids that pattern. It still overwrites
the original file after success, so this is not an inspection-only script.

## data_code/cleaning_merging/retail_convert.py

This is the retail analogue of panel conversion. Multiple mapping dictionaries
separate annual metadata, movement modules, and master files. All are ordinary
Python loops around lazy TSV reads and Parquet writes. `quote_char=None` disables
CSV-style quote interpretation for TSV inputs. Schema inference inspects a
prefix, so later inconsistent values may still need attention.

Cleaning keys are sorted by descending length, and a boundary-aware prefix test
distinguishes `products_extra` from `products`. This is a concrete reason for
the `sorted(..., key=len, reverse=True)` choice: testing the shorter prefix
first could select the wrong projection. `continue` skips already cleaned
files; `break` stops the key-search loop after a match. A `try/except` around
schema inspection reports unreadable files without aborting all inspection.

## data_code/cleaning_merging/retail_concat.py

This scans annual retail outputs, harmonizes selected dtypes, and vertically
concatenates tables. `glob.glob` returns matching filenames; `sorted` supplies
deterministic order. A missing match list triggers `continue`, skipping just
that target rather than the whole script. The list comprehension creates casts
only for columns actually present in a particular schema.

`how="vertical_relaxed"` allows compatible dtype coercion when stacking frames;
it is not permission to ignore arbitrary incompatible schemas. Explicit casts
for flavor/size codes and join IDs reduce annual type conflicts. Keeping codes
as strings can preserve nonnumeric categories that an integer cast would reject.

## data_code/cleaning_merging/retail_merge.py

This concatenates movement modules and uses them as the retail anchor. Integer
floor division by 10000 extracts a calendar year from a YYYYMMDD integer. Several
metadata tables are deduplicated before left joins to avoid multiplying movement
rows. `suffix` names colliding non-key columns, while composite `on=[...]`
lists specify exact matching keys.

First-row deduplication is a policy, not an inference of the correct UPC version.
Extra-product sorting prioritizes nonnull flavor descriptions before retaining
a UPC/year row. The diagnostics compare anchor and merged row counts and inspect
nonnull metadata. A conditional expression chooses which brand field is available
for a match diagnostic, so that diagnostic may inspect a source key rather than
a successful dictionary match. Output is under `Data/RMS/`, unlike the current
master script's `Data/merged_rms.parquet` input path.

## data_code/cleaning_merging/master_dat.py

This matches household purchases to retail quotes. The four-element key list
is reused for projection, deduplication, and joins, reducing accidental differences
between stages. The filtered retail yogurt table is collected once into memory;
panel purchases are scanned and joined year by year to bound panel-side work.
An inner join means a price match is a sample-selection condition.

Yearly results are staged as Parquet and scanned together into the final sample.
`os.remove` deletes the generated temporary files and `os.rmdir` removes their
directory after consolidation. These are real filesystem mutations. Final
diagnostic queries return scalar counts or small tables. `next(..., None)` picks
supported household/price names rather than assuming a suffix. Building all
years eagerly in one pandas merge would be shorter but much more memory-hungry.
Resolve its retail input path against the retail merge's actual output before
running a fresh pipeline.

## supply_code/model/symbols.py

`sp.symbols` constructs formal mathematical objects and tuple unpacking gives
them Python names. The name displayed in equations can differ from the Python
identifier, e.g. `p1` represents printed `p_1`. `real=True` records a symbolic
domain; `positive=True` on shares does not also encode share below one. Economic
restrictions such as alpha<0 remain assumptions used in interpretation and
numeric validation. `s01 = 1-sj1` defines an expression, not another independent
symbol. `z1,z2 = q,q2` aliases existing signed symbols for legacy code.

## supply_code/model/assumptions.py

This dictionary stores human-readable assumptions as strings. Python and SymPy
do not enforce alpha<0 just because a dictionary says so. It serves documentation;
the numerical evaluator performs explicit checks. Distinguish descriptive
metadata from executable validation.

## supply_code/model/shares.py

This module builds symbolic logit shares from normalized utility and defines
derivative expressions. The star import makes shared symbols available directly
without a module prefix, which shortens formulas but makes their origins harder
for beginners to track. Expressions such as `alpha*sj1*(1-sj1)` encode the
logistic price derivative using share placeholders. They are algebra, not
evaluated probabilities for a dataset. The explicit-share expression and the
share symbol are intentionally distinct objects.

## supply_code/model/profit.py

These assignments construct symbolic current and continuation profit. `pi2`
weights both optimized terminal values by first-period share and outside share.
`Pi = pi1 + delta*pi2` creates total discounted profit. Capitalization matters:
Python treats `Pi` and `pi1` as unrelated names. No optimizer runs in this file;
it provides the expression that FOC modules and tests inspect.

## supply_code/pricing/period1_foc.py

This module creates equivalent raw and collected forms of the dynamic first
derivative. Factoring `(p1-c+delta*DeltaV)` displays the effective-margin meaning.
`sp.Eq(expression, 0)` constructs a symbolic equation object; it does not solve
that equation or assign zero to the expression. Keeping an equation for export
beside an expression for differentiation avoids manually retyping formulas.

## supply_code/pricing/period1_price.py

`p1_star` is an implicit markup expression containing an equilibrium-share
symbol and both continuation values. `sp.Eq` labels it with a formal starred
price for display. It is not a numerical optimal price until those endogenous
objects are solved consistently. Direct substitution of an arbitrary share
would not solve the model.

## supply_code/pricing/period2_foc.py

The terminal FOC contains current share plus margin times its price derivative.
There is no discount term because period 2 is terminal. `sp.Eq` wraps it as an
equation for presentation; actual root solving happens in `model/numerical.py`.
Separating the expression permits tests to compare it with a literal derivative
of profit instead of trusting a displayed equation alone.

## supply_code/pricing/period2_price.py

The markup expression is cost minus `1/(alpha*s02)`, positive above cost for
negative alpha and interior shares. Parentheses make the whole product the
denominator: `1/alpha*s02` would mean `(1/alpha)*s02`, a different formula.
This is another implicit representation because outside share depends on price.

## supply_code/comparative_statics_symbolic.py

This convenience script imports canonical expressions and, only when run
directly, iterates over `(label, expression)` pairs and prints each. Tuple
unpacking connects a human-readable label to its actual symbolic object.
It does not compute independent derivative formulas. Reusing named expressions
keeps terminal display consistent with production numeric evaluation.

## supply_code/run_all.py

`Path(__file__).resolve().parent` locates this script's directory regardless
of the shell's current directory. A list of command-argument lists describes
the three reproducibility steps. `[sys.executable, *command]` uses the current
Python interpreter and expands each command's arguments without a shell.
`cwd=ROOT` makes relative imports and outputs predictable; `check=True` raises
if a step exits unsuccessfully, preventing later artifact generation after a
failed check. Running a shell string with semicolons would be less portable
and would require additional quoting/exit-status care.

## est_code/est_naive.py

This wrapper imports `main` and calls it with the `naive` default only under
the entry-point guard. The wrapper contains no new estimation algorithm.
`--model` can still override the supplied default because the common parser
processes the user's options. Reusing one main function avoids four copies of
data preparation and CLI validation.

## est_code/est_gamma.py

This wrapper selects the `homogeneous` default, meaning common alpha and common
unrestricted gamma. It imports and calls the shared `main`; it does not define
a function locally. The name of a script is not itself a Python function or
an instruction to fit a different formula unless its actual call says so.

## est_code/est_het_naive.py

This guarded wrapper supplies the `mixed_naive` default. The common estimator
then includes price heterogeneity while excluding gamma. The small file is a
convenient command name and shares all data, optimizer, and inference options
with the main script.

## est_code/est_het_gamma.py

This wrapper supplies the `heterogeneous` default to shared `main`. Both alpha
and gamma vary by persistent type. Parentheses call the imported function;
writing just `main` would retrieve the function object without executing it.

## supply_code/model/utility.py

After defining its two helpers, module-level code calls `raw_utilities` for
each period, unpacks inside/outside expressions, and subtracts outside utility.
`sp.expand` distributes products to obtain the normalized linear index.
This module combines function definitions with actual symbolic setup work
at import time. It does not read private data or solve numeric prices.

## supply_code/model/numerical.py

Module-level tuples `_ARGS`, `_EXPRESSIONS`, and `_KEYS` establish the ordered
bridge from symbolic algebra to named numerical derivative results. A leading
underscore is a convention for an internal name, not access control.
`sp.lambdify` executes once at import, avoiding repeated compilation of the
same symbolic formulas for all 1,202 benchmark evaluations. The functions below
reuse that callable. Positional order, rather than matching name spelling at
runtime, connects its inputs.

## supply_code/comparative_statics/canonical.py

Functions provide generic derivative rules; module-level assignments call them
to define named gamma, beta, curvature, and ratio expressions. The quotient-rule
ratio is created once with generic terminal symbols, then `.subs({...})`
specializes it to matched-inside, matched-outside, and acquisition histories.
This keeps three ratio definitions tied to one algebraic rule instead of three
independent hand-written implementations.


## est_code/history_distributions.py

Saved-fit distribution exporter. Read this when you want the two estimated population marginals without rerunning optimization. It reads fit metadata and coefficients, not household histories. The zero-SD case is a point mass. The reusable function is called automatically after a split fit.


## est_code/regenerate_weekly_inputs.py

Server-only workflow for the deleted household export. Saved settings reconstruct the original sample/history/menu table and optional first stages. Exact household/week totals and likelihood agreement are required before writing weekly_inputs.csv. Demand estimates and inference are preserved.


## pricing_exercises/run_split_targeting.py

Current original split-model empirical pricing implementation. It recovers alpha, novelty and own-stock types; ranks by novelty; retains both effects in demand; and runs all three focal flavors with fixed rivals. This file uses household batches; the readable final_simple version loops one household at a time with equivalent results.


## pricing_exercises/run_empirical_targeting.py

Earlier combined-gamma berry-only pricing comparison. It does not accept the split fit. Use run_split_targeting.py or final_simple/demand/split_pricing.py for the current model. Its .95 revenue discount default differs from the new split exercise's .92.


## pricing_exercises/run_supply_exercises.py

All three theoretical information cases: known initial history, hidden initial history, and advance commitment to two prices. The binary symmetric switching model is a mechanism benchmark; it is not the cumulative-stock split demand likelihood. Both continuation branches and unrestricted theoretical prices are retained.


## pricing_exercises/run_pricing_exercises.py

Compatibility launcher forwarding to the earlier supply or combined-gamma exercise. The current split workflow has a separate explicit entry point and is not selected by the old households branch.


## ../final_simple/demand/estimation.py

Readable demand estimator: editable settings, numbered sections, and visible run_estimation steps. Current defaults are heterogeneous_history and capacity 7. Section 3 contains split utility and persistent-type likelihood; section 6 exports population novelty/habit marginals. All supported earlier models remain available. No research functions are imported from final_draft.


## ../final_simple/demand/split_pricing.py

Recommended current pricing reading path. The parameter section selects all three focal flavors; sections 1–2 recover persistent types and initial history; section 3 visibly solves one household at a time; section 4 permutes/validates each focal case; run_pricing writes separate folders and a combined comparison. It retains habit even though reporting groups rank novelty alone.


## ../final_simple/demand/regenerate_weekly_inputs.py

Readable server reconstruction entry point. It imports preparation/likelihood functions from the local estimation module, avoiding a second demand implementation. Edit paths only on the server. Consistency failures stop before replacing the export; successful regeneration leaves fitted demand coefficients unchanged.


## ../final_simple/demand/history_distributions.py

Small saved-fit distribution entry point using the export calculation in the readable estimator. Edit fit_dir or use --fit-dir. It exports population density/CDF grids, quantiles/sign shares and plots without reading a weekly household table.


## ../final_simple/checks/test_split_pricing.py

Generated-array checks of split posterior likelihood, exact binary aggregation, timing, independent-coefficient backward induction, and all focal cases under both type treatments. The final_simple check runner blocks CSV readers.


## ../final_simple/checks/test_regenerate_weekly_inputs.py

Mocked regeneration checks. Synthetic records supply saved settings and parameters; mocked preparation/CSV loading allows the checks to run while CSV readers are blocked. Success preserves fitted artifacts and a likelihood mismatch preserves any old export.
