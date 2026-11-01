# Function-specific teaching notes

These notes are inputs to `build_python101.py`. Each heading identifies a source
file and function, including its enclosing function when it is nested. The book
combines these explanations with the actual source and syntax notes.

## est_code/est.py::normalize_upcs

**Job and contract.** Accept a pandas Series of UPC-like values and return a
Series of canonical string identifiers. `values.astype("string")` selects
pandas' nullable string dtype; `.str` applies a string operation to every
element. `.strip()` removes surrounding whitespace. `numeric` is a Boolean
Series, not a single true/false answer.

`fullmatch` requires the entire string to fit the numeric pattern. Replacement
removes a trailing decimal-zero suffix, then `lstrip("0")` removes leading zeros.
The last `where` keeps the original text where `~numeric` is true and uses the
canonical version elsewhere. Thus `"00123.0"` becomes `"123"`, `"000"` becomes
`"0"`, and `"A001"` stays `"A001"`. The inner `.replace("", "0")` fixes the
empty string produced when all digits were zero.

**Why this choice?** Converting to `float` then `int` looks shorter but can round
long identifiers and fails on nonnumeric UPCs. A Python loop would work but
duplicates the same string operations per row and loses convenient missing-value
handling. `where` is easy to misread: its true branch retains the receiver.

## est_code/est.py::normalized_upc_expr

This returns a **Polars expression**, not a loaded Series. Calling it constructs
a reusable recipe for the `upc` column. `pl.col` identifies that column;
`cast(pl.String)` standardizes its type. The local names `text`, `numeric`, and
`canonical` hold expression objects. Their calculations run later when the
surrounding lazy query is executed.

Nested `when/then/otherwise` first decides whether a UPC is numeric, then whether
stripping zeros would leave an empty identifier. `pl.lit("0")` means literal
text, not a column named zero. `.alias("upc")` ensures `with_columns` replaces
the intended column. This version avoids collecting an entire lazy table just
to normalize one field. A pandas conversion would forfeit the lazy query's
memory and projection advantages.

## est_code/est.py::read_flavors

Input `path` is a CSV path; optional `products_extra` is a Parquet path. Output
is a two-column pandas table with one row per UPC. This function has an early
return: if the CSV already contains resolved `upc,flavor`, validation succeeds
and the rest of the recovery/classification code is never executed.

`{"upc", "flavor"}.issubset(frame.columns)` asks whether both required labels
exist. `missing = required - set(frame.columns)` is set subtraction. Validation
uses `.any()` for "at least one invalid row" and `.all()` for "every label is
allowed." `if products_extra is not None` distinguishes an omitted optional
argument from a supplied path that happens not to exist.

The metadata path scans only useful columns and anti-joins existing reference
UPCs. It appends absent UPCs, rather than overwriting existing classifications.
`[... for c in columns if c != "upc"]` builds one aggregation expression per
non-key field. `np.select` picks the first matching condition, so plain wins
when both plain and berry text occur. A generic substring loop could be easier
to write at first, but would make precedence and vectorized missing-value
handling less explicit. `drop_duplicates(..., keep="first")` is a reconciliation
rule; it does not prove the retained label is correct.

## est_code/est.py::week_dates

`pd.to_datetime` interprets strings with an explicit `%Y%m%d` format: four-digit
year, two-digit month, two-digit day. The regex removes a trailing `.0` from
integer dates that were stored as float-like text. `errors="raise"` rejects a
malformed date instead of silently creating a missing date.

For example, `20100109` should mean January 9, 2010. Passing that integer
directly to a general timestamp constructor can instead mean nanoseconds from
an epoch. Explicit parsing prevents that mistake. This helper exists in the
module, but the current main preparation path parses purchase-date strings
with Polars directly; a function's presence does not mean the entry point uses it.

## est_code/est.py::prepare_data

This is the data assembly function. `args` is an argparse Namespace: a small
object whose attributes hold command-line settings. It returns a pandas table
with one row per usable household-week. Its work crosses a deliberate boundary:
large input filtering and quote aggregation use lazy Polars, while the smaller
household choice table becomes pandas for subsequent array calculations.

Read it in phases. First it scans and filters years, resolves labels, chooses
the price field, and projects columns. `getattr(args, name, default)` provides
compatibility when a manually constructed Namespace lacks an optional attribute.
`next((c for c in candidates if c in names), None)` chooses the first supported
price field. It is a small loop written as a generator, not a special price API.

Next it collects selected households, checks quantities, sums counts, and chooses
an exposure store. Sorting by descending quantity followed by ascending store ID
then keeping the first household-week row implements the tie-break exactly.
`groupby(...).sum().unstack(...)` turns long flavor counts into three columns;
`reindex` fixes column order even if one flavor is absent from an intermediate
table. Merge `validate` arguments catch unintended duplicate keys.

The quote section continues to use full `base`, not the household-limited
`selected`. It deduplicates quotes and checks conflicts before constructing
menus. The DMA section computes sums and counts and subtracts own-DMA totals.
This algebra avoids looping over every possible outside DMA for every household.
The same pooled IV is copied into all three flavor-named IV columns.

Finally it constructs histories **before** menu/IV likelihood exclusions,
builds Boolean availability masks, filters unusable rows, reapplies the two-week
rule, and adds outside counts. `complete &= iv_complete` combines masks in
place; it is not a comparison. `transform("nunique")` produces a row-aligned
count so the household restriction can be applied without a second merge.
Moving `add_stocks` below these exclusions is a plausible-looking alternative
that changes the information used to build history.

## est_code/est.py::add_stocks

Input is a weekly pandas table; output is a sorted copy with stock, novelty,
and Xi columns. The outer loop resets history for each household. `for _, group`
ignores the household label but keeps its subtable. The inner `iterrows()` loop
supplies a row index and a Series containing that row's values.

`stocks[index] = stock` stores pre-choice history into one output row. The
seven-day condition tests genuine adjacent calendar weeks. Then counts are
normalized into shares, stock is updated, and the current date becomes the
previous date. A gap resets recent novelty by leaving its preallocated zeros,
but does not reset cumulative stock. A loop is reasonable here because state
at the next observation depends on the preceding update; vectorization requires
careful shifted cumulative sums and calendar-gap checks.

The final `enumerate` loop writes one named column per flavor. `stocks[:, j]`
extracts all rows of the jth flavor. With `habit_weight=.7`, current Xi uses .3
times novelty minus .7 times own stock. The separately preallocated `stocks`
matrix means historical output rows do not change when `stock` is reassigned.
Try a one-household, two-week example before reading the likelihood.

## est_code/est.py::make_arrays

This converts a labeled table into a dictionary of NumPy arrays. `pd.factorize`
returns both dense integer household codes and their original labels: household
IDs `[10,10,30]` become codes `[0,0,1]`. Dense codes let `bincount` sum efficiently.
The expression `FLAVORS + (["outside"] if capacity else [])` builds either three
or four outcome columns without changing the inside-price array's three columns.

The function validates counts/Xi, checks menu consistency, and inserts zero
placeholders for missing prices. The separate availability mask ensures those
zeros are not interpreted as offered free goods. `all(c in frame for c in cf_cols)`
asks whether every residual column exists. If controls are absent, `zeros_like`
supplies the right-shaped residual array. The returned dictionary deliberately
keeps flags such as `outside` and `control` beside arrays so the likelihood can
use one interface for several models. Passing raw tables directly into every
optimization step would repeatedly perform label lookup and conversion.

## est_code/est.py::make_draws

Inputs are an even draw count and an integer seed; output has shape `(number,2)`.
`number % 2` is the remainder after division by two: zero for even numbers,
one for odd. `number // 2` is integer division. The generator draws half the
rows, then `vstack([z, -z])` appends their negatives. With two draws, an initial
row `[.4,-1.2]` gives a second row `[-.4,1.2]`.

The seed makes repeated calls reproducible, and a local Generator avoids
changing NumPy's global random state. Reusing these fixed antithetic nodes
reduces noise when comparing candidate parameter vectors. Drawing afresh inside
each objective call would make the optimizer chase random changes as well as
parameter effects. More draws still cost more computation.

## est_code/est.py::parameter_names

This returns the ordered labels defining the parameter vector. `names += [...]`
extends a list; it does not numerically add its string entries. Membership tests
such as `model in ("mixed_naive", "heterogeneous")` let related specifications
share branches. The default `control=False` omits a residual coefficient unless
the caller explicitly enables it.

For naive conditional demand without controls, the returned labels are two
betas and mean log-minus-alpha. Heterogeneous demand adds mean gamma and both
SDs; the optional outside constant is prepended and price control appended.
Keeping this logic in one helper is safer than independently remembering vector
positions in optimization, utility, and output. The order is part of the API:
zipping a reordered list with old coefficients silently changes their meaning.

## est_code/est.py::household_loglike

Input `theta` is a one-dimensional parameter array, `data` is the dictionary
from `make_arrays`, and `draws` is a draw-by-two matrix. The return is one log
likelihood per household, not a scalar. `dict(zip(...))` attaches coefficient
names so the rest of the function need not remember their numeric positions.

For each latent-type draw, negative exponential constructs alpha and an affine
normal transformation constructs gamma. Broadcasting makes beta and these
scalar coefficients operate across every row/flavor. Unavailable utility is
set to minus infinity; an optional zero outside column is appended.
`logprob = utility - logsumexp(..., axis=1, keepdims=True)` normalizes each row.
The retained column dimension makes subtraction broadcast across that row.

The count-weighted row sum uses `where(counts > 0, logprob, 0)` to avoid zero
times minus infinity. `bincount` adds all weeks of the same household. Only
after this aggregation does the final `logsumexp` integrate draws. This order
implements persistent preferences. Averaging weekly probabilities first would
be simpler computationally but would estimate a different preference process.

## est_code/est.py::demand_objective_gradient

This returns a tuple `(mean_negative_log_likelihood, gradient)`, which SciPy
requests when `jac=True`. The utility and likelihood steps mirror
`household_loglike`; the additional work computes derivatives in the same draw
loop. `total[:, None]` expands a weekly count vector to a column so it can
multiply the matrix of choice probabilities. `residual` is observed count
minus predicted count for the three inside alternatives.

The `columns` dictionary holds one weekly score vector per coefficient name.
For example, the berry intercept takes column 1 of residual, while the alpha
location score uses the price-weighted residual multiplied by alpha itself
because differentiating `-exp(mu + sd*z)` contributes alpha. Each score is
aggregated to households, and `column_stack` places parameters on a final axis.

`np.einsum("rh,rhp->p", weights, scores)` means multiply posterior weight at
draw r/household h by its parameter-p score, then sum r and h, retaining p.
A nested loop over r, h, p would express the same sum more verbosely and slowly.
The minus sign and household-count division match minimization of average NLL.
This is an analytic derivative of the finite-draw objective, not automatic
differentiation or a finite-difference estimate.

## est_code/est.py::optimize_demand

This sets up and runs multiple bounded local searches and returns the selected
fit, every starting solution, and the bounds. Initial SDs are .3 because zero
can be stationary under symmetric integration. Bounds are pairs `(lower, upper)`
in the same order as parameter names. The optional initial vector is copied
before perturbation, preventing caller-owned coefficients from being changed.

Each start uses `minimize(..., method="L-BFGS-B", jac=True)`. The callback reports
progress every ten iterations, while completed starts are saved to JSON.
`np.clip` keeps perturbed starts feasible. `min(results, key=lambda r: r.fun)`
uses each result's loss as its comparison key; it selects the smallest loss,
not necessarily a converged fit. An alternative is to select only converged
results, but that is not the rule this code implements. Multistart search probes
local solutions and does not establish a global optimum.

## est_code/est.py::optimize_demand.objective

This small nested adapter has one explicit input, `theta`. It forwards theta
and the enclosing call's data/draws/model to the actual objective/gradient
function. SciPy expects a callable accepting the changing candidate vector;
the closure supplies the fixed objects. A global variable for data could work
but would make concurrent or repeated fits harder to reason about. Returning
the whole pair is intentional because the surrounding minimizer sets `jac=True`.

## est_code/est.py::optimize_demand.progress

SciPy calls this nested callback after iterations, passing the current vector.
The body increments `iterations[0]`, checks a remainder modulo ten, and prints
elapsed time unless quiet mode is active. The callback does not return an
objective value and does not choose the next parameter vector.

The enclosing variable is a one-element list so the nested function can mutate
its element without rebinding the outer name. An ordinary integer counter with
assignment would require `nonlocal iterations`. That alternative is often clearer;
the existing list technique works because objects can be mutated through a
captured reference. `time.monotonic` measures elapsed duration without wall-clock
adjustments changing the result.

## est_code/est.py::plot_gamma_distribution

Input is an estimate table, output directory, and convergence flag. This is
a side-effect function: it saves a figure and implicitly returns `None`.
`set_index("parameter")["estimate"]` creates a name-indexed Series for coefficient
lookup. The early bare `return` skips naive models with no gamma.

For positive SD, `linspace` makes a smooth horizontal grid, `norm.pdf` computes
density, and `fill_between(..., where=x < 0)` shades the negative region. For
zero SD a continuous density would divide by zero or misrepresent a point mass,
so the `else` branch draws a vertical line. `norm.cdf(0, ...)` computes the
implied population habit share. `Agg` supports file rendering without a GUI;
`plt.close(fig)` releases the figure. This is a distribution of latent types,
not an uncertainty interval for the mean.

## est_code/est.py::fit_model

This coordinates point fitting, reporting, optional inference, and saved files.
It turns `best.x` into a labeled table. The boundary list comprehension pairs
each coefficient with its bounds and tests distance below `1e-4`; flags do not
automatically change the estimate. Saving point estimates before bootstrap
protects the useful main result if a long inference job is interrupted.

The nested `estimate` callback supplies demand fitting to the generic bootstrap.
The bootstrap runs only after a converged main fit and with the original weekly
frame available. Missing inference columns are explicitly filled with NaN so
an absent SE cannot be mistaken for zero. `vars(args).items()` enumerates settings;
Path objects become strings so JSON serialization can encode them. The returned
`best` is still a SciPy result object, while CSV/JSON provide portable outputs.

## est_code/est.py::fit_model.estimate

This nested function accepts one resampled weekly table and a starting coefficient
vector. It rebuilds numerical arrays, runs demand optimization with bootstrap
start count, and returns only the selected fit. `fitted, _, _` discards other
optimizer outputs. It captures capacity and other settings from outer `args`.
Passing this callback avoids importing `est.py` back into `bootstrap_se.py`,
which could create a circular module dependency. Price controls are reestimated
by the bootstrap before this function is called, not inside this callback.

## est_code/est.py::main

This is the command-line entry point. `default_model` lets small wrapper scripts
reuse the entire interface while selecting a different default model. Each
`add_argument` describes a setting, conversion, default, or permitted choices.
The parser turns a flag such as `--no-price-control` into the attribute
`args.no_price_control`; hyphens become underscores in attribute names.

Validation runs before expensive data loading. `parser.error` prints usage and
exits for invalid settings. After parsing, execution is preparation, optional
first stages, weekly input export, array construction, and fitting. This division
lets tests construct a small `SimpleNamespace` and call internal functions
without pretending to run a shell command. Repeating this code in all four
entry scripts would allow defaults and validations to drift apart.

## est_code/price_control.py::check_dma

This converts a pandas Series to integer DMA IDs, rejecting missing, nonnumeric,
nonpositive, and fractional entries. `pd.to_numeric(errors="coerce")` turns bad
strings into missing numbers so one Boolean mask can find all invalid rows.
`codes == np.floor(codes)` checks integrality without rounding fractional inputs
into a valid ID. Only after validation does `astype("int64")` convert the dtype.
`description` makes an error identify the consumer or retail source. Quietly
replacing an invalid DMA with zero would contaminate own-market exclusion.

## est_code/price_control.py::other_dma_instruments

This retained helper accepts three pandas tables and returns weekly rows with
UPC-matched flavor IVs. It is **not** the pooled instrument construction used
by current `est.py`. It first averages retail-DMA prices outside the consumer's
DMA per UPC/week, then averages across the exposure store's UPC basket.

Renaming the own-DMA table's columns makes its keys match the merge. `present`
is a Boolean Series; converting it to integers subtracts one from the DMA count
only when that own-market quote exists. The grouped `count` ignores missing
values whereas `size` counts all rows. Comparing them detects an incompletely
instrumented basket and marks the whole flavor IV missing, rather than quietly
averaging only a subset. Several joins replace a much slower nested lookup loop.

## est_code/price_control.py::absorb_effects

Input `values` is a numerical matrix; `groups` lists row-aligned group labels
such as week and DMA. Output is a copy with group means removed. Factorization
converts arbitrary labels to dense integers. `np.add.at(means, code, residual)`
accumulates rows into group sums even when indices repeat; ordinary fancy-index
`means[code] += residual` may not accumulate repeats as intended.

`counts[:, None]` makes group counts a column so every variable is divided by
its group size. The function cycles week and DMA demeaning until the biggest
absolute change falls below tolerance, then returns. A maximum iteration limit
raises on failure. A huge explicit dummy matrix is a valid alternative used in
tests, but its memory grows with the number of fixed-effect groups. One pass
is insufficient when group cells are unbalanced.

## est_code/price_control.py::add_price_controls

This returns `(augmented_frame, diagnostic_table)`. For each flavor it selects
covered prices, absorbs week/DMA effects jointly from price, instrument, and
Xi, then partials Xi out of y and z. `within[:, 0]`, `[:, 1]`, `[:, 2:]` split
those roles by known column order. `lstsq(...)[0]` keeps only the coefficient
vector from the least-squares result tuple.

Dot products compute the IV slope after partialling. A small residual instrument
sum of squares raises instead of dividing by an almost zero denominator.
`np.linalg.lstsq` tolerates redundant controls more gracefully than explicitly
inverting their cross-product matrix. The residual is assigned only to covered
rows; unavailable alternatives retain zero placeholders. Cluster score sums,
partial R-squared, and residual SD describe first-stage relevance and fit.
They do not prove instrument validity. Reusing this function in bootstrap
replications keeps the two-stage estimator consistent across samples.

## est_code/bootstrap_se.py::resample_households

Input is the weekly frame and a NumPy random Generator. Output is a concatenated
bootstrap sample with the same number of household draws as the original.
The comprehension stores whole household subtables. `rng.integers(0, H, size=H)`
draws indices with replacement; its upper endpoint H is excluded.

`enumerate(selected)` supplies a fresh ID and selected old-group position.
`.copy()` is essential before changing that ID so the original history is
not modified. If selection is `[2,2,0]`, two copies of old household 2 get new
IDs 0 and 1, keeping their likelihoods separate. Concatenating with
`ignore_index=True` gives clean row labels. Resampling individual weeks would
break persistent household preferences and within-household dependence.

## est_code/bootstrap_se.py::bootstrap_standard_errors

The `estimate` argument is a callable, not an estimate array. Each repetition
resamples households, refits enabled price controls, calls `estimate(sample,
initial)`, and records its result. `record.update` adds named fields to an
existing dictionary. Accepted vectors go to `values`; all attempts go to
`records`, so failure does not disappear from the audit trail.

The `try/except` catches expected data/linear-algebra problems. Every attempt
checkpoints the records as CSV. After the loop, the threshold requires at least
20 usable fits and at least 80% success. Returning `(None, report)` explicitly
signals inference was withheld. Otherwise, `std(ddof=1)` uses sample rather than
population SD, `quantile` forms empirical percentile intervals, and
`np.cov(..., rowvar=False)` treats columns as parameters. Analytic Hessian SEs
would be cheaper but would require handling the estimated first-stage residuals
and household dependence; this implementation uses a two-stage cluster bootstrap.

## data_code/summ_stats.py::normalize_upcs

This is the pandas normalization helper also used in estimation: strip whitespace,
recognize numeric-looking identifiers, remove `.0` and leading zeros, and retain
nonnumeric strings. Its argument is a Series and its return is a Series. The
`where` condition retains original text on nonnumeric rows. The reason for
string processing rather than floating-point conversion is identifier precision.
See the estimation function's worked examples; identical naming here does not
mean Python automatically shares their definitions across modules.

## data_code/summ_stats.py::normalized_upc_expr

This constructs the Polars-expression version of UPC normalization for a lazy
pipeline. It returns a recipe named `upc`, not a table. Nested conditions replace
an all-zero numeric identifier with `"0"`, preserving nonnumeric text. The
function avoids repeating a long expression at every join boundary. Using
matching canonical keys on both sides prevents `"001"` and `"1"` from
mistakenly appearing unmatched.

## data_code/summ_stats.py::valid_flavor_labels

Input is a pandas table with `flavor_descr`; output is a Boolean Series.
Missing text becomes an empty string, is cast to string, stripped, and lowercased.
`.isin` checks a list of unusable sentinel strings, then `~` negates the result.
Thus `" N/A "` is invalid while `"Strawberry"` is valid. Normalizing once
before comparison is clearer than writing separate conditions for every case
and whitespace variant. This helper exists alongside directly written lazy
filters; it is not automatically applied to every flavor-processing path.

## data_code/summ_stats.py::scan_purchases

This builds and returns a lazy, projected purchase query. It validates required
column labels and maps optional source labels to a common schema. `aliases.items()`
supplies output name plus candidate source names. `next(..., None)` picks the
first available candidate; otherwise `pl.lit(None, dtype=pl.String)` makes an
explicitly typed missing column. A consistent schema simplifies later summaries.

The comprehension builds expressions without reading rows. Dates, UPCs,
quantities, and prices get consistent representations. Head-age zero codes
become null, then `pl.coalesce` picks the first nonmissing head-age category,
preferring female then male. This is a priority rule, not a mean. Eagerly loading
all columns first would use more memory; dropping rows because an optional
demographic is absent would change the summary population unnecessarily.

## data_code/summ_stats.py::repeated_yogurt_households

This keeps household histories with at least two distinct purchase weeks after
the caller's year/price filters. `week` is a Polars expression using modulo
arithmetic to reach Saturday. The query obtains each household's minimum and
maximum week: unequal endpoints prove at least two distinct weeks without
retaining a full distinct-week set.

Only eligible household IDs are collected. A semi join filters the original
lazy purchase table while preserving its columns and all eligible purchase
rows. Joining an expanded eligibility table could multiply rows; the semi join
expresses membership only. Multiple trips in the same week do not satisfy this
rule. Estimation later repeats a similar restriction after additional exclusions.

## data_code/summ_stats.py::household_summary

Input is a lazy purchase table; output is a lazy one-row-per-household query.
The `expressions` list defines exact purchase totals, distinct counts, and
observed years. Demographic history follows a separate path: select a narrow
table, deduplicate, stably sort, and take each field's first nonmissing value.
Aggregating the full purchase table separately preserves exact trip/UPC/line
counts while avoiding repeated wide demographic sorts.

`group_by(...).agg([...])` evaluates each listed expression per group. A
validated one-to-one join reunites totals and demographic values. Later
`with_columns` steps preserve original race, apply the Hispanic override, and
calculate approximate weeks/trip rate. The second step can refer to the new
`approx_observed_weeks` created in the first. Counting only observed years
avoids filling panel gaps but still assumes full-year exposure within those years.

## data_code/summ_stats.py::as_lazy

This small adapter always returns a Polars LazyFrame. `isinstance` checks the
runtime type. An existing LazyFrame is returned unchanged, a pandas table is
converted through `pl.from_pandas(...).lazy()`, and the remaining expected Polars
DataFrame receives `.lazy()`. This lets callers supply convenient small eager
tables in tests while production uses lazy scans. It is not a universal converter:
an arbitrary unsupported object without `.lazy()` will still fail. Avoiding a
copy/conversion when already lazy preserves the existing query plan.

## data_code/summ_stats.py::partition_purchases

This executes a lazy scan once to write physical household-hash partitions,
then returns a list of lazy scans over those files. `hasattr(pl, "PartitionBy")`
checks API availability before starting. `(hash(seed=0) % batches)` maps every
row with the same household ID to the same bucket. This is deterministic routing,
not random sample selection.

`sink_parquet(pl.PartitionBy(...))` writes by bucket. The list comprehension
includes only existing bucket directories, important because small examples
can leave buckets empty. `sorted(directory.glob(...))` makes input order
predictable. `hive_partitioning=False` prevents folder names from automatically
adding a partition column that was intentionally excluded. The alternative
of filtering/recollecting the original source for every bucket repeats large
scans; physical partitions trade temporary disk space for reusable reads.

## data_code/summ_stats.py::household_batches

This accepts an optional list of already written partitions. The conditional
expression reuses them if supplied; otherwise it creates them once. `enumerate`
gives each partition an output filename suffix, `household_summary` runs on it,
and `sink_parquet` writes its reduced result. The final wildcard scan returns
one logical table of all household outputs without first concatenating them
eagerly in Python. Households never cross partition boundaries, so exact
household summaries can be computed independently and then stacked.

## data_code/summ_stats.py::save_table

This writes both a machine-readable CSV and an escaped LaTeX table and prints a
readable version. Polars inputs are collected and converted to pandas because
the chosen table formatting/export methods belong to pandas. A pandas input
already has those methods, so conversion is skipped.

`lambda value: f"{value:.2f}"` formats display numbers without changing the
stored numerical data. `index=False` suppresses an unwanted row-number column.
`escape=True` protects LaTeX syntax characters in labels. CSV retains reusable
data whereas LaTeX supplies paper-ready formatting; either output alone would
serve only one of those purposes. This function's useful result is its side
effects, so no explicit return is required.

## data_code/summ_stats.py::demographic_tables

Input is a one-row-per-household table, an output directory, and filename prefix.
The initial count is its denominator. The loop builds one categorical table
per demographic using null-to-`Missing` replacement, grouped counts, optional
codebook mappings, and percentages of all households. `.map(labels).fillna(code)`
keeps source codes visible when a verified label is absent.

The `specs` list pairs human-readable metric names with Polars aggregation
expressions. Enumerated string aliases let all metrics be computed in a single
query and then zipped back to names. `rows.extend(generator)` appends race
percentages to the overview. Equal household weighting is achieved by summarizing
households first; directly counting purchase rows would overweight frequent
shoppers. Numeric means of income/head-age codes remain means of categories,
not economic levels.

## data_code/summ_stats.py::scan_reference

This wraps `pl.scan_csv` with consistent parsing choices. `infer_schema=False`
keeps reference fields as strings instead of guessing that UPCs are numbers.
`null_values=[...]` explicitly recognizes several missing-value spellings.
The return is lazy, so calling the function sets up a CSV read without loading
the entire table. Centralizing these options prevents different recovery and
classification paths from interpreting the same file differently.

## data_code/summ_stats.py::recover_flavor_reference

This supplements, rather than overwrites, the descriptor reference. Both input
tables are lazy; an anti join excludes product UPCs already present in the CSV.
The filter removes missing/placeholder candidate descriptors. If annual metadata
exists, each UPC's candidate columns take earliest-year values; otherwise first
row deduplication is used. The output is a new combined reference plus a separate
CSV of just recovered rows.

`.with_name(output.stem + ...)` constructs a sibling output filename. Comparing
resolved paths rejects accidental overwrite of the original reference. After
writing additions, `scan_reference` rereads them under the same schema conventions
before diagonal relaxed concatenation allows unequal optional columns. The
return is the output Path, enabling `run` to use it for subsequent classification.
It does not repair blank descriptors on UPCs already in the original reference.

## data_code/summ_stats.py::flavor_reference

This resolves one label per normalized UPC and returns an eager Polars mapping.
`detailed` selects two different expression recipes. Detailed labels use trimmed
lowercase descriptors with whitespace collapsed; grouped labels use ordered
plain/berry text matching and an other fallback. `.replace` maps specific whole
values; regex `.str.replace_all` instead matches patterns within text.

Before deduplicating, the function counts UPCs assigned multiple distinct labels
and can save their rows for review. `unique(..., keep="first", maintain_order=True)`
implements first-reference-row precedence reproducibly. This is a transparent
rule but not a semantic resolution of conflicting descriptions. Returning only
UPC and final flavor gives downstream joins a small unique lookup table.

## data_code/summ_stats.py::yogurt_weeks

This attaches flavor labels, audits matched purchase lines, and returns a lazy
weekly quantity query plus an eager small audit table. A left join preserves
unmatched purchases long enough to count/report them. Unmatched UPCs are exported
and cause a ValueError, preventing an apparently successful summary of a silently
reduced sample. Nonpositive/missing quantities also stop the run.

The Saturday expression adds a row-dependent duration to purchase date. Grouping
by household/week/flavor and summing quantity keeps only observed combinations.
The alternative dense pivot would create a column for every detailed flavor
and potentially a great many empty cells. Price problems are audited here;
the main matched-sample caller has already applied its positive-price filter.

## data_code/summ_stats.py::flavor_summaries

This is the main descriptive-statistics engine. It returns pandas share and
transition tables for plotting, while writing much richer outputs. First it
sums quantities by flavor and computes package shares; a rounding filter removes
rows whose displayed share is zero but leaves the denominator unchanged. Next
it aggregates household/flavor totals to number of flavors, squared quantities,
and largest quantity, producing HHI and largest-flavor share.

The basket query groups again by household/week. `pl.when(pl.len() == 1)` gives
a single-flavor label only to unmixed baskets. `.shift().over("household_code")`
looks at the previous basket within the household, not across household borders.
Date differences identify adjacent weeks. A cumulative sum of spell-start flags
creates run identifiers; mixed baskets and gaps break the runs.

The expensive basket query is checkpointed once in a temporary directory so
transitions, spells, CSV exports, and global medians can reuse it. Transition
counts pivot into a small matrix; dividing by row sums produces probabilities.
Zero denominators become NaN. Spells are averaged as spells, not as households.
Finally a small metric table records basket frequencies and eligible-pair
switching. This sequence uses several aggregation units intentionally; replacing
them all with one grand groupby would obscure their different denominators.

## data_code/summ_stats.py::make_hhi_figures

This reads just the saved `flavor_HHI` column and writes histogram and CDF PDFs.
`usecols` avoids reading irrelevant household fields. `.dropna().to_numpy()`
leaves a numeric vector. If it is empty, a bare return avoids dividing by zero.
The histogram weights each household by `100 / n`, so heights are percentages.
`linspace(0,1,41)` defines 40 bins.

The CDF uses sorted observations and cumulative percentages. `np.r_` concatenates
an initial zero with the vectors to draw the starting segment. `where="post"`
creates a right-continuous step plot. A loop over two `kind` strings shares
styling/export code without confusing their different calculations. It reuses
saved household summaries instead of rerunning a large private-data pipeline.

## data_code/summ_stats.py::make_figures

Inputs are already computed flavor-share and transition tables. Sorting then
`head(20)` selects a readable subset for plots, and `reindex` aligns matrix rows
and columns with the displayed flavor order. It does not recompute transition
denominators within that subset, so a plotted row need not sum to one.

`fig, ax = plt.subplots` separates a Figure container from its plotting Axes.
`ax.bar` draws shares; `ax.imshow` colors transition cells. Two nested index loops
place formatted text inside each cell. Tick rotation improves long-label
readability. `tight_layout`, `savefig`, and `close` finish each output. Plotting
inside the data aggregation itself would make it harder to skip figures or
recreate them from existing summary tables.

## data_code/summ_stats.py::run

This orchestrates the summary workflow and writes final artifacts. `sources`
and `source_partitions` are dictionaries keyed by sample prefix, keeping full
panel and matched yogurt objects separate. It optionally supplements flavor
metadata, then prepares each population, writes reusable hash partitions,
computes household/demographic summaries, and audits annual coverage.

The second loop runs detailed then grouped flavor taxonomies. Partition-level
weekly outputs remain sparse, while small audit tables are summed. The last
reference therefore holds the grouped mapping, which is saved for estimation.
The final export intentionally reopens the original purchase file, rather than
the demographic projection, to preserve store/DMA and other fields. A semi join
keeps eligible households and a left join attaches the resolved flavor.

`prepared.drop("flavor")` avoids two competing label columns when reprocessing
an already labeled input. The resolved-path check prevents writing over the
original source. This orchestration keeps the population restrictions explicit;
running the estimator on an arbitrary intermediate table can produce another
sample even if its filename looks similar.

## data_code/summ_stats.py::main

This parses summary options and chooses ordinary pipeline work or HHI-only work.
`args.output or default_path` uses a fallback when the optional output is None.
The HHI-only branch checks its saved input, makes graphs, optionally repeats for
grouped summaries, then returns before raw-data setup. It is a true early exit,
not merely a plotting flag.

Ordinary runs create the output directory and enter a TemporaryDirectory
context there, passing its Path to `run`. Context management ensures intermediate
partitions are cleaned on ordinary completion or exceptions. Keeping temporary
work under output makes disk requirements predictable. `--no-figures` and
`--hhi-only` are rejected together because their requested actions conflict.

## data_code/recover_flavors.py::main

This small command-line wrapper imports the existing recovery function under
the shorter local name `recover`. `as recover` changes only the imported name,
not the implementation. Three Path arguments specify the original CSV, annual
products, and output file; `recover(...)` does the actual work. Reusing that
function avoids a second recovery algorithm drifting away from the summary
pipeline. The guarded entry point lets tests import the module without running
the recovery job.

## supply_code/model/utility.py::signed_state

This converts the two accepted string state names into integer signs. Two
independent `if` blocks are fine because each successful branch returns at
once. Any other value reaches `raise ValueError`. `signed_state("j")` returns
-1; `signed_state("0")` returns +1; numeric `0` is not string `"0"` and is rejected.
A dictionary lookup could shorten this to a mapping, but these explicit branches
make both accepted values and the custom error easy for a novice to see.

## supply_code/model/utility.py::raw_utilities

Input `price` may be a SymPy symbol and `signed_history` a sign or symbolic q.
The return statement contains two comma-separated expressions, so it returns
a tuple `(inside_utility, outside_utility)`. The combinations `(1+q)/2` and
`(1-q)/2` act as complementary indicators when q is ±1. At q=-1 the first is
zero and second one; at q=+1 they reverse.

Arithmetic with SymPy objects builds algebraic expressions rather than computing
a numerical utility immediately. Returning both raw utilities preserves the
outside switching payoff before normalization. Writing only an inside bonus
would encode a different model. Module-level unpacking later names each result
and subtracts the outside utility explicitly.

## supply_code/model/numerical.py::inside_share

This computes one numeric binary-logit probability. `signed_state` validates
the state; `expit` evaluates logistic demand stably, then `float` turns a
NumPy scalar into an ordinary Python number. Direct `exp(x)/(1+exp(x))` can
overflow at large x. The defaults beta=2 and alpha=-1 support the benchmark,
while callers can supply other calibrations. This function alone does not
validate alpha's sign; full parameter-domain checks live in `evaluate`.

## supply_code/model/numerical.py::optimal_price

This finds one optimal price for a utility intercept, negative alpha, and an
effective cost. `f = lambda p: ...` defines the transformed scalar FOC. `clip`
limits the exponential's argument to a representable range; it is a numerical
safeguard rather than a theoretical cap on utility.

The initial lower endpoint is effective cost, where f is positive. `max` sets
a reasonable initial width, and `while f(hi) > 0` doubles it until the bracket
changes sign. `brentq` can then locate the zero to `xtol=1e-12`. A fixed interval
is simpler but can miss a root under another calibration. Unconstrained generic
minimization ignores the monotonic scalar structure and can need more evaluations.
The maintained alpha<0 assumption establishes a unique root; the caller validates it.

## supply_code/model/numerical.py::evaluate

This is backward induction expressed as ordinary assignments. It accepts gamma,
initial history, and optional calibration numbers, then returns a dictionary.
`np.isfinite([...]).all()` validates every numeric input at once. The terminal
list comprehension solves two prices, unpacked into p2j and p20. Subsequent
tuple assignments compute their shares and values; first-period effective cost
then uses the value difference.

`_EVALUATE(...)` calls the numerical function built from symbolic derivatives
at import time. Its argument order must match `_ARGS`; a positional swap can
produce plausible but wrong derivatives. `result.update(dict(zip(_KEYS,
map(float, values))))` converts and labels those outputs. State-inapplicable
ratio derivatives become NaN intentionally. Final FOC residuals use original
profit derivatives as a diagnostic independent of the root's transformed form.
An external simultaneous optimizer would conceal the model's simpler sequential structure.

## supply_code/model/numerical.py::solve

This is a compact convenience interface. The bare `*` makes `initial_state`
keyword-only; `**parameters` collects other keywords. Membership tests reject
obsolete z1/z2 inputs. `parameters.pop("gamma", 0.)` retrieves and removes gamma,
allowing the rest to be expanded into `evaluate` without passing gamma twice.
The function returns seven values as a tuple. It is convenient for unpacking,
while `evaluate` is preferable when you need named derivative or diagnostic fields.

## supply_code/comparative_statics/canonical.py::price2_derivative

This returns a **symbolic expression** for the terminal optimal-price derivative
with respect to symbolic parameter theta. Default arguments `sj2`, `q2`, and
`p2` are shared symbols; callers substitute a particular branch by passing u/-1/p2j
or v/+1/p20. `sp.diff(beta, theta)` is one when theta is beta and zero for an
independent parameter, so one generic expression covers several comparative statics.

`direct` is the fixed-price derivative of the utility index. The returned
formula already incorporates implicit share/price feedback; `simplify` reduces
algebra. Differentiating the markup while holding its share fixed would omit
that feedback. Unlike numerical finite differences, symbolic derivatives do
not need a step size, though they still depend on a correct symbolic model.

## supply_code/comparative_statics/canonical.py::value2_derivative

This computes the symbolic envelope derivative of a reoptimized continuation
value. It uses the same fixed-price index derivative as terminal pricing, then
returns `-share*direct/alpha - share*diff(c,theta)`. There is no explicit derivative
of optimal price because the terminal FOC cancels that first-order effect.
Omitting price responses here is justified by the envelope theorem, not a
general Python or SymPy rule. Substituting fully expanded optimal prices would
be harder algebra with the same result under the maintained optimum conditions.

## supply_code/comparative_statics/canonical.py::price1_derivative

This forms the symbolic first-period price derivative. It builds its direct
utility effect, obtains the two terminal envelope derivatives, subtracts them,
and inserts the result into the implicit first-price formula. `sp.diff(delta,
theta)` includes a direct discount-factor change when theta is delta; for gamma
that term is zero but the continuation values still change.

The two helper calls use the correct opposite signed terminal histories.
Maintaining a single generic function avoids separate alpha/beta/delta/gamma
formulas disagreeing about those branches. `sp.simplify` does algebraic cleanup;
it does not decide whether a term is positive under the economic domain.

## supply_code/comparative_statics/canonical.py::total_derivative

Input is a symbolic expression and differentiation parameter. Output is its
total derivative along the solved equilibrium. The first assignments compute
price derivatives, then logistic chain rules compute derivatives of the three
shares. A dictionary maps each endogenous symbol to its derivative.

The return is explicit partial differentiation plus
`sum(diff(expr,k)*val for k,val in substitutions.items())`. This is the chain
rule over every mapped endogenous object. The dictionary is a derivative map,
not values actually substituted with `.subs` in this function. `sp.factor`
collects common algebraic factors. Plain `sp.diff(expr,theta)` would treat shares,
prices, and values as independent symbols and miss their indirect changes.

## supply_code/comparative_statics/canonical.py::terminal_total_derivative

This is the smaller one-branch version of the total derivative. It calculates
terminal price response, computes terminal share response including repricing,
and adds explicit theta, price, and share chain-rule terms. Because there is
only one terminal branch, a short direct expression is clearer than building
the full period-1 derivative dictionary. It is used for terminal curvature and
mixed derivatives; a second plain symbolic partial derivative would hold the
share fixed and give the wrong comparative static.

## supply_code/comparative_statics/canonical.py::profit_derivative

This returns the optimized two-period profit derivative with respect to theta.
`partial` combines direct current demand/cost effects, direct discounting, and
both branch envelope effects. `.subs(p1, markup_expression)` inserts the
first-period optimal pricing equation, then `simplify` reduces the result.

The code does not explicitly multiply by dp1/dtheta because the first-period
FOC makes that contribution zero at the optimum. Replacing the entire objective
with current revenue before differentiating would lose continuation effects.
This helper is reusable for gamma and beta, while relying on the same fixed-type,
two-branch, interior optimum as the rest of the model.

## supply_code/make_comparative_static_figures.py::save

This formatting/output helper takes a Matplotlib Figure and filename stem.
`tight_layout` adjusts spacing; a two-element suffix loop saves both PDF and
PNG; `plt.close` frees the Figure. `OUT / f"{name}.{suffix}"` constructs each
Path. Shared export code prevents one figure accidentally lacking its vector
PDF version. DPI matters chiefly for raster PNG content; it does not make an
ordinary vector line in a PDF more mathematically precise.

## supply_code/make_comparative_static_figures.py::main

This generates all benchmark rows from the production `evaluate` function,
writes a solution CSV, and plots six requested figures. The dictionary
comprehension creates a list of gamma evaluations for each initial-state key.
`csv.DictWriter` uses dictionary keys as field names, so output records retain
branch meanings. Plot `specs` is a data list describing several similar figures;
the loop executes the common plotting logic.

Separate blocks handle acquisition ratios and the two-panel price-level figure.
Terminal prices can be read from either initial-state collection because their
realized history determines them. A final generator computes the largest absolute
FOC residual and raises if it exceeds tolerance. Plotting algebra typed anew
here would risk drifting from the derivative expressions; numerical values
instead come from the common symbolic-to-numerical path.

## supply_code/make_latex_document.py::main

This writes a research appendix as text. The opening loop pairs LaTeX left-hand
side strings with actual SymPy expressions and delegates equation formatting.
The very long raw triple-quoted `document` string is document **data** inside
the Python program: its LaTeX commands are interpreted later by a TeX engine,
not executed as Python statements. A raw string preserves the many backslashes.

Later loops append benchmark rows and figure sections. An f-string inserts
computed numbers with three decimals; adjacent raw-string fragments preserve
LaTeX row-ending backslashes. The figure-existence check stops on missing assets
instead of generating a broken paper. `write_text` creates `.tex`, not a PDF;
compilation is a separate step. Repeated string `+=` is acceptable for this
small document; accumulating a list and `"".join(parts)` would scale better
for very large text assembly.

## supply_code/export/latex_export.py::display_equation

This returns a LaTeX display-equation string. `lhs` is already a LaTeX label;
`sp.latex(expression)` serializes the actual symbolic right-hand side.
`'\\[\n'` contains a backslash followed by a bracket and a newline; Python string
escapes are why source contains doubled backslashes. Concatenation places the
equation between display delimiters. Deriving the right side from SymPy avoids
manually copying a formula that later changes in the code. The function returns
text and does not itself invoke a TeX compiler.

## supply_code/export/pretty_print.py::show

This is a terminal-output helper returning None. Blank `print()` creates a
new line; `"=" * 70` repeats a string seventy times, whereas multiplication
of numeric values has a different meaning. `sp.pprint` renders an expression
in readable mathematical form in the terminal. It is useful for inspection,
while `sp.latex` is the appropriate alternative for document source.

## supply_code/structural_switching_reference.py::expit_safe

This imports and returns SciPy's stable logistic transform. The local import
occurs when the function is called; Python caches imported modules, so it is
not reloading SciPy from scratch each time. A module-level import would be
more conventional here. The numerical advantage over a naive exponential ratio
is avoiding overflow at extreme utility indices.

## supply_code/structural_switching_reference.py::inside_share

This independent reference maps a state to a sign with nested conditional
expressions, rejects invalid states, and evaluates logistic demand. The final
`else None` makes invalid input detectable before multiplying it by gamma.
Defaults capture the benchmark constants at function definition time. Unlike
the production state helper, the sign logic is written independently to help
cross-check the implementation rather than simply calling the same helper twice.

## supply_code/structural_switching_reference.py::p2_opt

This reference solves the terminal markup equation using a lambda and a fixed
bracket [-30,40]. Its gamma/state inputs enter through the closed-over variables.
The return is the `brentq` root. It is appropriate for the reference benchmark;
the production solver's expanding bracket handles a wider calibration range
without assuming the root remains in this interval. Separate implementations
make agreement more informative than two wrappers around the same solver.

## supply_code/structural_switching_reference.py::evaluate

This independently computes backward-induction prices, branch values, profit,
and hand-written analytic gamma derivatives. Comma-separated assignments unpack
small tuples; `r = 1-s` names the outside share. Conditional expressions choose
the correct matched denominator, and acquisition ratios are NaN outside their
applicable initial history. An `assert np.isclose(...)` checks the price
decomposition internally. Production uses lambdified canonical formulas,
whereas the reference writes formulas directly to provide an independent oracle.
Its interface and benchmark constants are narrower than production's.

## supply_code/structural_switching_reference.py::verify

This loops over selected gamma values and both histories. For each it re-solves
at gamma+h and gamma-h, then compares analytic derivatives with centered finite
differences. The step h balances approximation and floating-point error; making
it arbitrarily tiny does not ensure accuracy. `assert condition, details` raises
AssertionError carrying a diagnostic tuple when a check fails. Ratios must be
recomputed from reoptimized price levels, not differentiated holding terminal
prices fixed. This is a numerical check, not a symbolic sign proof.

## supply_code/structural_switching_reference.py::plot_series

This plots one result field for two initial histories and saves a raster image.
`[r[key] for r in rows[state]]` extracts that field from every result dictionary.
Separating the data key from axis label/filename lets one helper generate price,
derivative, and ratio plots. The function expects rows already aligned to
GAMMAS. Its outputs belong to the retained reference program, not the six-figure
production path used by `run_all.py`.

## supply_code/structural_switching_reference.py::main

This first verifies the reference formulas, creates its `outputs` directory,
solves benchmark rows, writes CSV, and saves several reference plots. `with ...
open(..., newline="")` lets the CSV writer handle line endings correctly.
Some lines contain semicolons separating multiple statements; Python executes
them left-to-right just as if written on separate lines. That compact style
saves space but is less friendly to beginners. Production uses another entry
point and another output folder; running this does not replace `run_all.py`.

## supply_code/checks/test_symmetric.py::SymbolicChecks.equal

This helper asserts symbolic equality by simplifying the difference to zero.
`self` supplies the current unittest instance, so `self.assertEqual` records a
normal test failure instead of only printing a warning. Structural `left == right`
may fail for algebraically equivalent expressions written in different forms.
Simplifying their difference is a stronger algebraic comparison for these
expressions, though symbolic simplification is not a universal theorem prover.

## supply_code/checks/test_symmetric.py::NumericalChecks.test_independent_reoptimization_and_derivatives.negative_profit

This nested objective accepts a candidate first-period price, computes its
logit share, and returns the negative of complete two-branch discounted profit.
It captures current gamma, state, calibration, and solved continuation values
from the enclosing test loop. `minimize_scalar` minimizes, so negating profit
turns maximization into minimization. Using a fresh direct objective here
checks the production root against another computational route.

## supply_code/checks/test_symmetric.py::NumericalChecks.test_all_parameter_and_mixed_derivatives_numerically.arguments_at

This nested adapter turns a result dictionary into the exact tuple expected
by lambdified symbolic derivatives. The q sign uses initial history, and branch
prices/shares/values retain their order. Its benchmark constants deliberately
match the enclosing test calibration. A tuple is appropriate because the
numerical evaluator uses positional arguments; a dictionary would require
explicit ordered extraction anyway. Swapping u/v could make a wrong formula
look correct at the symmetric gamma=0 point, so tests also use nonzero gamma.

## est_code/test_est.py::EstimationChecks.frame

This fixture method returns a tiny three-week pandas table. It includes a
two-week gap, mixed counts, all three prices, and outside counts so history,
menu, and likelihood tests can reuse the same explicit example. `self.frame()`
returns a fresh table each time, letting one test modify it without contaminating
another. A shared global mutable table would make results depend on test order.

## est_code/test_est.py::EstimationChecks.test_parquet_pipeline_and_fit.append

This nested fixture builder appends one purchase dictionary to an enclosing
`rows` list. Positional inputs describe household/week/UPC/store/DMA/price;
quantity defaults to one. A dictionary lookup turns UPC 1/2/3 into the three
flavor names. `len(rows)+1` assigns a fresh trip ID at append time. The helper
reduces repetitive fixture text while keeping test prices and households explicit.
It mutates the captured list and returns None; callers need not assign its result.

## est_code/test_price_control.py::synthetic_panel

This constructs a reproducible simulated panel with 60 households and eight
periods, first-stage prices, residuals, utilities, and multinomial package
counts. `repeat` holds each household ID for eight rows; `tile` repeats the
eight-date sequence for all households. Stable softmax subtracts each row's
largest utility before exponentiating. `rng.multinomial(8,p)` samples an
eight-package vector whose entries sum to eight. This gives the tests known
structure without private data; it is a test fixture, not estimated reality.

## est_code/test_price_control.py::PriceControlChecks.test_bootstrap_refits_first_stage_and_reports_empirical_sd.estimate

This nested fake estimator returns an `OptimizeResult` with coefficients equal
to resampled residual SDs. It deliberately depends on the recomputed controls,
so frozen controls would change the bootstrap distribution. It declares success
and a harmless objective because the test targets bootstrap orchestration,
not numerical optimization. Mocking the demand fit makes this check faster and
isolates whether first stages are rerun and empirical SEs calculated correctly.
