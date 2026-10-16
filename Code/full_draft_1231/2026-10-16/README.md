# Flavor preferences, purchase history, and price diagnostics

This folder is a self-contained snapshot created on October 16, 2026. It leaves
the earlier scripts and estimates unchanged. `demand_core.py` copies the existing
Step 4 engine; `estimate.py` adds a controlled comparison between shared tastes
and persistent random tastes. `diagnostics.py` examines the relative design.
`price_audit.py` inspects prices **before** any imputation.

## What beta and gamma mean

Beta does not require random flavor coefficients to be identified. In the shared
taste model, beta describes common baseline preference differences. In the random
taste model, households have persistent deviations from those means:

```
beta_ij = mean_beta_j + sd_beta_j * normal_draw_ij
utility_itsj = beta_ij + alpha * price_tsj - gamma * log(1 + stock_itj)
stock_ibj = delta ** calendar_gap_weeks * (stock_iaj + cups_iaj)
```

The first flavor's beta and taste SD are normalized to zero; other coefficients
describe taste differences relative to it. Gamma applies to every flavor,
including the reference. Positive gamma means fatigue; negative gamma means
reinforcement. Stock is a decayed **purchase proxy**, not observed consumption.
Initial stock is zero. All stores in a household-week use the same pre-week
stock; their purchases update the next week's stock once, in aggregate.

People who always like strawberry both accumulate strawberry exposure and buy
it repeatedly. Shared beta can misattribute their persistent tastes to negative
gamma. Random beta provides a competing explanation. It does not manufacture
identification. If relative exposure barely varies, beta and gamma can compensate
for each other even in a correctly programmed model.

The model conditions on observed yogurt basket totals, dates, stores, and the
retained flavor set. It uses a multinomial count likelihood, treating cups as
independent conditional allocations given household type and pre-week history.
The same taste draw persists throughout a household's sequence; likelihoods are
multiplied over the sequence **before** integrating over draws. Multinomial
constants are omitted consistently. There is no invented outside-option count,
participation equation, within-basket satiation, or stockpiling decision.

## Files and dependencies

- `estimate.py`: audit, shared/random taste fits, no-history fits, gamma profiles,
  and optional recursive parameter recovery.
- `diagnostics.py`: design rank, residual exposure variation, and within-household
  variation by flavor pair.
- `price_audit.py`: raw price quality, subgroup patterns, candidate fields,
  optional payment comparison and original RMS key verification.
- `make_demo.py`: reproducible synthetic data and deliberately invalid prices.
- `demand_core.py`: frozen estimation/data preparation engine.
- `CORE_REFERENCE.md`: detailed reference copied with the engine. Its old
  `est_step4.py` commands refer here to `demand_core.py`; its income defaults
  concern that engine's own CLI, not the comparison CLI in `estimate.py`.
- `requirements.txt`: Python dependencies. Python 3.10+ is required.

Install dependencies in your chosen environment with
`python3 -m pip install -r requirements.txt`. Run the following commands from
this folder. Relative data and output paths resolve against this folder, even
if launched from somewhere else. Absolute input paths are supported. Outputs
must remain here; each new run requires a fresh output directory.

## 1. Diagnose prices first

Without `--input`, scripts use the existing cluster scanner-panel path.

```sh
python3 price_audit.py --output prices_raw

# Optional: verify the actual original retail join keys.
python3 price_audit.py --output prices_with_keys \
  --rms /scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets/full_retail.parquet

# After consulting field definitions, compare a documented line expenditure:
python3 price_audit.py --output prices_with_payments \
  --spend-column YOUR_LINE_EXPENDITURE_COLUMN
```

Use `--price-column` if the actual field has another name. `--prefiltered` is
required when an already restricted input lacks the standard filter columns.
Filters match Step 4: single-person households, the two established single-serving
codes, and yogurt module codes 3603/3612. Quantity must represent single cups
for estimation; single-serving product labels alone do not prove pack conversion.

Inspect outputs in this order:

1. `sample_stages.csv` and `price_status.csv`: confirm the denominator and separate
   missing, malformed, infinite, zero, negative, and valid observations. The saved
   prior run had 10,488 unusable prices among 17,538 retained purchase rows;
   this audit recomputes the counts from the current input.
2. `by_*.csv`: identify stores, UPCs, periods, and flavors responsible for failures.
   A high failure rate can reflect sparse RMS coverage, join conventions, or source
   values. Group patterns alone cannot distinguish those explanations.
3. `candidate_fields.csv` and `invalid_examples.csv`: inspect possible payment
   fields and actual failed examples without dropping purchases.
4. With `--rms`, inspect `rms_key_cross_tab.csv`. Missing prices with an exact
   original RMS key mean that absent key coverage alone is insufficient to explain
   the problem. No exact key means inspecting store/UPC/date alignment and source
   coverage. Nearby matches at +/-1 and +/-7 days suggest date checks; they are
   **not** permission to match a different week automatically. Key existence does
   not establish valid source price values. Inspect the source value for those
   keys next. If the original RMS price field is present, also inspect
   `rms_source_price_cross_tab.csv` and
   `invalid_target_with_valid_source_examples.csv`. These distinguish missing
   source prices from valid source values that did not reach the target, and
   flag conflicting source prices at a key. Use `--rms-price-column` for a
   differently named source field. The audit normalizes date formats and identifier formatting; this
   may find a key that a differently typed original join missed.
5. With a documented spend field, inspect `payment_comparison.csv`: compare
   recorded price with total line spending divided by cups. If a coupon field is
   supplied, the candidate becomes `(spending - coupon)/cups`. Only subtract it
   if spending is documented as pre-coupon and the coupon is a line total. Missing
   coupons are not silently treated as zero. Free purchases are separately visible
   and are not classified as positive-price observations.

The optional RMS check lazily selects and deduplicates store/UPC/week keys and
the source price field when present using Polars, then collects these in memory. Very large RMS files may still
require a high-memory cluster job or a pre-restricted key file. No data are modified.

Candidate payment fields are diagnostics, not an automatic replacement price.
Verify dollars versus cents, package size, discounts, and the distinction between
HMS transaction prices and RMS shelf/store averages. Observed coupon prices do
not reveal prices available for unchosen products. Estimation currently reads
the column named `price`; prepare a separately validated input if changing it.

## 2. Audit the estimation design

```sh
python3 estimate.py --audit-only --output design_audit

# Prefer original individual-flavor labels when available:
python3 estimate.py --audit-only --flavor-column flavor_str \
  --output individual_flavor_audit
```

Check that `flavor_str` actually contains original labels; a name is not a
guarantee. The default `flavor` field may contain only 0/1/2 categories. Splitting
flavors requires genuine source labels, not relabeling those three values.
Top flavors are selected by purchased cups, with remaining flavors excluded,
not pooled into a miscellaneous exposure category. This changes the conditional
estimand and is recorded in the audit. Check sensitivity to the retained set.

`audit.json` reports sample attrition, price fallback counts, and design rank.
`household_variation.csv` reports switching and centered relative exposure/price
variation within household and available flavor pair. `flavor_variation.csv`
reports exposure diagnostics by flavor. Two distinct purchase weeks are required;
first weeks are retained. This is a sample restriction, not an identification theorem.

The homogeneous design uses flavor intercept differences, relative price, and
negative relative log stock. Full rank means these linear log-odds columns are
numerically distinct under the maintained homogeneous model. Small singular values
or near-zero history residual RMS show compensating directions. These are
unweighted descriptive diagnostics, not Fisher information or a mixed-logit proof.
Within-household centering is diagnostic only; fitted stock is not demeaned.

Without an independent menu, all retained flavors are assumed available and
prices are completed from store/flavor means, then overall flavor means, then
an overall mean. These assumptions preserve quantities but do not create valid
price variation. Prefer a supplied `--menu` with observed available alternatives
and verified prices. Its rows require `store_code_uc`, `week_end`, the selected
flavor column, and `price`. Purchase inputs also require `household_code` and
positive integer `quantity`, plus the sample filter columns unless prefiltered.

## 3. Estimate the competing explanations

For SLURM, activate a Python 3.10+ environment with `requirements.txt` installed,
then submit from this folder (or use `2026-10-16/run_slurm.sh` from its parent):

```sh
sbatch run_slurm.sh
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --time=48:00:00 run_slurm.sh \
  --input /path/to/purchases.parquet --menu /path/to/menu.parquet \
  --flavor-column flavor_str --draws 256 --recovery-reps 10
```

`run_slurm.sh` defaults to one CPU, 32 GB RAM, and 24 hours; override resources
with `sbatch` options before the script name. It forwards all script arguments to
`estimate.py`, including `--audit-only`. Results default to
`slurm_results/JOB_ID` inside this folder; `--output` can select another fresh
directory here. SLURM logs appear in the submission directory. Set `PYTHON` to
an environment's absolute Python path if needed, and `RUN_DIR` to this folder's
absolute path when submitting from elsewhere. Without `--input`, the existing
cluster data path is used. No dependencies are installed by the batch job.

```sh
python3 estimate.py --output comparison \
  --sd-caps 0 4 8 --draws 128 --starts 3 \
  --profile -1 -0.25 0 0.1 0.5 1

# Supply credible menu and exact labels if those are available:
python3 estimate.py --input /path/to/purchases.parquet \
  --menu /path/to/menu.parquet --flavor-column flavor_str \
  --output comparison_exact
```

Cap zero fixes taste SDs to zero and estimates shared tastes. Positive caps
estimate persistent independent normal taste differences up to the specified
bound. These caps are sensitivity checks, **not** imposed heterogeneity levels.
Every fit uses identical households, counts, menus, stocks, and fixed Sobol draws.
The engine rounds draw counts upward to a power of two. Repeat a final random
fit with more draws to assess integration accuracy.

For each cap, the script saves `estimates.csv`, `fit.json`,
`no_history_estimates.csv`, and `gamma_profile.csv`. `model_comparison.csv`
collects convergence, starts, gamma, bounds, and no-history likelihoods.

Read these together:

- A gamma sign change after adding random tastes means the shared model and
  heterogeneous model assign repetition differently. It does not prove either
  interpretation causal or prove that beta was unidentified.
- A flat gamma profile with compensating beta means weak empirical separation.
  The profile re-estimates nuisance parameters at every fixed gamma.
- A taste SD at a cap calls for the larger-cap comparison. If it keeps moving
  outward, the random-taste distribution or count-independence assumption may be
  inadequate. Do not interpret a larger cap as an identification fix.
- Better profile or no-history likelihood than the unrestricted fit flags an
  optimization problem. Nonconverged estimates should not support conclusions.
- Better fit from random tastes uses more parameters. Raw likelihood improvement
  alone does not establish predictive performance or a test of causal fatigue.

No random-model standard errors or automatic significance labels are supplied.
Profiles and recovery diagnose different things; neither substitutes for a
household bootstrap or valid clustered inference. Boundary parameters require
care before applying routine likelihood-ratio thresholds.

## 4. Can the short panel recover gamma?

```sh
python3 estimate.py --output recovery_check --sd-caps 0 4 \
  --recovery-reps 10 --recovery-gammas 0 0.5
```

For each fitted taste specification, simulation substitutes gamma=0 and gamma=0.5
while retaining its fitted nuisance parameters. It draws persistent household
tastes, recursively simulates basket counts, updates stocks from those simulated
counts, and re-estimates from generic starts. It holds observed dates, stores,
menus, prices, and basket totals fixed. Paired seeds make the gamma scenarios
comparable. `recovery.csv` contains parameter truths, errors, and convergence.
Ten replications are a preliminary check; use more to characterize error reliably.
Successful recovery validates separation under that simulated model and design,
not price exogeneity, consumption measurement, or the taste distribution.

Repeat empirical fits with `--delta 0.5`, `0.8`, and `0.95`, each in a fresh
output directory. Zero initial stock is a maintained assumption; the current
wrapper does not implement an initial-conditions correction or estimate delta.
Unknown pre-sample purchases, inventory, serial preference shocks, promotions,
and selected shopping dates can all confound a fatigue interpretation. There is
no IV/control function in these fits. They do not directly support category
participation, retailer-profit, or welfare counterfactuals.

## Synthetic verification without Nielsen access

```sh
python3 estimate.py --self-test
python3 -m unittest discover -s . -p test_diagnostics.py
python3 make_demo.py
python3 price_audit.py --input demo_data/price_failures.csv --prefiltered \
  --spend-column line_spend --output demo_prices
python3 estimate.py --input demo_data/purchases.csv --menu demo_data/menu.csv \
  --prefiltered --output demo_fit --sd-caps 0 2 --draws 16 --starts 1 \
  --maxiter 150 --profile 0 0.5 --recovery-reps 1
```

The engine self-test checks calendar decay, multi-store aggregation, persistent
panel likelihood, unavailable alternatives, recursive simulation/recovery, and
the inherited income-model implementation. The demo is an execution check,
not evidence that a tiny synthetic sample gives precise empirical estimates.

## Validation performed when creating this folder

The engine self-test and four targeted regression checks passed. The latter
verify that constant relative exposure is confounded with beta, independently
varying exposure restores homogeneous design rank, raw price failures remain
distinct, and missing identifiers remain missing during key normalization.
The complete demo ran both shared and random tastes, gamma profiles, no-history
comparisons, and recursive recovery at zero and positive gamma. The price audit
correctly classified all five injected failures and verified the synthetic RMS
keys. Saved `demo_*` folders contain synthetic inputs and results only.

The real `/scratch/...` Nielsen inputs are unavailable on this local machine.
No new empirical Nielsen fit or source-data price diagnosis has been run here.
