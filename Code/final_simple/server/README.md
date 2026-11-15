> **Current presentation exercise:** use `demand/quantified_theory.py` with the
> saved aggregate parameter estimates. It computes common versus informed pricing
> in the original theory for all three flavors. No server job is needed. See
> `Code/final_simple/demand/QUANTIFIED_THEORY.md`. The server-based exercises
> below are earlier work and are not the current presentation exercise.

## Binary pricing: all focal flavors

Use `sbatch Code/final_simple/server/run_information_pricing.sh EXISTING_FIT_DIRECTORY all`
on the research server. This runs plain, berry, and other separately for the
three-category fit, each as focal flavor versus not focal in both periods.
Use only the plain output for the presentation. Existing multinomial estimates
are reused; no demand estimation is rerun. See `../demand/INFORMATION_PRICING.md`
for the current specification and walkthrough. Earlier full-basket pricing
results are not results from this new binary specification.

# Server jobs and recovery

Run these commands from the research-server checkout after syncing the code. The shells require Linux, the research filesystem and a Slurm job. No local empirical overrides are supported.

## Bundle-inclusive switching and flavor spells

```bash
sbatch Code/final_simple/server/run_bundle_statistics.sh \
  /scratch/dtm63837/Kilts_Panel/PATH_TO_EXISTING_SUMMARY_OUTPUT
```

Pass the summary folder containing `yogurt_weekly_flavor_quantities.csv` and
`aggregated_flavors/yogurt_weekly_flavor_quantities.csv`, not a fit directory.
These are existing **server-only** weekly exports from `data/summaries.py`.
The job reuses both detailed flavors and the selected aggregated flavor grouping.
It does not rerun matching, summary sample selection, or demand estimation.

All mixed baskets are included. New statistics are:

- Flavor-set change rate, mean numbers of flavors added/dropped, and mean set
  turnover (`1 - intersection/union`, or Jaccard distance).
- Mean flavor-set changes per household, including households with no changes.
  This count depends on the observation window and is reported beside pair rates.
- Mean and median flavor-presence spell lengths, pooled over spells, overall
  and by flavor. A flavor remains present even when bought alongside other flavors.

Each measure uses both successive observed yogurt purchase weeks (allowing gaps)
and adjacent calendar weeks (gaps excluded from pairs and breaking spells).
Purchase-clock spell lengths count purchase weeks, not elapsed calendar weeks.
Changing package quantities without changing flavors is not a flavor-set change.
The first/last observed spells are included without a censoring adjustment.
No purchase in a calendar gap is inferred as a switch or a nonpurchase choice.

Recover **only** the printed
`/scratch/dtm63837/Kilts_Panel/results/bundle_stats_JOB_ID/RESULTS_ONLY` folder.
It contains aggregate CSV/LaTeX tables for `detailed` and `aggregated_flavors`,
plus `definitions.json`; no household IDs, weekly records, partitions or logs.
Existing source summaries are preserved. Use the aggregate spell statistics to
replace the presentation's older single-flavor spell statistic after the run.

Local arithmetic checks use synthetic in-memory fixtures only:

```bash
python3 Code/final_simple/server/test_bundle_statistics.py
```

## Normalize detailed flavor labels without collapsing to three groups

After syncing the code, submit from the server checkout:

```bash
sbatch Code/final_simple/server/run_flavor_family_statistics.sh \
  /scratch/dtm63837/Kilts_Panel/PATH_TO_EXISTING_SUMMARY_OUTPUT
```

Use the same detailed summary folder as the bundle-statistics job: it must
contain `yogurt_weekly_flavor_quantities.csv`. Do not pass `aggregated_flavors`.
The job reuses this export; it does not rebuild raw data or refit demand.

`flavor_families.py` merges recognizable abbreviations and naming variants
(for example, `st`, `strb`, and `strawberry`). It retains ingredient blends,
recognized dessert styles and toppings. Strawberry-banana and strawberry-orange
remain distinct from strawberry; strawberry cheesecake also remains distinct.
Unexplained descriptor text retains the original label rather than being guessed,
dropped, or pooled into an other category. The mapping is a conservative
normalization, not a validated new flavor taxonomy.

Only recover
`/scratch/dtm63837/Kilts_Panel/results/flavor_families_JOB_ID/RESULTS_ONLY`.
It contains `summary_comparison.csv`/`.tex` (detailed versus normalized statistics),
`flavor_family_mapping.csv` (labels and their mapping only, without counts),
`mapping_coverage.csv`/`.tex`, pooled by-flavor spell tables, and `definitions.json`.
Inspect mapping coverage and retained ambiguous labels before using the new
statistics in the slides. Recoverable statistics contain only means, medians,
and percentages. Sample counts, flavor totals, spell counts, household IDs and
weekly records are not written to this output. Existing samples and results are
preserved.

Synthetic local checks:

```bash
python3 Code/final_simple/server/test_flavor_family_statistics.py
```

## Change grouping or rerun summaries and estimates

```bash
sbatch Code/final_simple/server/run_after_data.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

Optional second argument: full descriptor-reference CSV/Parquet. This job reuses the original completed matched data, runs separate three/five summaries and uncontrolled estimates, and runs plain/berry pricing for each. It creates `after_data_NEW_JOB_ID` beneath the original rebuild. Source samples and original results are preserved.

## Reuse a completed original-share uncontrolled fit for pricing only

```bash
sbatch Code/final_simple/server/run_split_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

This job expects `fits/no_control` in the specified directory and regenerates its matching weekly table. It does not rerun summaries or demand estimation. It refuses controlled or experimental novelty fits.

The pricing export includes `pricing_revenue_gain.pdf` for each flavor. A successful type-information calibration also produces `information_price_paths_comparison.pdf`, which compares type-specific offers with common offers under equal type probabilities for both initial states.

## What to inspect and copy

Copy only the printed `RESULTS_ONLY` path. The after-data job includes separate `three` and `five` directories, estimate/pricing comparison CSVs and `run_status.json`. Start with that status, then each specification's fit summary and pricing provenance. A failed pricing stage is not reported as complete.

Coupon benchmarks can be complete even when the separate binary type-information calibration is rejected. Each flavor directory has `information_calibration_status.json`; a failed calibration also provides aggregate price-moment/optimizer diagnostics and no information-value figures. This is not a valid zero information-value estimate.

The export verifies uncontrolled-fit parameter hashes and restricts allowed filenames. Do not copy weekly inputs, prepared samples, raw/household data, data partitions or logs. Full internal manifests stay on the server. The textbook's current-workflow chapter explains the function order and revenue definitions.

`run_novelty_estimation.sh` is a retained, unadopted lagged-switch experiment. It is not used by either current command.

## Revised novelty-information exercise

Use `run_information_pricing.sh EXISTING_FIT_DIRECTORY [plain|berry|both]` for
history-adaptive prices in both information regimes. Only novelty is hidden;
price sensitivity and habit are known. This reuses the saved fit without
estimating demand. See [the model and run guide](../demand/INFORMATION_PRICING.md).
The older shared schedule is a different benchmark.
