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
