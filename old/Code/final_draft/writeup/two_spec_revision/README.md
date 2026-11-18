# Two specifications, restarting after the data build

Submit from the research-server checkout after syncing the updated source:

```bash
sbatch Code/final_simple/server/run_after_data.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

The optional second argument is the full descriptor-reference CSV/Parquet. Otherwise the shell prefers the original run's recovered CSV reference, then the existing default Nielsen descriptor reference. A resolved three-category UPC mapping is not a substitute for the full descriptors needed for the new grouping.

The job reuses `run_ORIGINAL_JOB_ID/data/merged_panel.parquet`, its matched `yogurt_estimation_sample.parquet`, and the existing converted retail metadata. It does not invoke conversion, cleaning, annual combination, panel/retail merging, or matching. It reruns summaries and constructs separate labeled estimation inputs on the server. Existing results and user files are preserved.

## Specifications

- `three`: original groups `other`, `berry`, `plain`, with the original descriptor rule retained.
- `five`: `other`, `berry`, `plain`, `other_fruits`, `desserts`. Plain/unflavored retains precedence; dessert cues take priority over fruit cues. Vanilla, chocolate, pies, cakes, cookies, custard, caramel, and related named desserts go to desserts, including strawberry cheesecake and banana cream pie. Remaining berry labels go to berry; non-berry fruits go to other fruits; residual flavors remain other. Cherry/black cherry goes to other fruits in the expanded specification, while the original three-group rule continues to classify cherry as berry. Coffee/honey without dessert or fruit cues remains residual other. These rules operate on the combined available flavor/product descriptors, as in the original pipeline.

Both specs retain the original share-based novelty `N = 1 - own share in the preceding calendar week`, the existing cumulative own-share stock, seven-choice capacity, 128 draws, three starts, Hessian inference, and **uncontrolled** heterogeneous-history demand. Price menus, fixed effects, histories, likelihood gradients, and basket enumeration use the appropriate number of categories. Plain and berry remain the two focal sellers; all five alternatives stay in expanded demand and pricing states. At capacity seven, exact basket states rise from 120 to 792. The full 101-point discount grid is retained, so the expanded pricing exercise is slower.

The pipeline estimates both specifications before starting either pricing exercise. Each pricing exercise is attempted even if the other's pricing subprocess fails.

## Screenshot error

The posted traceback is the auxiliary binary supply calibration rejecting its numerical result. It does not identify whether the failing condition was convergence, a bound, or rank. The code now increases the optimization budget and chooses the best admissible solution across starts. Bound/rank checks remain in place. If the binary calibration is still unsupported, it writes aggregate price moments and optimizer diagnostics and marks the information exercise failed. Coupon revenues and policy benchmarks still finish and export; rejected information-value charts are omitted. No fallback calibration or fabricated information result is substituted.

When calibration succeeds, offers are chosen using fixed beliefs and policy revenues are evaluated under persistent consumer types, as agreed.

## Results to copy

Each submission creates a fresh server directory:

```
run_ORIGINAL_JOB_ID/after_data_NEW_JOB_ID/
  three/                       # server-only summaries, fit and pricing work
  five/                        # server-only summaries, fit and pricing work
  RESULTS_ONLY/                # the only directory to recover locally
    three/
      summary_statistics/
      no_control_fit/
      pricing/plain/
      pricing/berry/
      empirical_information/   # only successfully calibrated figures
      pricing_provenance.json
    five/
      ...
    estimate_specification_comparison.csv
    pricing_specification_comparison.csv
    run_status.json
```

Read `run_status.json` first. Each flavor's calibration status is also in its provenance and `pricing/<flavor>/information_calibration_status.json`. Diagnostic moment CSVs are aggregates without household IDs. Exports contain explicit allowed estimates, aggregate tables/figures and sanitized provenance. No raw panels, household tables, weekly inputs, UPC mappings, data partitions, or logs may be copied locally.

Both estimation inputs and summary intermediates stay on the server. Job-created weekly likelihood tables are removed after each pricing attempt. The original completed data build is reused and never removed. The old fit and result directories are not replaced. Presentation TeX is unchanged.

This change has been checked locally using abstract multinomial count arrays, descriptor strings, and theoretical price moments only. No empirical pipeline has been run locally or submitted remotely from this session.
