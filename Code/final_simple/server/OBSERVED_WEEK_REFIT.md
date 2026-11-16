# Corrected observed-shopping-week refit

Sync this correction to the existing server checkout. From that checkout:

```bash
sbatch Code/final_simple/server/run_observed_week_refit.sh \
  /absolute/server/path/to/original/three/fits/no_control \
  /absolute/server/path/to/completed/data/merged_panel.parquet
```

The first argument must contain the original `fit.json` and
`point_estimates.csv`, not the stripped RESULTS_ONLY export. The saved settings
locate the already prepared yogurt sample, so summaries/matching are not rerun.
If that prepared sample moved, append `--sample /new/server/path/sample.parquet`.
The shell uses the existing `hugemem_p` partition, eight CPUs and 96G; override
partition/resources with sbatch options if another available queue starts sooner.

The job runs small in-memory checks on the server first. It retains the original
selected yogurt-purchasing households and single-person restriction. Within that
sample it includes every week appearing in the full shopping panel, rather than
requiring a yogurt purchase. It does not invent weeks without shopping records
or add households absent from the original prepared sample.

Each genuine zero-yogurt week has zero inside counts and seven outside counts.
Full-panel yogurt totals are checked against matched totals; weeks with missing
or invalid yogurt quantities or incomplete matching are excluded and counted,
never labeled as nonpurchase. The existing above-seven rule is unchanged.

Existing purchase-week exposure stores/prices are retained. For a zero-purchase
week, exposure is the observed store with most distinct shopping trips, breaking
ties by store code. Missing flavor prices use that week's consumer-DMA flavor
means from the existing quote universe. No prices are carried across time. All
flavors must have a covered price in a zero week; otherwise it is excluded and
counted in the sample audit. This is an explicit additional price-exposure
assumption, required because zero weeks have no yogurt-purchase store.

History uses zero vectors without division by zero. Novelty following an
outside-only week is zero, as in the existing missing-previous-purchase rule.
Habit remains the average of prior observed purchase-week shares: zero weeks
leave that average unchanged. Including zeros in the likelihood does not
redefine that established habit measure.

A new sibling `observed_weeks_JOBID` directory preserves all original inputs
and fits. The job prepares the weekly table once, then runs:

1. `quick_32`: one warm start, 32 draws, no standard errors. Preliminary.
2. `refined_128`: warm start from step 1, 128 draws, Hessian standard errors.

Whole household histories are distributed across CPU workers. Each worker uses
the same simulated draws; objective and gradient are the household-weighted
aggregate of their original calculations. This does not change the likelihood.
The job checkpoints current parameters every ten iterations. Run time depends
on server queue and observed sample size; completion within three hours is not
guaranteed. `--quick-only` stops after the first pass; `--maxiter` changes the
per-stage limit (default 300). Inspect convergence and bounds before use.

Watch `/scratch/dtm63837/Kilts_Panel/observed_week_refit_JOBID.log` **on the
server**. It prints the new directory, sample audit, iteration progress, and
the exact results-only recovery path after each stage. Only copy the printed
`RESULTS_ONLY` folder back. Weekly inputs and full fit settings remain server-only.
The first-pass output can be recovered while refinement runs. The existing
quantified-theory script can then consume the new `point_estimates.csv` locally.

Local syntax checks and in-memory calendar/count/price checks passed. Full
SciPy/parallel/gradient tests are included in the submission shell and must
pass in the server environment before empirical preparation starts.
