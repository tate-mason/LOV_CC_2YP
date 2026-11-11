# Completed pricing results imported

The main deck now includes the completed revised comparison from
`new_pricing_res/`. Only aggregate outputs were read. The run covers 1,839 eligible
environments and retains the existing cumulative-history demand fit. Under the
fitted distribution, information raises revenue 0.15% initially inside and 2.21%
initially outside. See `seminar_pres/PRICING_RESULTS.md` for price paths and checks.

# Short presentation: revised information comparison

The main empirical exercise retains the saved habit-inclusive demand model. Both
firms know history, price sensitivity and habit, and both can respond to the
first-period purchase. Only novelty preference is hidden in limited information.
Prices are common across novelty types within otherwise known environments;
they are not one fixed schedule across time or all consumers. Beliefs stay fixed.

The old binary calibration and shared-schedule pricing results are explicitly
labeled earlier benchmarks in the appendix. Their numerical gains are not used
as results for the revised exercise. The saved cumulative-history fit is retained;
the planned average-history specification is documented separately.

## Pricing-only rerun

Sync the source and submit on the research server:

```bash
cd /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP
sbatch Code/final_simple/server/run_information_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID/fits/no_control plain
```

The argument is the actual fit directory, including for an after-data run.
Recover only the job's RESULTS_ONLY folder. See
`Code/final_simple/demand/INFORMATION_PRICING.md` for definitions, validation and
quadrature/search sensitivity. The completed server run has been recovered as aggregate outputs in `new_pricing_res/` and imported. No empirical records were processed locally.

## Import aggregate outputs and build

From the local repository root:

```bash
python3 Code/final_draft/seminar_pres/refresh_information_pricing.py \
  --source /path/to/recovered/RESULTS_ONLY --flavor plain
```

This reads named aggregate outputs only and generates plots matching supply.
Once imported, the main deck shows the new price, ratio/profit, price-path,
price-change and revenue figures. Until then, one clearly labeled results-pending
slide appears. Frame counts adjust automatically. No old empirical gains stand
in for the new comparison.

Compile from `Code/final_draft/seminar_pres` twice. The source in
`writeup/shortened_presentation` also uses paths relative to that working directory;
it retains its separate introduction. Use distinct job names/output directories
if an editor is building concurrently. Do not regenerate from the obsolete
`build_short_version.py`; that would replace later user edits.

The identification explanation sits between the demand model and estimation.
The likelihood indexes weekly choices correctly, and model history notation is
explicit about observed purchase occasions.
