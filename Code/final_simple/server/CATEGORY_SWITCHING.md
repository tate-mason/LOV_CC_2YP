# Optional pooled-category refit: weekly yogurt versus no yogurt

This is a new empirical specification, not an exact aggregation of the existing
flavor-share history model. It has not been run merely by creating these scripts.
Use the existing population-parameter bridge diagnostic before attributing
differences between old theory plots and empirical plots to parameter estimates.

## Observation and utility

One observed shopping week has one binary outcome: any yogurt (1) or no yogurt
(0). Multiple trips in that week are pooled. Flavor and package quantity do not
change that outcome. Genuine no-yogurt weeks are retained; calendar gaps do not
create observations. The previous actual observed shopping week supplies history,
even when its menu lacks sufficient price coverage to enter the likelihood.

For yogurt flavor j, relative utility is
`beta0 + beta_j + alpha_i*p_j + gamma_i*(1 - 2*previous_yogurt)`.
Other is the reference flavor. The one gamma is shared across flavors.
The likelihood integrates persistent alpha/gamma draws over each household's
entire history. Flavor utilities enter the binary likelihood through their
log-sum-exp. There are no seven-capacity outside counts, package-count likelihood,
within-week consumption sequences, habit stocks, or flavor-specific gammas.

The first actual observed shopping week initializes h0, including when it has
no yogurt. Category status comes from the full shopping panel, so an unmatched
flavor basket is not automatically an unknown category choice. The input cohort
remains the prior single-person matched-yogurt household cohort, not every
never-yogurt household in the panel. The audit reports the exact retained sample.

Menu exposures are same-store/week mean quoted UPC prices by flavor. Missing
exposures use same-week DMA means, never future or backward time interpolation.
All three flavor prices must be covered for a scoring week. These are exposure
indices, not prices imputed from the basket chosen by that household.

## Pricing and reports

Pricing uses fixed empirical mean alpha, fixed flavor intercepts, and the newly
estimated population gamma distribution, held identical across initial histories.
Uniform observes history but not type; Informed also observes the gamma quintile.
No posterior updating occurs during either priced week. Initial history is
observed; the following two weekly choices are endogenous under optimized prices.

Both common and separate flavor prices are reported. With one owner, common
alpha, equal zero costs, and category-only histories, the separate-price optimum
has equal flavor prices. The solver uses the exact common-markup reduction and
checks all three individual price derivatives and reported revenue accounting.

The maintained utility levels for welfare are
`V_j=beta0+beta_j+alpha*p_j+gamma*1{previous no yogurt}` and
`V_0=gamma*1{previous yogurt}`, as in the theory. Consumers are myopic. This
normalization is explicit and can affect experienced welfare through history.

## Submit on the research server

After syncing the updated source, run:

```bash
sbatch Code/final_simple/server/run_category_switching.sh
```

Optional arguments: prior no-habit fit directory, then previous aggregate pricing
results directory. Defaults use job 50281568's no-habit fit for input locations
and optimizer starting values; its gamma is **not** treated as the new estimate.
The job reconstructs observed weekly inputs in memory, runs a 32-draw quick stage
and a 128-draw full stage with three starts, and generates pricing only after a
converged non-boundary full-stage fit. Eight CPU workers partition entire household
likelihoods. No demand estimates, existing outputs, or research inputs are replaced.

Recover only:

```text
/scratch/dtm63837/Kilts_Panel/rebuilds/category_switching_JOB_ID/RESULTS_ONLY/
```

Open `common_price/pricing_results.pdf` and
`separate_flavor_prices/pricing_results.pdf`. Price policies, within-state welfare
changes, observed-initialization summaries, and quintile revenue contributions
are in separate tables under each pricing mode. Welfare/revenue changes divide
by the absolute Uniform baseline; a zero baseline is flagged as undefined.

`parameter_comparison.csv` compares coefficients with the old flavor-switching
fit; their history predictors differ, so gamma is not the same estimand.
`fit_check_by_history.csv` compares observed/predicted weekly yogurt participation.
`price_structure_comparison.csv` verifies common/separate-price equality.
If a previous results directory is supplied, its aggregate information gains
are preserved for reference; the product scope differs, so those are not matched
causal counterfactuals. Standard errors are not produced by this initial run.

All empirical I/O is guarded by the unchanged server-only policy. Nothing in
this workflow authorizes downloading household histories or weekly inputs.
