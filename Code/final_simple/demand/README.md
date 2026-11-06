# Read the revised demand workflow

The model and pricing changes are implemented here with editable Python settings
and numbered calculation sections. `pricing.py` keeps the earlier combined-gamma
berry exercise; **`split_pricing.py` is the current exercise**. Both use a visible
one-household loop. No research file imports `final_draft`.

The full [Python 101 textbook](../../final_draft/code_guides_pdf/python101_walkthrough.pdf)
now includes this workflow. Start with worked lessons 11–16, then use its
function-by-function chapters for the four readable demand scripts.

## Plain and berry comparison

Use `plain_berry_pricing.py` for the main two-focal comparison. It reuses the
split solver and retains other, berry and plain in demand, posterior recovery
and history updates. Only plain and berry receive focal-seller counterfactuals;
other is always part of the composite outside option. Outputs are separate
`plain/` and `berry/` folders plus `focal_flavor_comparison.csv` and run records.
No re-estimation is needed. Run on the server after regenerating the weekly file:

```bash
python3 Code/final_simple/demand/regenerate_weekly_inputs.py \
  --fit-dir /path/to/res_het_hist
python3 Code/final_simple/demand/plain_berry_pricing.py \
  --household-input /path/to/res_het_hist/weekly_inputs.csv \
  --fit-dir /path/to/res_het_hist --output /path/to/plain_berry_pricing
```

The all-three `split_pricing.py` exercise remains available as a sensitivity.

Read in this order:

1. `estimation.py`, Parameters and sections 3 and 6: split-coefficient likelihood
   and fitted population distributions. Choose `model = "heterogeneous_history"`
   and `capacity = 7` for a new fit matching the saved pricing specification.
2. `regenerate_weekly_inputs.py`: recover the deleted server export from saved
   settings, refitting controls and checking the likelihood without refitting demand.
3. `history_distributions.py`: export both population distributions from saved fits.
4. `split_pricing.py`, Parameters, sections 1–4 and the final run function:
   posterior types, initial history, backward induction, three focal cases.
5. `../server/run_split_pricing.sh`: execute regeneration, distribution exports,
   and the plain/berry comparison on the server. Submit from the repository root with
   `sbatch Code/final_simple/server/run_split_pricing.sh` after syncing code and
   installing `Code/final_simple/requirements.txt` in your server environment.

# Split-history pricing for each focal flavor

`split_pricing.py` uses the existing `heterogeneous_history` estimator:

    V_ijt = inside_constant + beta_j - a_i p_ijt + b_Ni N_ijt + b_Si S_ijt + rho cf_ijt
    b_N ~ Normal(mean_novelty, sd_novelty)
    b_S ~ Normal(mean_own_stock, sd_own_stock)
    a ~ Lognormal(mu_log_minus_alpha, sd_log_minus_alpha)

The population coefficients are independent in the fitted specification; posterior
coefficients need not be independent. Positive b_N rewards switching from recent
other-flavor purchases. Positive b_S reinforces accumulated own-flavor purchases;
negative b_S represents satiation. Neither coefficient is multiplied by lambda.
All three random coefficients persist over the full household history.

From the repository root:

```bash
python3 Code/final_simple/demand/split_pricing.py \
  --household-input /path/to/res_het_hist/weekly_inputs.csv \
  --fit-dir /path/to/res_het_hist --output split_pricing
```

The default `--focal-flavor all` solves **three separate seller problems**, with
other, berry, and plain respectively as the focal inside flavor. The remaining
two flavors form the composite outside alternative. Nonpurchase also enters
the outside alternative when the fitted capacity convention includes it.
Only the focal flavor's price changes and only its sales earn seller revenue.
These are separate counterfactuals with fixed rival prices, not simultaneous
three-firm equilibrium prices. Use `--focal-flavor berry` for just one case.

For each focal j, the exact binary probability is

    P_j = logistic(V_j - logsumexp(V_k for k != j, V_nonpurchase=0)).

This is aggregation of choice probabilities, not averaging rivals' utilities.
The code retains rival flavor identities and integrates their baskets so that
novelty and cumulative stock remain consistent with the estimated demand model.
The full flavor basket is observed before the terminal offer. Combining rivals
for the seller's inside/outside comparison does not make their future history
states interchangeable.

## How it matches the supply exercise

The firm's known estimated type is `(a_hat, b_N_hat, b_S_hat)`. Household posterior
means use likelihood weights from all observed weeks, with the estimator's exact
three-dimensional antithetic draws. The saved likelihood must be reproduced
before pricing starts. Households are ranked by **b_N only** into bottom-quarter,
middle-half, and top-quarter reporting groups. Both b_N and b_S enter all demand
forecasts. High/low groups receive individually optimized coupons; the middle
stays at posted prices. The groups are common to all three focal cases; initial
menu eligibility can differ. Ranking does not hold habit or price sensitivity
fixed, so group comparisons do not isolate a causal switching effect.

Timing matches the known-state supply baseline: an observed basket establishes
h0; p1(h0) induces a basket h1; terminal p2(h1) is chosen after observing h1.
Backward induction maximizes current net revenue plus discounted expected
optimized terminal revenue, integrating **every** basket, including no focal
purchase and all nonpurchase. Both continuation branches matter. The default
uses known posterior mean types; `--type-treatment posterior-integrated` is a
separate sensitivity that integrates persistent type uncertainty and updates it
after the first counterfactual basket.

This implements the supply theory's known-type, observed-state mechanism for
all three focal products. It is not an exact calibration of its symmetric
`+gamma/-gamma` binary Markov utility: b_N and b_S are separate, own stock is
cumulative, and the rivals' inclusive value can depend on their histories.
Consequently **b_N is the estimated switching coefficient, not an algebraically
identical theoretical gamma**. Using only b_N and dropping b_S would violate
the fitted demand specification. Binary aggregation is exact at each state;
compressing all rival histories to a single binary lag would not be exact.

Prices are constrained to posted-price coupons (default 0% through 100% in 1%
steps), whereas theoretical prices are unrestricted. Cost is zero, so net
revenue equals profit. Consumers follow current-period fitted demand, without
strategic delay. The empirical discount factor defaults to .92, matching the supply script. The first observed
menu and control residuals are held fixed across both pricing weeks. Initial
stocks add the first observed basket once. Known estimated types use the full
sample, making this a retrospective exercise. Constructed nonpurchase counts
and K forecast opportunities remain assumptions of the capacity specification.

## Distributions and saved outputs

The split estimator now writes `history_distribution_summary.csv` (normal
parameters, sign shares, population quantiles), `history_distribution_grid.csv`
(densities and CDFs), and `history_distributions.pdf/png` immediately after fitting.
These are population heterogeneity distributions, not sampling confidence
intervals or histograms of posterior means. Common split coefficients produce
point masses. Existing fits can be exported without re-estimating:

```bash
python3 Code/final_simple/demand/history_distributions.py --fit-dir /path/to/res_het_hist
```

Copies of the earlier distribution exports are in
[`fitted_distributions/with_control`](fitted_distributions/with_control/history_distributions.pdf)
and [`fitted_distributions/without_control`](fitted_distributions/without_control/history_distributions.pdf). This directory's server workflow recreates those exports
from your selected fits. No new optimization is needed for already estimated
distributions. `demand/estimation.py` contains the export calculations directly;
`demand/history_distributions.py` is the entry point for a saved fit.

Pricing saves one folder per focal flavor, each with household posterior means
and SDs for switching and habit, eligibility, household policies, group and
aggregate revenues, figures, and optional compressed state policies. Internal
solver coordinates swap the focal flavor into the berry slot; public household
outputs use `focal_*` names. The manifest records the focal flavor. The root
`focal_flavor_comparison.csv` stacks aggregate policy results across the three
cases. Every outcome is from the same split fitted demand model; no-coupon
revenues supply the benchmark. No sampling confidence intervals are computed.

The current local saved fit directories do not include `weekly_inputs.csv`;
the deleted export must be regenerated on the server. The new
`server/run_split_pricing.sh` first rebuilds `weekly_inputs.csv` using the
controlled split fit's saved settings, reconstructs its price controls, and
checks household/week totals and the saved likelihood before writing the CSV.
It then exports both distributions and runs the plain/berry comparison. It does not
refit demand or replace the saved estimates or standard errors. Regeneration
and full empirical pricing are server operations; local checks use synthetic
fixtures only.

To regenerate the export separately on the server:

```bash
python3 Code/final_simple/demand/regenerate_weekly_inputs.py --fit-dir /path/to/res_het_hist
```

If the prepared sample or flavor reference moved, use `--sample` or `--flavors`
to override its saved path. A likelihood mismatch stops the job without writing
a replacement CSV.
To refit, run `demand/estimation.py --model heterogeneous_history --capacity 7 --draws 64 --starts 3` with the
appropriate sample/output paths first. Estimation retains the same likelihood;
only its distribution exports have changed.

## Checks

```bash
python3 Code/final_simple/demand/split_pricing.py --self-test
```

Synthetic checks compare posterior weights with a scalar likelihood, test exact
binary aggregation, compare backward induction with exhaustive scalar basket
calculations using independently chosen novelty/habit coefficients, and exercise
all three focal cases under both type treatments, with CSV reads blocked. Run `python3 checks/run_checks.py` from `final_simple`
for all synthetic checks and `python3 checks/compare_original.py ../final_draft`
for direct numerical comparisons. None of the research scripts imports
`final_draft`.


The server jobs default to saved fit directories
`$project_dir/shell_files/res_het_hist` and
`$project_dir/shell_files/res_het_nocf`. The estimation job's `--output` paths are
relative to its submission working directory; change the two fit-directory
variables if you submitted estimation elsewhere. A fit directory must contain
`fit.json` and `point_estimates.csv`; it is not a single file. Both jobs check
those artifacts before regenerating household inputs on the server.
