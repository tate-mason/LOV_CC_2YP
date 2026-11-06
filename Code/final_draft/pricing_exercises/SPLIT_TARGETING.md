# Split-history pricing for each focal flavor

`run_split_targeting.py` uses the existing `heterogeneous_history` estimator:

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
python3 Code/final_draft/pricing_exercises/run_split_targeting.py \
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
python3 Code/final_draft/est_code/history_distributions.py --fit-dir /path/to/res_het_hist
```

`split_fit_distributions/with_control` and `without_control` contain exports from
the existing local `shell_files/res_het_hist` and `res_het_nocf` fits. No new demand
optimization was necessary to recover these already estimated distributions.

Pricing saves one folder per focal flavor, each with household posterior means
and SDs for switching and habit, eligibility, household policies, group and
aggregate revenues, figures, and optional compressed state policies. Internal
solver coordinates swap the focal flavor into the berry slot; public household
outputs use `focal_*` names. The manifest records the focal flavor. The root
`focal_flavor_comparison.csv` stacks aggregate policy results across the three
cases. Every outcome is from the same split fitted demand model; no-coupon
revenues supply the benchmark. No sampling confidence intervals are computed.

The current local saved fit directories do not include `weekly_inputs.csv`;
full empirical pricing must run where those estimation outputs reside. The new
`shell_files/est_split_pricing.sh` exports both distributions and runs all three
focal cases using the server's controlled split fit. It does not refit demand.
To refit, run `est_het_history.py --capacity 7 --draws 64 --starts 3` with the
appropriate sample/output paths first. Estimation retains the same likelihood;
only its distribution exports have changed.

## Checks

```bash
python3 Code/final_draft/pricing_exercises/run_split_targeting.py --self-test
```

Synthetic checks compare posterior weights with a scalar likelihood, test exact
binary aggregation, compare backward induction with exhaustive scalar basket
calculations using independently chosen novelty/habit coefficients, and exercise
all three focal cases under both type treatments, including plots and exports.
