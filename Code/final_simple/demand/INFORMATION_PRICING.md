# History-adaptive pricing when only novelty is hidden

This is the theory's information comparison applied to the saved multinomial,
habit-inclusive demand model. Demand parameters are not refitted. The previous
shared price schedule and the separate binary price calibration remain earlier
exercises; neither supplies the revised results.

Both regimes observe the same initial purchase basket and the first-period
basket. Price sensitivity and the habit coefficient are treated as known estimated
household posterior means. With full information, the seller also observes the
novelty coefficient. With limited information, only novelty is hidden. Common
prices are common across possible novelty types *within the same known-price,
habit, history, rival-price and menu environment*. Prices can differ across those
known environments and can respond to the first-period basket.

## Consumer model and history

Use the exact saved utility, including flavor constants, price sensitivity,
novelty, own-stock habit, nonpurchase normalization, and any saved fixed price
controls. Rival prices and menus stay fixed. Own price can rise or fall from its
observed level. The first price can be negative, as in the theoretical model;
the terminal optimum is nonnegative at zero marginal cost. Every novelty type is
optimized, including the middle group.

The first observed basket establishes the initial state; pricing covers the next
two consecutive calendar weeks. Both regimes use the same basket-history update.
The saved fit selects cumulative, average or decaying habit construction. A fit
without history metadata retains the original cumulative construction. The new
average-history default is never silently applied to an old fit.

This is an extension of the binary theoretical model, not a claim that its
closed-form formulas still apply. Novelty measures other-flavor purchase share;
habit is a separate state. Both enter the continuation calculation.

## Distribution and evaluation

The fitted normal population novelty distribution is integrated by Gauss-Hermite
quadrature (nine nodes by default). It is crossed with each observed environment,
conditional on known price/habit tastes. This is a model-based population
experiment, not a prediction assigning a new true novelty type to an identified
household. Support and weights are recorded in aggregate metadata. Low/middle/high
labels use the population lower/upper-quartile cutoffs. Sensitivity cases assign
one group probability one-half and the others one-quarter, retaining relative
weights within groups. The baseline retains the fitted population distribution.

Consumers keep their novelty type in both periods. Limited-information beliefs
remain fixed after a purchase, matching the theory's behavioral restriction.
Common terminal prices maximize the fixed-belief mixture revenue in each state;
the first price maximizes that fixed-belief continuation objective. This perceived
objective need not equal realized revenue under persistent types. Reported
information gains always evaluate *both* policies under the same persistent
types. This is not a Bayesian-learning optimum.

First-price optimization screens stationary points on a dimensionless price grid
and refines all detected local maxima. Terminal mixture prices are bracketed by
the type-specific optima and refined by first-order conditions. State probabilities
are exact multinomial probabilities over all baskets. Current first-order tests,
independent direct enumeration, grid refinement, degenerate-type equivalence and
full-information dominance checks pass. Numerical screening is not a formal
proof that all stationary points are found for arbitrary utility scales; wider
grid checks and quadrature sensitivity are available with command-line options.

## Run on the research server

After syncing source, submit using the **fit directory**, not its parent build:

```bash
cd /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP
sbatch Code/final_simple/server/run_information_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID/fits/no_control plain
```

An after-data fit can instead be selected with its full path, such as
`.../after_data_JOB_ID/three/fits/no_control`. Use `both` instead of `plain` to run
plain and berry. The job rejects non-server execution and existing destinations.
It reconstructs the sample in memory, verifies the saved likelihood, and does not
modify the saved fit or persist weekly inputs.

A small server-only check can call `demand/information_pricing.py` with
`--max-environments 20 --fit-dir ... --output NEW_DIRECTORY`. Such a run is labeled
as limited coverage and must not be presented as the complete exercise. For
numerical sensitivity, change `--nodes` and/or `--search-points` in a separate run.

## Outputs and supply-style plots

Recover only `/scratch/dtm63837/Kilts_Panel/results/information_pricing_JOB_ID/RESULTS_ONLY`.
It contains aggregate tables and figures, not individual records or policies.

For each focal flavor:

- `optimal_prices_by_gamma.pdf`: first and terminal price panels, matching supply.
- `price_ratios_and_profit.pdf`: four initial/realized-history combinations.
- `information_price_paths_comparison.pdf`: known types and common prices by initial state.
- `information_revenues.pdf` and `information_value.pdf`: levels and gains by prior.
- `price_changes.pdf`: price movements relative to observed posted prices.
- Three aggregate accounting CSVs, an information-gain table, and figure provenance.

Gamma curves average over observed environments and connect quadrature nodes.
Terminal history means at least one focal purchase versus no focal purchase;
full basket transitions are retained inside the solver. Ratios are conditional
means of realized price ratios, and profit means expected revenue at zero cost.
All plots distinguish these definitions from the simpler binary supply model.

The entire job's `information_pricing_summary.json` records nuisance-parameter
knowledge, gamma integration, history construction, likelihood and source hashes.
No empirical run has been completed locally. Pure numerical checks can run with:

```bash
python3 Code/final_simple/checks/check_information_pricing.py
python3 Code/privacy/check_policy.py
```
