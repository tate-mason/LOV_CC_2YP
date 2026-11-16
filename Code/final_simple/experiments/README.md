# Separate estimation and quick substitution checks

These scripts leave the baseline demand/pricing pipeline unchanged. Empirical
runs require the research server and inputs/outputs under the approved research
root. Only estimates, diagnostics and aggregate metadata are written. Use a new
output directory for each run. No household histories or policies are exported.

## Demand variants

`estimate.py --spec novelty_only` removes the own-stock habit coefficient and
its dispersion. Flavor intercepts remain common; price sensitivity is lognormal
and novelty gamma is normal, independent and persistent across household weeks.
The price-control first stage uses novelty, with habit weight set to zero.

`estimate.py --spec random_beta_common_alpha` gives **every flavor**, including
other, an independent normal random intercept. Mean flavor intercepts are
parameterized by an inside constant and premiums relative to other; random
intercepts are anchored relative to the outside option. Alpha is common and
negative, gamma is heterogeneous normal novelty. Baseline heterogeneous
own-stock habit is retained. All random coefficients are independent and
persistent across weeks. This specification requires an outside option to
identify random beta for every flavor.

Both use the baseline data construction, sample restrictions, history measures,
optional price-control implementation and fixed antithetic integration draws.
They use separate analytic likelihood scores, multiple starts and Hessian
inference conditional on the first-stage controls and fixed draws. A zero SD
is a boundary hypothesis; ordinary normal SD tests are not provided.

Run from the checkout on the server, substituting the same approved input paths
used for the baseline. `$EXPERIMENT_OUTPUT` must be a new server directory:

```sh
python3 Code/final_simple/experiments/estimate.py \
  --spec novelty_only --sample "$SERVER_SAMPLE" --panel "$SERVER_PANEL" \
  --flavors "$SERVER_FLAVORS" --output "$EXPERIMENT_OUTPUT"
```

Run again with `--spec random_beta_common_alpha` and another output directory.
Use `--no-price-control` to match a baseline without price controls. These are
separate fits, not automatically consumed by current pricing scripts.

## Quick substitution test

```sh
python3 Code/final_simple/experiments/cross_price.py \
  --sample "$SERVER_SAMPLE" --panel "$SERVER_PANEL" \
  --flavors "$SERVER_FLAVORS" --output "$CROSS_PRICE_OUTPUT"
```

Two contemporaneous level equations, including zero purchases:

- Q_berry = b_berry p_berry + b_plain p_plain + household FE + week FE + error.
- Q_other = b_other p_other + b_plain p_plain + household FE + week FE + error.

Each equation instruments both prices, using both corresponding flavor-specific
same-week mean prices outside the consumer's DMA in both first stages. The
baseline currently repeats a **pooled** price IV across flavors, so this script
constructs separate instruments independently; it does not change the baseline.
Quotes are deduplicated by store/week/UPC/DMA/flavor/price before averaging.
These quick instruments do not hold UPC composition fixed. They require
cross-market demand shocks and product mix not to contaminate the exclusion
restriction; national promotions and chain-wide pricing are potential threats.
See [Nevo's cross-market instrument discussion](https://www.tse-fr.eu/sites/default/files/medias/doc/conf/fpi/papers_2000/nevo.pdf).

A positive plain-price coefficient supports substitution with plain, conditional
on IV validity and the equation's limited controls. This is a quick diagnostic,
not a dynamic response or a complete demand system; omitted flavor prices and
changing quote composition can affect interpretation. Coefficients are quantity
units per price unit, not elasticities. Quantity uses the baseline weekly flavor
purchase count. No sample selection on whether plain's price changed.

Outputs: coefficients and confidence intervals, first-stage diagnostics for
both endogenous prices, and aggregate sample information. Covariance clusters
by DMA and week using `linearmodels`; no small-sample absorbed-FE degrees-of-freedom
correction. The first-stage Wald statistics are not conditional weak-IV F tests
for multiple endogenous prices, and exactly identified equations cannot test
exclusion via an overidentification test. Rank failure stops the regression.
Dependencies: baseline environment plus `linearmodels`.

## Numerical checks (no research I/O)

```sh
python3 -m unittest discover -s Code/final_simple/experiments -p test_models.py
```

Checks compare analytic scores with numerical derivatives, confirm no-habit and
common-alpha parameter restrictions, recover known IV coefficients, and verify
that empirical entry points reject execution on the local computer.
