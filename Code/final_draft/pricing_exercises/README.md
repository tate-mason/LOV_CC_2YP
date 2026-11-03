# Two separate pricing exercises

The theoretical supply extensions and empirical targeting exercise have separate,
standalone scripts. Neither script imports or changes the estimation script.
Dependencies are NumPy, pandas and matplotlib; SciPy is not required.

## Supply: baseline and both extensions in one file

```bash
python3 pricing_exercises/run_supply_exercises.py
```

This runs the known-state baseline, hidden-initial-state pricing, and ex ante
commitment straight through. It saves optimal prices in both periods, all four
history-specific price ratios, optimal profits, ex ante comparisons, numerical
checks, and PNG/PDF figures. Defaults cover 61 gamma values and three priors.

For a short run:

```bash
python3 pricing_exercises/run_supply_exercises.py \
  --gamma-points 7 --priors 0.5 --output pricing_exercises/supply_short
```

See [SUPPLY_EXERCISES.md](SUPPLY_EXERCISES.md) for equations, ratio definitions,
profit conventions, output columns and numerical limitations.

## Empirical: household-specific berry targeting

This follows the base supply model's known-type, observed-history pricing logic.
It does not impose uniform/group prices or apply the new theoretical information
restrictions. Gamma quartiles define reporting groups, not pooled pricing groups.

Run yourself with explicit paths:

```bash
python3 pricing_exercises/run_empirical_targeting.py \
  --household-input "/YOUR/EXPLICIT/HOUSEHOLD_INPUT.csv" \
  --fit-dir "/YOUR/EXPLICIT/FITTED_RESULTS_DIRECTORY" \
  --output pricing_exercises/empirical_outputs
```

The fit directory is the existing fitted-results directory (for example your
chosen `res_hess` directory). No demand re-estimation is performed. Input paths
have no defaults, and no empirical file discovery is performed.

The default treats posterior mean household types as known and optimizes
household-specific coupons by backward induction over every 1% discount from
0% through 100%. Timing is each household's first observed choice establishing `h0`, then
`p1(h0)`, a basket establishing `h1`, and terminal `p2(h1)`: three choice
dates with two pricing decisions. The first observed menu, prices and controls
are carried forward; estimated types still use the complete sample.
Terminal coupons depend on
realized consumption histories. Low/high gamma households receive personalized
offers; the middle group remains at posted prices. Dynamic pricing is the only
optimized specification; no coupons supplies the revenue-gain benchmark.
Revenue is net of coupons.

See [EMPIRICAL_TARGETING.md](EMPIRICAL_TARGETING.md) for the complete timing,
posterior construction, policy assumptions, output definitions, optional posterior
uncertainty integration, and extensions beyond two periods.

## Checks and data boundary

```bash
python3 pricing_exercises/run_supply_exercises.py --self-test
python3 pricing_exercises/run_empirical_targeting.py --self-test
```

Supply validation uses theoretical parameters. Empirical validation generates
synthetic households, including temporary synthetic input files for export tests.
No restricted empirical input is accessed for development checks. Empirical
results have not been computed here.

`run_pricing_exercises.py` is a compatibility launcher: by default it runs the
standalone supply script; the old `--exercise households` option forwards to the
standalone empirical script. Use the explicit entry points above. The earlier
`outputs/` and `short_supply/` directories contain the earlier supply-only
artifacts; the current supply script writes `supply_outputs/` by default.
