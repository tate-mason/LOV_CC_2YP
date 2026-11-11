# Completed results now imported

The run in `new_pricing_res/` is complete and its aggregates have been imported.
The new figures and conclusion are live in the main deck. Knowing novelty adds
0.15% initially inside and 2.21% initially outside under the fitted distribution.
See `PRICING_RESULTS.md` for mean price paths, revenue levels and validation.
Earlier pending-run language below describes the rebuild workflow.

# Revised pricing comparison

The theoretical model explains why a current price changes purchases and future
demand. The empirical extension retains the fitted multinomial utility, including
novelty and habit, and solves the same two-period timing with more state variables.
The binary formulas are a benchmark; the empirical optimal prices are recomputed.

In both regimes the firm knows price sensitivity, habit preference, initial
history, menus and rival prices. It observes the first-period basket before setting
the second price. Full information also reveals novelty preference. Limited
information knows its fitted distribution and holds its weights fixed. Prices
are common across novelty types within that known environment, not necessarily
across all households. Both first- and second-period prices can differ by history.

Consumers retain their type. Evaluate realized revenue for each type through both
periods and average only afterwards. The limited firm's fixed-belief optimization
objective need not equal actual revenue under persistent types. The benchmark
restricts learning; it is not the Bayesian optimum for an uninformed firm.

The old shared schedule was chosen before purchase outcomes, restricted prices
to reductions from observed prices, and left the middle group at posted prices.
It is not the revised information comparison. It remains supplementary. Its
+7.23% figure is not a result for the new regime and is not in the main conclusion.

The other earlier binary calibration uses three novelty group means and a common
intercept/price slope. Those figures are in backup, labeled earlier illustrations.
The revised full-model results require a pricing-only server run. No demand refit
is needed if saved history construction is retained. The displayed estimates are
cumulative-history estimates; an average-history demand model is a separate refit.

The new figures mirror supply: prices versus gamma, price ratios and profit by
four histories, type-specific and common price paths, revenue levels and gains.
An added price-change plot directly answers how offers move from observed prices.
Empirical curves average over known environments; terminal history distinguishes
at least one focal purchase from no focal purchase while the solver keeps full
basket states. These definitions appear in captions and provenance.

Import recovered RESULTS_ONLY outputs with refresh_information_pricing.py. The
main deck automatically shows the new result figures once imported. Until then,
it explicitly says results are pending; it does not reuse an old revenue gain.
