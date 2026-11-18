# Old flavor-switching gamma: 5th–95th percentile sensitivity

Open comparison.pdf for distribution and aggregate information-gain comparisons. Open old_trimmed_05_95/common_price/pricing_results.pdf for the new pricing and welfare figures. Separate flavor-price outputs have their own folder under each scenario.

The primary comparison is old_uncut versus old_trimmed_05_95: alpha, all flavor betas, initial-state weights, no-learning policy and solver are identical. Only the gamma distribution changes. Both apply the old flavor-history coefficient as a category-history calibration; no demand fit is rerun. binary_reference uses all new binary-fit coefficients and is a separate reference, not the isolated effect of trimming.

Trimming discards the old distribution below its 5th percentile and above its 95th percentile and renormalizes the middle 90%. It does not cap observations. Each new quintile has 20% of retained mass (18% of original mass). All scenarios use the same aggregate first-observed-shopping-week counts from binary_est. No individual households or research records are read, removed, or reclassified locally.
