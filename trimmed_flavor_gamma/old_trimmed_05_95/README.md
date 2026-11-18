# Category switching with flavor tastes

Scenario: Total welfare.

Yogurt-versus-no-yogurt switching on observed shopping weeks, including zeros. Initial state is the first actual shopping week. Both priced periods are further observed shopping occasions, not necessarily adjacent calendar weeks. Future choices are endogenous.

Open common_price/pricing_results.pdf and separate_flavor_prices/pricing_results.pdf. Tables are saved separately in those folders. Uniform does not observe type; Informed knows its quintile. Neither learns after purchase. Alpha and flavor beta are fixed in pricing.

Under one owner, equal zero costs, a common alpha and category-only histories, the separate-price optimum has equal flavor prices. Flavor constants affect category demand through log-sum-exp. This equality is not evidence that flavor tastes were omitted.

Welfare/revenue changes use 100*(Informed-Uniform)/abs(Uniform); zero baselines are undefined. Revenue-contribution tables report shares as separately requested.

parameter_comparison.csv compares coefficients to the earlier flavor-switching fit. Gamma now describes category switching, so the two estimates have different meanings. Any previous flavor-seller results also have different product scope and are not a matched causal counterfactual. No empirical records or household identifiers are exported.

Gamma retains original percentiles 5–95, with the retained mass renormalized to one. Tails are removed, not capped. Each new quintile has 20% of retained probability. Initial-state weights are held fixed; actual household records are not trimmed locally.
