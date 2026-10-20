# Toy model numerical results

Uniform means common across types; prices vary by product and period-one purchase history.
Consumers are myopic. Period-one outside choices exit. Initial history is pooled.
Profits are per initial consumer, with N normalized to one.

50/50 common profit: 0.97727336

50/50 targeted profit: 0.99664374

50/50 targeting gain: 0.01937038

These are incumbent-history benchmarks. Newborn consumers have no purchase history; their discount decision is reported separately in newborn_results.md. No fixed adoption cost or noisy classifier is assumed.

profit_by_gamma.csv reports conditional expected profits, not a probability density: a density over profits requires specifying a distribution over gamma. The population experiment uses two point masses, H and L.

Optimization uses three starts and checks solver status; this is numerical evidence, not a proof of global optimality. See prices_50_50.csv for all prices.
