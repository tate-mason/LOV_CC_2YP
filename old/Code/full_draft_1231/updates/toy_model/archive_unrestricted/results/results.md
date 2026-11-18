# Toy model numerical results

Uniform means common across types; prices vary by product and period-one purchase history.
All prices are committed in advance using fixed, known type masses; no learning or belief updating.
Consumers are myopic. Period-one outside choices exit. Consumers have no prior purchase.
Profits are per initial consumer, with N normalized to one.

50/50 common profit: 0.68367095

50/50 targeted profit: 0.69428287

50/50 targeting gain: 0.01061192

The fixed terminal-discount decision for these same consumers is reported separately in newborn_results.md. No fixed adoption cost or noisy classifier is assumed.

profit_by_gamma.csv reports conditional expected profits, not a probability density: a density over profits requires specifying a distribution over gamma. The population experiment uses two point masses, H and L.

Optimization uses three starts and checks solver status; this is numerical evidence, not a proof of global optimality. See prices_50_50.csv for all prices.

The three fixed H, L, and 50/50 menu transitions are reported in fixed_menu_results.md and fixed_menu_transitions.png.
