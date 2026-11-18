# Free pricing with initial purchase histories

The active implementation is `toy_model.py`. Consumers arrive with observed initial state j, k, or 0, independently of type, with probability 1/3 each. Initial 0 carries no switching reward. Consumers choose myopically with logit shocks in both periods. A period-one outside choice exits permanently.

The firm freely sets both prices in both periods: there are no coupons, maintained regular prices, or imposed price bounds. Initial prices depend on initial state; terminal prices may depend on both initial state and first purchase. All menus commit before choices. The common menu objective accounts for the endogenous sorting of types into histories; first purchases are no longer assumed uninformative. Observed-type pricing also conditions on actual gamma.

For type r and initial state a, write terminal profit as w_r(a,h). Total expected customer value is

$$\pi_r(P)=\sum_a\mu_a\sum_{h\in\{j,k\}}s_{rh1}(a)\left[p_{h1}(a)-c_h+\delta w_r(a,h)\right].$$

The firm maximizes q*pi_H(P)+(1-q)*pi_L(P). Period-2 profit per arriving consumer includes the probability of surviving period 1; period-2 profit per survivor divides by that probability. These quantities must not be confused with a unit margin.

Known-type interior prices satisfy equal terminal product margins, -1/(alpha*s_02). First-period effective margins p_h1-c_h+delta*w(h) are equal to -1/(alpha*s_01). A larger continuation value therefore supports a lower current margin, holding the common effective margin fixed. The known-type solver uses these backward equations directly; heterogeneous common prices use three unconstrained BFGS starts with analytic gradients. This is numerical evidence for the common optimum, not a global proof.

Illustrative beta=(2.2,2), costs=(1.2,0.8), alpha=-1.5, delta=0.95. Good 2 costs 0.8 versus 1.2 for good 1; both prices remain freely optimized. The baseline is gamma_H=2, gamma_L=1. The sensitivity holds gamma_L=1 and q=0.5 and reoptimizes common prices as gamma_H varies from 1.5 to 10 in steps of 0.5. Observed-type optima are also reported as a benchmark. Induction sacrifice is known-type static one-period optimal profit minus dynamic period-one profit.

`updates/toy_model/results/gamma_H_sweep.csv` contains the decomposition, survival, and static benchmark. `value_distribution.csv` records the two expected-value atoms, each of mass 0.5. This is a distribution of expected values across types, after averaging over initial states and shocks, rather than a density of realized profits. A continuous value distribution requires additional assumptions about the distribution of gamma.

Rebuild from the repository root:

```sh
.venv/bin/python toy_model.py
MPLCONFIGDIR=/tmp/toy-mpl .venv/bin/python updates/draft_2/presentation/build_pricing_figures.py
.venv/bin/python -m unittest test_toy_model -v
```

Build `updates/toy_model/toy_model.tex` and `updates/draft_2/presentation/variety_pricing_slides.tex` with `latexmk -pdf` in their respective folders. Numerical macros and figures share the current CSV inputs. The former fixed-price exercise in `fixed_regular_price_variant/` and the no-initial-state unrestricted exercise in `archive_unrestricted/` are historical; neither is the active specification. Older coupon/newborn result filenames in the results directory are historical and not consumed by the current slides.
