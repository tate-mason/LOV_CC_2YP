# Fixed regular prices and targeted discounts

Effective price = regular price - discount. Only good 2 can be discounted, in period two after buying good 1. All other prices remain at the fixed regular price.

Illustrative regular prices (2.00, 2.00), costs (1.20, 0.20), price coefficient -1.5, gamma_H=2, gamma_L=0.1, delta=0.95. The cost gap and fixed-price restriction replace the previous unrestricted-price calibration. The monopolist can set regular prices before the targeted-offer stage. This exercise is conditional on posted regular prices; it does not solve the upstream regular-price problem or claim a joint unrestricted monopoly optimum.

No prior purchase, myopic logit consumers, permanent exit after period-one outside. First purchases do not reveal type. Known type probabilities; no Bayesian learning or classifier. Profits are per initial consumer.

H discount: 0.00000000; L discount: 0.49333833; 50/50 common discount: 0.04934851.

50/50 common profit: 0.90215442; observed-type profit: 0.90809988; information gain: 0.00594546.

Offering the fixed L-optimal coupon instead of no coupon breaks even at Pr(L)=0.76402445. Continuous discount optimization is a different problem.

Fixed discount regions (q=Pr(H)):
- 0.00000000 < q < 0.21215451: low discount.
- 0.21215451 < q < 0.54295850: uniform_50_50 discount.
- 0.54295850 < q < 1.00000000: high discount.

The discount increases demand for good 2 and reduces its unit margin. In this calibration the L-optimal discount leaves good 2 with a larger margin than good 1. H has zero optimal discount; logit switching is nevertheless probabilistic, not guaranteed.

Only the good-1 history is eligible for a coupon. This isolates switching toward good 2; after good 2 a coupon would instead induce retention. Population profits include both first-period purchase histories.

Solver checks boundaries and all stationary roots bracketed by a 129-point derivative grid. Tests compare solutions with independent bounded maximization and dense feasible grids. Concavity of conditional mispricing profits is not imposed.
