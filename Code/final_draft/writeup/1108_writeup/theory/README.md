# Switching-type uncertainty

Run `python3 gamma_uncertainty.py` (NumPy, SciPy, Matplotlib).
This is a theoretical exercise and reads no empirical input.

Types are gamma = (-1.25, 0, 1.25). Priors are exactly (1/3,1/3,1/3)
and all permutations of (.50,.25,.25). Initial purchase state is observed;
the same prior is applied separately to each initial state. Other parameters
are beta=2, alpha=-1, delta=.92, cost=0. Prices are unrestricted.

The uncertain firm keeps the original type probabilities fixed after either
purchase outcome. It solves pooled terminal mixture demand separately in each
observed state, using the same original distribution. First-period continuation
values are fixed with respect to the initial offer. There is no Bayesian
learning. This fixed-belief approximation evaluates continuation demand using
the unconditional distribution rather than the selected type composition
of consumers reaching each state. Reported profits use that same subjective
fixed-belief objective. The known-type comparison uses personalized prices.

Outputs are in ../assets/results/gamma_uncertainty. Expected offered prices
are averaged over all relevant types and purchase histories, including offers
on paths where no inside purchase occurs. This is separate from the empirical
coupon exercise, which has not been rerun.

Validation checks wider first-price grids, independently maximized terminal
objectives, first-price numerical derivatives, belief invariance,
known-type profit bounds and certain-type limits. Numerical grid searches
are benchmark checks, not a global optimality proof for arbitrary calibrations.
