# Standalone empirical berry targeting

This exercise uses the **base supply logic**: the seller targets a particular
household's estimated type and observed consumption history, using backward
induction. It does not impose the new theoretical uniform-pricing or commitment
restrictions. High and low gamma are comparison groups, not groups required to
receive identical prices or coupons.

`run_empirical_targeting.py` is separate from estimation. It does not import the
estimator, change its files, estimate coefficients, or bootstrap a fit. It reads
an existing fitted-results directory and an explicitly supplied saved estimation
table only when the person running it requests an empirical run. No empirical
input has been accessed to develop or validate this version.

## Syntax

From the repository root, run yourself:

```bash
python3 pricing_exercises/run_empirical_targeting.py \
  --household-input "/YOUR/EXPLICIT/HOUSEHOLD_INPUT.csv" \
  --fit-dir "/YOUR/EXPLICIT/FITTED_RESULTS_DIRECTORY" \
  --output pricing_exercises/empirical_outputs
```

The existing `res_hess` fit is supported. The fit directory must contain
`fit.json` and `point_estimates.csv`; supply its path explicitly. No household
input path is defaulted or discovered automatically.

Optional additions:

By default, the firm optimizes over every 1% discount from 0% through 100%
(101 candidate offers). Omit `--discounts` to use this grid; supplying it replaces
the candidate grid.

Plots are saved as both PNG and PDF:
`household_coupon_results` shows estimated consumer types and dynamic berry
total expected revenue (`p*q`) by type, with one thin line per type connecting dots for the
first and terminal pricing periods. Revenue is summed across eligible households within each type group;
terminal revenue averages `p2(h1)*q2` over histories, and neither period is
time discounted in this graph. These are revenue levels, not percentage gains;
`discounts_by_consumer_type` shows mean optimal dynamic discount percentages by
low/middle/high gamma group at `p1(h0)` and terminal `p2(h1)`;
`discounts_by_household_type` shows dynamic household discounts against estimated
gamma and price sensitivity. Period 2 discounts are expectations over future
consumption histories. Group means weight berry-eligible households equally;
the middle group receives zero discounts by policy design.

```text
--demand-delta 1
--batch-size 24
--save-state-policies
```

`--save-state-policies` writes every chosen household/basket terminal offer to
a compressed CSV. This can be large. Output files of the same names in the
chosen output directory are overwritten; use separate directories for different
specifications. Dependencies: NumPy, pandas, matplotlib.

## 1. Recover types and define comparison groups

The supported model is the converged original single-normal heterogeneous-gamma
model. It has persistent household gamma and lognormal positive price sensitivity
a, with current-period inside utility

\[
 u_{ijt}=b_0+\beta_j-a_i p_{ijt}+\gamma_i\Xi_{ijt}+\rho\widehat v_{ijt}.
\]

Here beta_other=0. All three flavors and the outside option remain in demand.
The seller receives berry revenue only. The existing price control remains
fixed when evaluating a coupon intervention.

The script recreates antithetic normal draws with the fit's draw count/seed.
For each actual household it computes its complete observed history's likelihood
at each persistent draw, then normalizes:

\[
 \ell_{id}=\sum_{t,k}n_{ikt}\log P_{ikt}(a_d,g_d),\qquad
 w_{id}=\frac{e^{\ell_{id}}}{\sum_m e^{\ell_{im}}}.
\]

Posterior means are `a_hat_i=sum_d w_id a_d` and
`gamma_hat_i=sum_d w_id g_d`. Households are ranked by gamma_hat: the lowest
and highest `floor(N/4)` form low/high gamma groups. The remaining households
are middle. Ties are resolved by lexicographically sorted household ID.
Groups are determined before checking forecast berry availability.

The default `--type-treatment posterior-mean` treats each household's estimated
pair `(a_hat_i,gamma_hat_i)` as known and fixed, matching the base supply model's
information assumption. This is a plug-in type approximation: the type is
estimated, rather than observed without error. The full posterior is used to
estimate type and rank actual households, but is not integrated again in default
pricing. Posterior SD and effective sample size are exported as diagnostics.

An optional sensitivity is

```text
--type-treatment posterior-integrated
```

This integrates persistent types in both periods and conditions their posterior
on the first basket. The terminal action is chosen after integrating over the
remaining type uncertainty, not separately for each unobserved true draw. It
changes the seller's type-information assumption and should be reported
separately from the default base-model exercise.

## 2. Forecast timing and demand history

There are three choice dates: an observed initial choice establishes `h0`,
the seller offers `p1(h0)` and the resulting basket establishes `h1`, then
the seller offers terminal `p2(h1)`. Only the latter two dates are priced
counterfactuals and enter the revenue objective. Thus references to two pricing
periods count the decisions after initialization, not the initial observed
choice. This is the timing of the known-state supply theory.

The first chronologically observed basket for each household is relabeled
from observed `t=1` to model `t=0` and establishes `h0`.
The counterfactual begins in the following calendar week, with `p1(h0)`;
`p2(h1)` is terminal in the next consecutive week. Initial stock equals the
first observation's pre-purchase stock (zero at sample entry under the estimator)
plus that basket's inside flavor shares. Initial novelty uses the same basket.
Later observed baskets do not enter this initial demand state. Household type
estimates still use the complete estimation sample: this is a retrospective
known-estimated-type counterfactual, not an out-of-sample forecast. Households'
initial dates differ; these are household scenarios, not one simultaneous
retail-market equilibrium.

Carry the first observed posted prices, available flavor menu and price-control
residuals forward for both weeks. Competitor prices do not respond. Missing
prices are not imputed. A household is eligible only if berry is covered in its
first menu. Quartiles are not recomputed on that eligible subset, and group
eligibility is reported. Aggregate revenue refers to eligible households only.

For `K=capacity`, forecast weekly baskets are multinomial across other, berry,
plain and outside, conditional on type. All `choose(K+3,3)` baskets are integrated
exactly: for K=7 there are 120. If the first basket's inside counts are n, set
`x=n/sum(n)` when inside purchases occur and x=0 for the all-outside basket.
Update the full demand state:

\[
 S_2=S_1+x,\qquad N_{j2}=\sum_{k\ne j}x_k,\qquad
 \Xi_{j2}=(1-\lambda)N_{j2}-\lambda S_{j2}.
\]

The all-outside basket leaves stock unchanged and sets next novelty to zero.
History is updated for every simulated basket, not at expected purchase shares.
The seller observes the household's complete first-week flavor basket and
can condition its terminal coupon on that realized state. A seller who observes
only its own berry sales would need a belief over unobserved competitor baskets;
that is a different information restriction.

## 3. Household-specific pricing by backward induction

An action is a per-unit berry coupon fraction d from the supplied grid. Its net
price is `p_i(d)=(1-d)*posted_berry_price_i`. Default fractions are
0,.01,.02,...,1. Coupons apply to all berry units bought in the week and are
automatically redeemed; there is no fixed coupon delivery cost, transfer or
redemption friction. Net revenue already subtracts the discount.

This is the coupon-constrained version of base optimal pricing: the firm can
reduce prices relative to each household's posted price but cannot raise them
above that price or choose introductory negative prices. Therefore it need not
equal unrestricted theoretical supply prices. A chosen endpoint calls for grid
refinement/extension before interpreting it as an unrestricted optimum.

For the default known estimated type, define terminal revenue in full basket
state n:

\[
 W_{i2}(n)=\max_{d_2}\ p_i(d_2)K P_{i,berry,2}(n,d_2;\widehat a_i,\widehat\gamma_i).
\]

Choose the first offer using

\[
 \max_{d_1}\ p_i(d_1)K P_{i,berry,1}(d_1)
 +\delta_D\sum_n M_i(n\mid d_1) W_{i2}(n).
\]

This is backward induction for **each household separately**, not a revenue
maximization over pooled gamma groups. Both initial history and type affect the
first offer; the realized first basket affects the second. Low/high gamma
households can receive many different prices, including different offers within
the same quartile. The gamma label never enters the pricing objective except
to designate which households are eligible for targeting; swapping low and high
labels leaves every household's action unchanged.

For a reporting type group G, the revenue graph plots

\[
 R_{G1}=\sum_{i\in G}p_{i1}(h_{i0})\,
     \mathbb E[q_{i1}\mid h_{i0},p_{i1},\widehat a_i,\widehat\gamma_i],
\]
\[
 R_{G2}=\sum_{i\in G}\sum_{h_1}
     \Pr(h_1\mid h_{i0},p_{i1},\widehat a_i,\widehat\gamma_i)\,
     p_{i2}(h_1)\,
     \mathbb E[q_{i2}\mid h_1,p_{i2}(h_1),\widehat a_i,\widehat\gamma_i].
\]

The terminal price is optimized conditional on each realized history and known
estimated type before revenue is averaged over histories. The graph does not
multiply an average terminal price by an average quantity. Initial observed
choices can be berry, plain, other, or a mixed basket; every subsequent basket,
including all-outside, contributes with its probability under the chosen first
price. For capacity one these are individual choice transitions such as
berry-to-berry and berry-to-plain; for capacity seven the solver integrates all
120 possible first-period baskets and the terminal demand probabilities. Only
berry sales generate seller revenue.

Dynamic pricing is the only optimized specification. The script retains a
no-coupon benchmark for measuring revenue gains:

| Policy | Interpretation |
|---|---|
| `no_coupon` | Observed posted berry prices in both future periods |
| `personalized_dynamic` | Each targeted household's first offer accounts for optimized terminal continuation revenue; its terminal offer is state contingent |

Only low/high gamma households receive personalized offers. The middle group
remains at posted prices under every policy, providing an untreated bookkeeping
group. There is no common/group coupon policy and no empirical commitment
exercise. Dynamic revenue must weakly exceed no-coupon revenue for each
household. Graphs show only dynamic results. Exported revenue gains are measured
relative to the no-coupon benchmark.

Consumers use the fitted current-period demand rule; they do not strategically
delay purchases in anticipation of future discounts. `--demand-delta` controls
revenue discounting (default 1), independently of the stock history specification
and theoretical supply calibration.

## 4. Inputs and guards

The explicit household input is the saved weekly estimation table, with
`household_code`, `week`, counts `other/berry/plain/outside`, and columns
`price_*`, `stock_*`, `novelty_*`, `xi_*` for all three flavors. Controlled fits
also require `cf_*`. One row per household/week is required.

The script checks convergence, boundary flags, capacity, household/week totals,
covered prices for purchased alternatives, finite counts/history/controls, and
the stored Xi identity. It recomputes the household-integrated negative log
likelihood and requires agreement with the saved fit before forecasting. This
guards against mismatched fits and saved samples. It does not rerun estimation
to repair a mismatch.

## 5. Outputs and interpretation

| File | Contents |
|---|---|
| `household_posterior_summary.csv` | Household type estimates, posterior uncertainty, quartile, first-menu eligibility, initial observed week and two pricing weeks |
| `household_group_summary.csv` | Counts, eligible counts, gamma ranges, alpha and posterior uncertainty by reporting group |
| `targeted_household_results.csv` | Each eligible household's prices, first coupon, expected second coupon, price ratios, period revenue/units and revenue gains for each policy |
| `targeted_revenue_by_gamma_group.csv` | Revenues and units summed by policy and gamma group; mean prices/discounts/ratios |
| `targeted_revenue_aggregate.csv` | Total revenue, units, coupon expenditure and gains across eligible households |
| `targeted_state_policies.csv.gz` | Optional terminal coupon, prices, ratios and probabilities for every targeted household/first-basket state under the optimized policies |
| `household_coupon_results.png` | Posterior mean gamma distribution and dynamic berry revenue (`p*q`) by consumer type and pricing period |
| `run_manifest.json` | Settings, input/fit/parameter/script hashes, type treatment, likelihood and probability diagnostics |

The main economic comparison is revenue gain for low versus high gamma under
household-specific dynamic targeting. This does not isolate gamma causally from
alpha, menu and initial-history differences across actual households. Those
other characteristics also influence optimal offers.

Revenue gains use the **same model's no-coupon forecast**, not historical
receipts. All revenue is net of coupons. With zero marginal cost it also equals
profit; no positive-cost profit claim is made. Coupon expenditure is the rebate
on all berry units bought. Adding it to net revenue recovers posted-price
receipts on those policy purchases, including inframarginal units.

Household `expected_p2` averages the offered terminal price over its induced
basket distribution. `ratio_of_expected_prices` is p1/E[p2];
`expected_price_ratio` is E[p1/p2]. Conditional p2 means for berry versus no berry
are also reported, although the seller conditions on the full basket. These
are offered-price objects rather than prices conditional on purchasing twice.
Zero terminal prices make some ratios undefined; the script exports missing
values in that case.

Group price/ratio columns are arithmetic means across eligible households;
they are not sales-weighted transaction prices or ratios of group mean prices.
Group/aggregate percentage revenue gains instead divide **summed revenue gains**
by summed baseline revenue. They are not averages of household percent gains.

## 6. Validation and extensions

Run synthetic checks without supplying any empirical paths:

```bash
python3 pricing_exercises/run_empirical_targeting.py --self-test
```

To check the first-observation initialization and an independent scalar
backward-induction benchmark without reading or writing any CSV files:

```bash
python3 -m unittest pricing_exercises.test_empirical_timing -v
```

Checks cover posterior weights against a scalar likelihood, basket probability
mass including missing competitors and no-purchase baskets, revenue accounting,
household policy dominance, invariance to low/high reporting-label swaps, and
the no-history/no-type-learning limiting case. An independent scalar exhaustive
two-period calculation checks one household's dynamic optimum. Temporary
synthetic fixtures exercise file loading, likelihood guards, compressed state
exports and plots. No restricted empirical input is used, and no empirical
revenue result has been calculated here.

For a longer horizon, recurse over stock S, latest flavor shares x, and the
forecast environment. Under known estimated type, type remains fixed; under
posterior integration, also update the posterior weights after each basket.
Optimize a household-specific action at each observed state. Exact enumeration
grows like 120^(T-1) before coalescing states at K=7, so practical extensions
need memoization, state interpolation, approximate dynamic programming, or
simulation with independent policy-evaluation paths. The present exact
two-period solver provides a benchmark for those approximations.

Important empirical sensitivities are posterior draw count/seed (with a matching
fit), type treatment, coupon grid, initial-state/menu selection, redemption
frictions, revenue discounting and capacity. Constructed outside counts and
purchase-week selection do not by themselves establish a model for weekly
category participation: forecasting exactly K opportunities, including zero
inside purchases, is a maintained extrapolation. Observed weeks with more than
K packages can have more than K likelihood counts. Treat forecasts accordingly.

There are no sampling confidence intervals in this exercise. A household
bootstrap would need to refit demand/first stages, recover types/ranks, and
reoptimize policies in every replication. A prospective holdout or randomized
coupon experiment can evaluate targeting out of sample. These extensions do
not require merging targeting into the estimation script.
