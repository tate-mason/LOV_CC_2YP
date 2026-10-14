# Step 4: separating persistent flavor preferences from purchase-history effects

`est_step4.py` estimates conditional flavor demand for single-person households
buying single-serving yogurt. Its default model has persistent heterogeneous
flavor tastes and **one common own-flavor fatigue coefficient**. It is deliberately
smaller than the heterogeneous beta-and-gamma specification in `est_step2.py`.

The script also supports exact-flavor-specific fatigue, heterogeneous common
fatigue, flavor-family exposure, menu brand/style controls, dose/timing
interactions, likelihood profiling, and recursive simulation-based recovery.
These extensions are sensitivity exercises; the default does not activate them.

The default purchase input is
`/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet`.
Input files may be outside this folder; relative inputs resolve against the
script's folder. Outputs must remain inside the script's folder.
The script does not import earlier estimation scripts or create their
`../Output` directories. Importing this script does not
load data, fit models, or create files.

## 1. The economic question

Suppose a household usually prefers strawberry to plain. That is a persistent
taste difference, represented by beta. Suppose it becomes less likely to choose
strawberry after buying several strawberry cups and more likely again after a
long gap. That is an exposure effect, represented by gamma.

Persistent taste and fatigue can coexist: someone can have a high baseline
preference for strawberry and still want a break after repeated purchases.
Looking only at average shares cannot distinguish these mechanisms. Histories
and changes within households provide the useful information.

Single-serving cups provide a natural dose measure, but buying a cup does not
tell us when it was eaten. Throughout this script, “exposure” means a decayed
**purchase proxy**, not observed consumption. Single-person households reduce
confounding from multiple eaters but do not eliminate gifts, guests, stockpiling,
or differences between buying and eating.

## 2. The observation and the conditional choice problem

Let $i$ denote household, $t$ calendar purchase week, $s$ store, and $j$ retained
exact flavor. The observed count is $y_{itsj}$, and

$$
N_{its}=\sum_{j\in A_{ts}} y_{itsj}
$$

is the number of retained-flavor cups bought at that store in that week.
$A_{ts}$ is the observed menu. The likelihood conditions on $N_{its}$, purchase
dates, and visited stores. It explains the allocation across flavors, not the
decision to buy yogurt or visit the store.

There is **no outside option and no weekly capacity of 14**. A common inside
intercept would cancel from all flavor probabilities, so none is estimated.
This specification cannot estimate the yogurt participation margin, inventory
policy, overall category demand, or a welfare measure involving other foods.

The maintained count likelihood treats cups as conditionally independent flavor
allocations within a basket, given household type and pre-week history. This
does not model deliberate variety within a basket or correlated basket shocks.
All purchases during one week use the same pre-week stock. Weekly data cannot
reveal which of two purchases happened first.

## 3. A stable exposure stock

Write $Q_{itj}=\sum_s y_{itsj}$. Before the first observed purchase week, set
$S_{itj}=0$. For successive observed purchase weeks $a$ and $b$,

$$
S_{ibj}=\delta^{\Delta_{ab}}\left(S_{iaj}+Q_{iaj}\right),
\qquad
\Delta_{ab}=\frac{b-a}{7\text{ days}},\qquad 0<\delta<1.
$$

New purchases are added at the end of week $a$ and decay through the gap to
week $b$. This timing convention slightly differs from adding an undecayed
previous purchase directly to the next pre-choice stock. It is used consistently
in estimation and simulation.

The stock uses **cup quantities**, not shares. Buying six strawberry cups creates
more exposure than buying one. Old quantities decay once; previous stock states
are not repeatedly added into the memory sum. A three-week gap is three weeks
of decay even if no yogurt purchase was recorded in between.

In plain language: carry forward the remaining exposure, let it fade with elapsed
time, and add purchases once. The default $\delta=0.8$ is an illustrative weekly
retention rate, not an estimated or literature-validated value. Compare several
rates, such as 0.5, 0.8, and 0.95, before drawing substantive conclusions.

Flavor-family stock is an optional additional measure:

$$
F_{itj}=\sum_{k:\,g(k)=g(j)} S_{itk}.
$$

The provisional families are plain, berry, and other. The mapping is a text
heuristic; numeric flavor codes will not classify berry/plain meaningfully.
Use meaningful labels and review this mapping before a family exercise. Because
family stock includes own stock, own and family effects can be highly correlated.

## 4. Utility, tastes, and normalizations

The default systematic utility is

$$
v_{itsj}=\beta_{ij}+\alpha p_{tsj}
-\gamma\log(1+S_{itj})+W_{itsj}'\kappa,
$$

with independent type-I extreme-value choice shocks. Therefore

$$
P_{itsj}(\theta_i)=
\frac{\exp(v_{itsj})}{\sum_{k\in A_{ts}}\exp(v_{itsk})}.
$$

Unavailable alternatives have probability zero. The most purchased retained
flavor is the reference, indexed by zero. The persistent taste specification is

$$
\beta_{i0}=0,\qquad
\beta_{ij}=\mu_j+\tau_j\nu_{ij},\quad
\nu_{ij}\sim N(0,1),\quad j>0.
$$

This parameterizes taste differences relative to the reference. Independent
normal differences are a covariance restriction; they are not a fully flexible
joint taste distribution. The reference normalization removes an arbitrary
utility location. **Gamma still applies to the reference flavor**. A reference
flavor's history slope cannot be normalized away just like its intercept.

Positive gamma means fatigue because utility subtracts gamma times exposure.
Negative gamma means reinforcement. The mean gamma is allowed to be signed so
the estimator can reveal either pattern. Price alpha is constrained nonpositive.
Taste standard deviations are estimated and reported in their natural units,
and zero is allowed. There is no extra exponentiation when printing results.

The implemented bounds are numerical guardrails: taste means in [-12,12], SDs
in [0,4], history means in [-8,8], price in [-10,0], and controls in [-8,8]. An
estimate at a bound needs investigation; increasing the bound is a sensitivity
check, not evidence of identification.

Optional history specifications replace the common gamma with $\gamma_j$ for
each flavor, or with $\gamma_i=\bar\gamma+\omega\xi_i$, where $\xi_i$ is a
standard normal independent of the taste draws. The latter allows some household
types to experience reinforcement even if mean fatigue is positive. Neither
extension establishes that the data identify the extra parameters.

With `--family-effect`, utility additionally subtracts
$\gamma_F\log(1+F_{itj})$. With `--timing-controls`, $W$ includes

$$
\log(1+S_{itj})\log(1+Q_{i,a,\mathrm{total}}),\qquad
\log(1+S_{itj})\log(1+\Delta_{at}).
$$

These are dose and timing interactions. A common lag-total or gap term alone
would cancel across flavors, so adding it as a scalar utility control would do
nothing. The interactions permit a different exposure relationship after large
purchases or long gaps, but also compete with gamma for identification. They
are sensitivity controls, not a structural inventory model.

## 5. The household panel likelihood

For simulation draw $r$, form the likelihood for the **whole household**:

$$
\ell_{ir}=\sum_{t,s:\,\mathrm{keep}_{its}=1}\sum_j
y_{itsj}\log P_{itsj}(\theta_{ir}).
$$

Then integrate over types:

$$
\ell_i=\log\left[\frac{1}{R}\sum_{r=1}^R\exp(\ell_{ir})\right],
\qquad
\widehat\theta=\arg\min_\theta -\sum_i\ell_i.
$$

The multinomial coefficient is omitted because it is constant in parameters.
The script uses log-sum-exp to avoid numerical underflow. It handles unavailable
alternatives without multiplying zero counts by negative infinity.

The same draw $r$ supplies one household type for all its weeks and stores.
Integrating each cup or week separately would instead erase the persistent-type
information that helps distinguish tastes from history effects. Scrambled Sobol
normal draws remain fixed during optimization; requested draw counts are rounded
up to a power of two. The same numerical nodes can be reused across households
because each household's integral is still calculated separately.

L-BFGS-B fits several starts and reports each objective. These starts help detect
local optima; they do not guarantee a global solution. Increase draw counts and
compare seeds after the baseline behaves sensibly. The script does not report
numerical inverse-Hessian standard errors or Wald p-values.

## 6. Building choices, menus, and controls

Purchase inputs require `household_code`, `store_code_uc`, `week_end`, `quantity`,
`price`, and the selected exact-flavor column (`flavor_descr` by default).
CSV and Parquet are supported. Parquet requires a pandas-compatible engine.
Prices must already be per-cup prices in dollars; the script cannot infer whether
a source column is a transaction total. Verify this before estimation.

Missing, malformed, infinite, zero, and negative prices are treated as missing
rather than stopping the run. Purchased quantities remain in exposure history.
Each store-week-flavor price averages its valid source prices, ignoring missing
ones. If any listed retained flavor still has no valid price, the entire
household-store-week is excluded from the likelihood. Its purchases still update
stocks. This preserves the menu rather than declaring an unpriced flavor
unavailable, and avoids imputing prices from unrelated stores or dates. Zero
placeholders in internal arrays belong only to excluded observations. The run
summary reports invalid source rows and exclusions; a sample with no informative
priced panels raises an explanatory error. Missing-price selection may still
affect estimates, so inspect these counts.

If present, the script filters `household_size == 1`, serving codes 67181961 or
65622705, and yogurt module codes 3612 or 3603, matching the earlier scripts.
If those columns are absent, `--prefiltered` explicitly declares that the input
has already been restricted. This declaration does not verify product packaging;
inspect the serving-code definitions in your data.

The top exact flavors are selected by purchased cups. Purchases of the remaining
flavors are excluded, with their cup count reported. There is no heterogeneous
OTHER bucket. Consequently, demand is conditional on purchases from the retained
set, and stocks cover retained flavors only. Family exposure omits excluded
flavors. Change `--top-flavors` to check sensitivity to that restriction.

Purchases are summed by household, store, week, and flavor. A household visiting
two stores gets two store-specific count vectors. Both use the same pre-week
history; updating history sums those vectors once. Household totals are never
copied into each store's choice observation.

Prefer an independently observed `--menu` containing store, week, exact flavor,
per-cup price, and requested attributes. Supply rows representing product listings
from the appropriate yogurt/single-serving universe. Mean listing prices create
flavor-level prices. The script does not manufacture prices for absent flavors.
An observed purchase missing from the supplied menu raises an error.

Without `--menu`, the pooled purchases at a store-week provide an availability
proxy and mean price. This is convenient for exploration but selected purchases
do not reveal the full shelf menu. Availability and category prices may then
depend on household demand. Output explicitly identifies this fallback.
Rows offering only one retained flavor contribute no likelihood information,
but their purchased cups still update history.

`--attribute-columns brand_cd yogurt_style` adds categorical menu-composition
controls: within each store-week-flavor, the fraction of listings at each brand
or style, omitting one level per attribute. In the purchase-derived fallback,
these are fractions of purchase records instead. They are not chosen-brand
dummies or a joint brand-flavor choice model. They can control changing menu
composition but cannot fully separate brand loyalty from flavor loyalty.
Attributes that are invariant within flavor can be collinear with flavor means.
The script requires explicitly named columns rather than guessing a style from
product descriptions.

## 7. Initial conditions and the limits of causal interpretation

The first observed purchase is not a first-ever purchase. Zero initial stock is
a computational starting assumption. By default, the first three calendar weeks
after the household's first retained-flavor purchase construct history but do
not enter the likelihood. At least three distinct informative post-burn-in
purchase weeks are required. Dates without observed purchases create decay,
not synthetic choice observations.

Burn-in reduces dependence on the unknown initial stock; it does not solve the
initial-conditions problem. The remaining unknown initial contribution is scaled
by $\delta^{\text{elapsed weeks}}$. Try longer burn-in and different delta values.
Households with short panels may then disappear. Compare sample sizes as well
as parameters. Treating the burn-in history as a conditioning variable while
integrating over an unconditional taste distribution also leaves potential
dependence between initial histories and tastes. A fuller model could use
pre-sample histories or a conditional distribution of tastes given initial
history. This version does not claim to implement that correction.

History arises from earlier choices. Persistent taste heterogeneity accounts
for one source of correlation, but serially correlated unobserved preferences,
promotions, learning, health changes, inventory, and selected purchase dates can
still masquerade as fatigue. Price endogeneity is also unresolved: this script
does not automatically import the earlier other-market-price control function.
A credible IV/control-function extension requires checking the instrument and
first stage on the actual menu data. Conditional estimates are not automatically
causal effects.

## 8. Identification diagnostics

`identification.csv` reports exposure variation for each flavor. Define
$z_{itj}=\log(1+S_{itj})$. On informative rows offering flavor $j$, the script
reports overall variation and within-household deviations

$$
z_{itj}-\bar z_{ij}.
$$

It also uses rows offering both $j$ and the reference to report within-household
variation in

$$
(z_{itj}-z_{it0})-\overline{(z_{ij}-z_{i0})}.
$$

The second diagnostic matters because only relative utility affects conditional
flavor probabilities. A common movement in every flavor's exposure can cancel
even when each flavor's raw stock varies substantially. The script also counts
households buying both this flavor and another retained flavor.

These are descriptive diagnostics. Centering does not create new variation or
establish exogeneity. Centered regressors are not substituted into the fitted
model: doing that changes how persistent tastes relate to history, especially
when means use future purchases. The reference's relative-stock diagnostic is
zero by construction.

## 9. Gamma profiles and the no-fatigue comparison

With `--profile`, hold mean gamma at each supplied grid value and re-estimate
all nuisance parameters using the same draws and data:

$$
\ell^*(g)=\max_{\theta:\,\gamma=g}\ell(\theta).
$$

The output contains fixed gamma, maximized negative log likelihood, nuisance
parameters, convergence, and twice the likelihood loss relative to the best fit
found across unrestricted and profile fits. A flat profile and compensating beta
changes indicate weak separation. A profile fit improving on the unrestricted
fit flags an optimization problem. In random-gamma mode the profile fixes the
mean, while SD is re-estimated; it is not a profile of a single household slope.
Flavor-specific gamma needs a multidimensional profile and is not supported by
the one-dimensional CLI.

Every run also fits a no-fatigue comparison on the same sample and menus,
fixing all gamma slopes/SDs and optional history-interaction coefficients to zero.
Brand/style controls and persistent taste heterogeneity remain. The saved
history coefficients in that comparison are restrictions rather than estimates.
The script does not turn likelihood
differences into automatic significance claims, especially with variance
parameters on boundaries or weak identification.

## 10. Recursive parameter recovery

`--recovery-reps B` treats the fitted parameter vector as a known truth and
repeats the following experiment:

1. Keep observed household panel lengths, purchase dates, stores, prices, menus,
   and per-store basket totals.
2. Draw one continuous normal taste type per household (and gamma type when
   requested), independently of the estimation quadrature nodes.
3. Simulate flavor counts in chronological order, including burn-in purchases.
4. Update stock from the **simulated** counts, regenerate history interactions,
   and continue. Never use the original history as a frozen covariate.
5. Re-estimate and record truth, recovered value, error, and convergence.

For observations whose menus lack valid prices, recovery holds the observed
basket fixed and carries its quantities into the simulated stock. It cannot
simulate a price-dependent choice distribution at an unknown price. All priced
observations, including priced burn-in observations, are simulated recursively.

Large errors or inconsistent solutions under the model's own data-generating
process suggest that the design or estimator cannot reliably separate the
parameters. Successful recovery is encouraging but does not validate consumption
measurement, menu measurement, price exogeneity, initial conditions, or purchase
timing. Because recovery uses fitted truth, test alternative plausible parameter
vectors through the `simulate()` Python function too. Recovery error is not a
household-bootstrap standard error for the empirical estimates.

## 11. Running the script

Running without `--input` uses the default scanner-panel path above. Override it
with `--input` when needed. Replace example menu/input filenames with your files.

```sh
python3 est_step4.py --self-test

# Use the default scanner-panel input:
python3 est_step4.py

python3 est_step4.py --input purchases.parquet --menu yogurt_menu.parquet \
  --attribute-columns brand_cd yogurt_style \
  --profile -0.5 0 0.25 0.5 1 2 --recovery-reps 5

# A restricted input without the original filter columns:
python3 est_step4.py --input restricted_yogurt.csv --prefiltered \
  --flavor-column flavor --output step4_exploratory

# Examine sensitivity to inventory proxies and broader flavor fatigue:
python3 est_step4.py --input purchases.parquet --menu yogurt_menu.parquet \
  --timing-controls --family-effect --output step4_family_timing

# Richer heterogeneity only after the baseline behaves well:
python3 est_step4.py --input purchases.parquet --menu yogurt_menu.parquet \
  --gamma-mode random --draws 256 --output step4_random_gamma

python3 est_step4.py --input purchases.parquet --menu yogurt_menu.parquet \
  --gamma-mode flavor --output step4_flavor_gamma
```

Use distinct output folders for different specifications. Reusing an output
folder overwrites the current run's files and may leave optional outputs from
an earlier run; do not combine stale profiles/recovery with a new summary.

Outputs are `estimates.csv`, `no_fatigue_estimates.csv`, `identification.csv`,
`run_summary.json`, and, when requested, `gamma_profile.csv` and `recovery.csv`.
The summary records the sample, menus, price assumptions, optimizer status,
draw count, initial-stock treatment, and lack of price correction. SDs and bound
flags in `estimates.csv` use the actual fitted parameterization.

The self-test checks multi-store stock aggregation, calendar decay, explicit
household integration, the zero-heterogeneity likelihood limit, unavailable
alternatives, recursively generated histories, and parameter recovery in a
small synthetic panel. It does not replace validation on Nielsen data.

## 12. Suggested empirical sequence

Start with the common-gamma specification and a credible supplied menu. Verify
price units and exact-flavor labels. Inspect within-household relative exposure,
the no-fatigue comparison, convergence, start agreement, and gamma profiles.
Run recovery before interpreting weak or extreme estimates.

Then compare retention rates, longer burn-in, flavor-set size, brand/style
controls, and dose/timing interactions. A stable result across these comparisons
is more informative than a single large coefficient. Examine family fatigue,
flavor-specific gamma, and random gamma only once the simple history effect is
credible. Short panels may support the mean effect but not a distribution of
fatigue types.

For a change in own exposure from $S_a$ to $S_b$, holding other attributes fixed,
the default utility change is

$$
\Delta v=-\gamma\,[\log(1+S_b)-\log(1+S_a)].
$$

If alpha is strictly negative, the corresponding per-cup compensating discount
for a positive fatigue loss is

$$
d=\frac{\gamma[\log(1+S_b)-\log(1+S_a)]}{-\alpha}.
$$

Gamma divided by alpha alone is not an economically specified willingness to
pay: the exposure change must be stated. This discount interpretation is
conditional on the model and its price assumptions, and applies to an individual
type if gamma varies across households.
