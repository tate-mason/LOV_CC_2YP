# Step 4: separating persistent flavor preferences from purchase-history effects

`est_step4.py` estimates conditional flavor demand for single-person households
buying single-serving yogurt. The default `flavor` column uses the draft's three
categories: 0 = other, 1 = berry, 2 = plain. An individual-flavor column can be
supplied explicitly. Its default model uses **household-income interactions for
both flavor preferences and fatigue, with no simulation draws or taste SDs**.
The legacy normal random-coefficient model remains available through
`--heterogeneity random`.

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

The provisional families are plain, berry, and other. The draft's 0/1/2 codes
map to other/berry/plain; other labels use a text heuristic. Arbitrary numeric
Nielsen flavor codes need an explicit taxonomy before a family exercise. Because
family stock includes own stock, own and family effects can be highly correlated.
With only the draft's three categories, family stock equals own stock, so the two
common history slopes cannot be separately identified. Do not enable
`--family-effect` for that three-category specification.

## 4. Utility, tastes, and normalizations

### Default: observed income heterogeneity

Let $h_i$ be standardized log household income. The model is

$$
\beta_{ij}=b_j+c_jh_i,\qquad \gamma_i=g_0+g_1h_i,
$$
$$
v_{itsj}=b_j+c_jh_i+\alpha p_{tsj}
-(g_0+g_1h_i)\log(1+S_{itj})+W_{itsj}'\kappa.
$$

Both reference-flavor preference coefficients, $b_0$ and $c_0$, are zero.
Gamma applies to all flavors. A common income main effect alone would cancel
from conditional flavor probabilities, so income interacts with category
indicators and exposure instead. For the three-category baseline there are
seven coefficients: two flavor intercepts, two income-preference slopes, gamma,
gamma-income, and price. There are no normal taste draws and no SD parameters.
This changes the economic restriction: observed income explains systematic
heterogeneity; remaining persistent household tastes are not estimated.

`household_income` is assumed to contain bracket codes. The script copies the
midpoint mapping from `est_step3.py` without importing that file or triggering
its data loading. The open top bracket is represented by $100,000. A code is not
treated as a dollar amount. For a genuine dollars column, use
`--income-encoding dollars --income-column COLUMN`.

Choose each household's earliest observed usable income and hold it fixed.
Calculate the mean and population SD of log income once across the retained
households with usable income, giving households equal weight. Thus $h=0$ means
average log income and $h=1$ means one log-income SD above that average, not one
SD of dollar income. Save the transformation and code diagnostics in the summary.
At least two distinct usable incomes are necessary to estimate these interactions.

Missing, invalid, and unmapped incomes do not remove households. They receive
$h=0$ and a missing-income indicator $m_i$. If any are present, estimate separate
$b_j^M m_i$ preference terms and a $g_M m_i$ fatigue term. This avoids representing
an unknown income as a genuinely observed average-income household. With three
categories, missing income adds three coefficients. These additional coefficients
also need variation to be identified; small missing-income groups may not provide it.

The relative log odds are linear in parameters:

$$
\log(P_j/P_0)=b_j+c_jh_i+\alpha\Delta p_j
-g_0\Delta z_j-g_1h_i\Delta z_j,
\quad z_j=\log(1+S_j),
$$

plus missing-income and other control terms when applicable. `income_design_rank.json`
reports column rank and singular values of the **relative** design. With three
categories and complete incomes, rank seven is required to distinguish all seven
coefficients. The implementation scales columns to unit norm for the numerical
rank check. A full sample rank supports the model's rank argument but does not
prove price/history exogeneity or validate price imputations. See
`est_step4_identification.md` for the formal argument.

The deterministic logit likelihood sums counted-choice log probabilities;
there is no integration over types. Analytic gradients and exact logit curvature
use the same design as utility. The objective is convex in these coefficients
for fixed histories/covariates, though a minimizer can lie on bounds or fail to
be finite under separation. Multiple starts remain a numerical check.

Income-mode estimates include household-cluster sandwich standard errors:

$$
H=\sum_{i,t,s}N_{its}\operatorname{Cov}_{P}(X_{itsj}),\qquad
s_i=\sum_{t,s,j}(y_{itsj}-N_{its}P_{itsj})X_{itsj},
$$
$$
\widehat V=\frac{G}{G-1}H^{-1}\left(\sum_i s_i s_i'\right)H^{-1}.
$$

Here $G$ counts households. Clustering permits within-household score dependence;
it does not correct omitted-variable bias or establish the assumed conditional
mean. Standard errors are omitted when optimization fails, curvature is singular
or ill-conditioned, or the coefficient is at a bound. Reported 95% intervals
use the normal approximation for interior coefficients. No Wald p-values are
automatically reported. The gamma profile fixes $g_0$ and re-estimates $g_1$;
fixing $g_0=0$ alone does not remove all fatigue. The no-fatigue comparison fixes
both gamma coefficients (and all optional history terms) to zero.

### Optional: normal random-coefficient heterogeneity

With `--heterogeneity random`, systematic utility is

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
`price`, and the selected flavor column (`flavor` by default). The default income
specification also requires the selected income column (`household_income`).
CSV and Parquet are supported. Parquet requires a pandas-compatible engine.
Prices must already be per-cup prices in dollars; the script cannot infer whether
a source column is a transaction total. Verify this before estimation.

Missing, malformed, infinite, zero, and negative prices are treated as missing
rather than stopping the run. Purchased quantities remain in exposure history.
Each store-week-flavor price averages its valid source prices, ignoring missing
ones. Missing cell prices are filled using the store-flavor average over dates,
then the overall flavor average, and finally the overall all-flavor average if
a flavor has no usable price anywhere. No purchase observation is discarded for
missing prices. If there are no usable prices anywhere, the script raises an
error rather than inventing a price level. Fallback counts are reported. Average
prices preserve the sample but do not create genuine local price variation;
the all-flavor fallback is particularly strong and should be visible in results.

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
flavor-level prices. The same mean-price fallbacks apply to unpriced listings.
An observed purchase missing from the supplied menu raises an error.

Without `--menu`, every retained category is assumed available at every observed
store-week, as in step 2. Pooled purchases supply prices with the above fallbacks.
Absence of sampled purchases does not establish that a category was unavailable.
This common-menu assumption is explicit; purchase-derived prices may still
depend on household demand. Output identifies this fallback.
Rows offering only one retained flavor contribute no likelihood information,
but are retained, and their purchased cups still update history.

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
a computational starting assumption. Households must have at least **two distinct
yogurt-purchase weeks**. Multiple stores or baskets in one week do not satisfy
this requirement; the weeks need not be consecutive. Every observed purchase
row of eligible households, including their first week, enters the likelihood.
There is no burn-in, two-alternative restriction, or complete-price restriction.
The draft reports about four yogurt
purchase weeks per household during eight weeks of observation; a three-week
burn-in would discard much of that useful history.

The input is already a yogurt-purchase sample. Gaps between observed purchases
create calendar decay; the script does not add nonpurchase observations or
outside-option counts. First-week exposure is zero and therefore does not
directly identify fatigue, but first-week choices contain information about
persistent preferences and relative prices. Keeping first weeks does not solve
unknown pre-sample consumption: a fuller model could use
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
2. Income mode holds observed household income fixed; random mode draws one
   continuous normal taste type per household (and gamma type when requested),
   independently of the estimation quadrature nodes.
3. Simulate flavor counts in chronological order, including first-week purchases.
4. Update stock from the **simulated** counts, regenerate history interactions,
   and continue. Never use the original history as a frozen covariate.
5. Re-estimate and record truth, recovered value, error, and convergence.

Recovery uses the completed price arrays, including mean-price fallbacks, and
simulates all observations recursively. This checks recovery under the assumed
prices; it does not validate the imputed prices against actual shelf prices.

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

# Income heterogeneity is the default; results go to step4_income_output:
python3 est_step4.py --profile -0.1 0 0.05 0.1 0.2 --recovery-reps 5

# Reproduce the previous normal random-coefficient specification:
python3 est_step4.py --heterogeneity random

# Inspect sample counts without fitting; writes fresh audit files:
python3 est_step4.py --audit-only --output step4_audit

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
python3 est_step4.py --heterogeneity random --input purchases.parquet --menu yogurt_menu.parquet \
  --gamma-mode random --draws 256 --output step4_random_gamma

python3 est_step4.py --heterogeneity random --input purchases.parquet --menu yogurt_menu.parquet \
  --gamma-mode flavor --output step4_flavor_gamma
```

Use distinct output folders for different specifications. Reusing an output
folder overwrites the current run's files and may leave optional outputs from
an earlier run; do not combine stale profiles/recovery with a new summary.
The default income output folder is `step4_income_output`; random mode uses
`step4_output`. The earlier random-coefficient results are therefore preserved
when running the new default income specification. `--draws` has no effect in
income mode, and `actual_draws` is recorded as zero.

Outputs are `estimates.csv`, `no_fatigue_estimates.csv`, `identification.csv`,
`run_summary.json`, and, when requested, `gamma_profile.csv` and `recovery.csv`.
`household_attrition.csv`, `source_sample_stages.csv`, and
`choice_sample_stages.csv` show input/product-filter counts and verify that
the two-purchase-week restriction and retention of all eligible households' rows.
Audit-only mode
saves these, `identification.csv`, and `sample_audit.json` without estimation.
Income mode also saves `household_income.csv` and `income_design_rank.json`.

The summary records the sample, menus, price assumptions, optimizer status,
draw count, initial-stock treatment, and lack of price correction. SDs and bound
flags in `estimates.csv` use the actual fitted parameterization.

The self-test checks multi-store stock aggregation, calendar decay, explicit
household integration, the zero-heterogeneity likelihood limit, unavailable
alternatives, the income log-odds identity, analytic gradients, rank, clustered
inference, no-fatigue restrictions, recursively generated histories, and recovery in a
small synthetic panel. It does not replace validation on Nielsen data.

## 12. Suggested empirical sequence

Start with the common-gamma specification and a credible supplied menu. Verify
price units and exact-flavor labels. Inspect within-household relative exposure,
the no-fatigue comparison, convergence, start agreement, and gamma profiles.
Run recovery before interpreting weak or extreme estimates.

Then compare retention rates, initial-stock assumptions, flavor-set size, brand/style
controls, and dose/timing interactions. A stable result across these comparisons
is more informative than a single large coefficient. Examine family fatigue,
flavor-specific gamma, and random gamma only once the simple history effect is
credible. Short panels may support the mean effect but not a distribution of
fatigue types.

For a change in own exposure from $S_a$ to $S_b$, holding other attributes fixed,
the default utility change for a household with income $h_i$ is

$$
\Delta v=-(g_0+g_1h_i)\,[\log(1+S_b)-\log(1+S_a)].
$$

If alpha is strictly negative, the corresponding per-cup compensating discount
for a positive fatigue loss is

$$
d=\frac{(g_0+g_1h_i)[\log(1+S_b)-\log(1+S_a)]}{-\alpha}.
$$

Gamma divided by alpha alone is not an economically specified willingness to
pay: the exposure change must be stated. This discount interpretation is
conditional on the model and its price assumptions, and applies to an individual
type if gamma varies across households.
