# Identification with the draft's three yogurt categories and short panels

This note uses other, berry, and plain, as in
`updates/draft_2/draft_1010.tex`. It does not require splitting categories or
discarding short histories. The draft reports 998 yogurt-purchasing households,
about four yogurt purchase weeks per household, and eight weeks of observation.
These are different lengths: four purchase occasions do not mean four consecutive
calendar weeks. Exposure should decay using calendar gaps.

An identification proof concerns whether different parameter vectors generate
different population distributions of observed choices. A large household count,
optimizer convergence, or a significant coefficient is not such a proof. A short
panel may identify a parsimonious parametric model in a population with many
households, but does not automatically identify a flexible distribution of tastes.

## 1. Establish the object being identified

The input is already restricted to yogurt purchases. Step 4 explains flavor
allocation conditional on yogurt quantities, purchase dates, and stores. It does
not observe a per-serving yogurt-versus-other-food choice.

For household $i$, store $s$, week $t$, and category $j$, write

$$
v_{itsj}=\beta_{ij}+\alpha p_{tsj}-\gamma_i z_{itj},
\qquad z_{itj}=\log(1+S_{itj}).
$$

Take other as the reference category, $j=0$, and normalize $\beta_{i0}=0$.
Normalize the extreme-value error scale to one. Every category still has a
history term: setting the reference intercept to zero does not justify setting
its history effect to zero.

The working exposure process is

$$
S_{ibj}=\delta^{\Delta_{ab}}(S_{iaj}+Q_{iaj}),
$$

where $Q_{iaj}$ sums cups across stores in week $a$, and $\Delta_{ab}$ is the
calendar gap to week $b$. Delta is fixed. With no burn-in, initial stock is set to
zero at the first observed purchase. This is an identifying assumption about
pre-sample exposure, not something revealed by the data.

Unlike the draft's utility plus a cross-category switching stock, this utility
subtracts own-category exposure. Positive gamma means fatigue. Identification
arguments must use the actual state definition and sign convention of the model
being estimated.

## 2. An exact identification result for homogeneous coefficients

First consider nonrandom beta and gamma. Let
$\Delta p_{tsj}=p_{tsj}-p_{ts0}$ and
$\Delta z_{itj}=z_{itj}-z_{it0}$. Conditional logit implies

$$
\log\frac{P_{itsj}}{P_{its0}}
=\beta_j+\alpha\Delta p_{tsj}-\gamma\Delta z_{itj}.
$$

For berry and plain, define regressor rows

$$
d_{its,1}=(1,0,\Delta p_{ts1},-\Delta z_{it1}),\qquad
d_{its,2}=(0,1,\Delta p_{ts2},-\Delta z_{it2}),
$$

and $\theta=(\beta_1,\beta_2,\alpha,\gamma)'$.

**Proposition.** Under the stated logit model, observed menus, known initial
states/retention, and the conditional exogeneity assumptions below, the population
choice-probability map identifies $\theta$ if the supported regressor rows span
$\mathbb R^4$. Equivalently, a suitably weighted population Gram matrix
$E[d'd]$ is positive definite.

**Proof.** Suppose $\theta$ and $\widetilde\theta$ generate identical supported
choice probabilities. Their log odds are equal, so
$d(\theta-\widetilde\theta)=0$ on the support. Full column rank implies
$\theta=\widetilde\theta$. Conversely, a nonzero null vector $a$ satisfies
$da=0$. Parameters $\theta+ca$ give identical log odds for any sufficiently
small $c$ that remains in the parameter space, and hence identical probabilities.
Thus an interior parameter cannot be separately identified along that direction.

This is a population result, not a claim that the current empirical regressors
have the required rank. A fixed-coefficient rank proof also does not prove
identification of the mixed-logit model.

## 3. What identifies each homogeneous coefficient?

| Parameter | Distinct source of variation | Failure case |
|---|---|---|
| Berry and plain beta | Relative category odds after accounting for relative prices and exposure | Category absent from every informative menu, or its intercept column is redundant |
| Price alpha | Relative price changes not explained by category constants or exposure | Prices always equal across categories, or relative prices are constant within category and absorbed by beta |
| Common gamma | Relative exposure changes not explained by category constants and relative prices | Exposure differences are constant, or move exactly with relative prices |

To isolate gamma constructively, compare two supported conditions for the same
category pair with the same relative price but different relative exposure:

$$
\Delta\log(P_j/P_0)=-\gamma\,\Delta(\Delta z_j).
$$

Beta cancels. Conversely, relative price changes at fixed relative exposure
isolate alpha. Once these slopes are known, the levels of the log odds identify
the category intercepts. These comparisons are sufficient conceptual examples;
exact matches are unnecessary if the joint design has full rank.

If $\Delta z_j=c_j$ on the whole support, only
$\beta_j-\gamma c_j$ is identified. Increasing gamma and offsetting beta by
$c_j$ times that increase leaves the probabilities unchanged. This gives a direct
observational-equivalence explanation of beta/gamma confounding.

With zero initial stock, first observed weeks have $\Delta z_j=0$. They contain
information about baseline preferences and prices but no direct history slope.
Later purchase occasions contain the history variation. Removing the first
weeks is not required for this argument. In principle, one-week households can
contribute to baseline demand, although they cannot reveal a within-household
history effect. The implemented sample now requires at least two distinct
purchase weeks per household and retains their first-week choices.

## 4. Flavor-specific gamma requires a different rank condition

If each category has its own gamma, relative utility contains

$$
-\gamma_j z_{itj}+\gamma_0 z_{it0}.
$$

One must include the separate own and reference exposure columns in the
regressor matrix, with their category interactions. Own and reference stocks
must vary in sufficiently independent ways to give full rank. Applying the
common-gamma proof to $z_j-z_0$ does not identify three distinct slopes.

The step-2 restriction $\gamma_0=0$ reduces the number of slopes by assumption.
It is not an innocuous utility-location normalization. A zero history regressor
or an unavailable category gives no information about its slope.

## 5. Persistent heterogeneity: what changes in the proof?

The default step-4 model specifies

$$
\beta_{ij}=\mu_j+\tau_j\nu_{ij},\quad \nu_{ij}\sim N(0,1),
\qquad j\in\{1,2\},
$$

with independent normal taste differences and common gamma. The parameter vector
is $\theta=(\mu_1,\mu_2,\tau_1,\tau_2,\alpha,\gamma)$.

For a household history $y_i$, with history stocks rebuilt from its preceding
choices, the model assigns probability

$$
G_\theta(y_i\mid x_i)=
\int\prod_{t,s}\left[
\frac{N_{its}!}{\prod_j y_{itsj}!}
\prod_j P_{itsj}(b_i,z_{it}(y_{i,<t}),p_{ts})^{y_{itsj}}
\right]dF_\theta(b_i).
$$

The same latent type appears throughout the household's sequence. Integration
occurs after multiplying its occasion likelihoods. Past choices select the
posterior distribution of types at later occasions; unconditional average
probability ratios therefore do **not** satisfy the homogeneous linear log-odds
equation. One cannot prove identification by treating the latent household
beta as observed, or by taking logs of integrated probabilities.

The economic sources of information are:

- Taste means shift the relative frequency of category choices across the
  population, accounting for the other parameters.
- Taste variances affect the joint pattern of repeat choices and within-basket
  allocations. Persistent types induce dependence across occasions and cups
  after integrating them out. This is distinct from a new independent taste
  draw on every cup.
- Gamma changes conditional choices with evolving exposure. Diverse histories
  and price paths help distinguish that response from persistent taste selection.
- Alpha changes the joint choice distribution with relative prices.

These are mechanisms, not a proof that all six parameters are identified by any
four-week panel. Multiple cups can provide mixture information even in one week
under the maintained conditional-independence model, but shared basket shocks
or deliberate within-basket variety can invalidate that interpretation.

A rigorous **sufficient local identification condition** for this finite
parametric model is to choose supported history/price/basket conditions $x_m$
and observable events $E_m$, form

$$
M_m(\theta)=\Pr_\theta(Y\in E_m\mid x_m),\qquad
J(\theta)=\frac{\partial M(\theta)}{\partial\theta'},
$$

and show that $J$ has column rank six at the parameter of interest. A nonsingular
six-by-six minor gives local injectivity by the inverse function theorem. Means,
variances, alpha, and gamma are then locally identified jointly, rather than
each being assigned an arbitrary separate moment. A rank-deficient Jacobian can
indicate a first-order failure or weak identification; it is not by itself a
universal proof of nonidentification, because higher-order identification is
possible. Global identification additionally requires ruling out distant
observationally equivalent parameter vectors.

The equivalent regular local criterion uses positive-definite population Fisher
information for full observed histories. Near zero taste SD, differentiation in
the SD parameter can be singular; a variance parameterization and boundary
analysis are needed. A numerical Jacobian at a specified design can illustrate
the condition but cannot replace a theorem establishing its rank over a claimed
support. Simulated recovery and likelihood profiles are useful checks, not proofs.

Adding random gamma introduces another mean/variance or covariance structure
and requires an expanded rank argument. Four purchase occasions do not identify
an unrestricted distribution of beta and gamma simply because household count
is large. Retaining the short panels is compatible with stating a conditional,
parametric identification claim and being explicit about its support assumptions.

## 6. Outside intercept and control-function parameters in the draft

The draft includes an inside intercept and an outside option. With genuinely
observed outside-choice opportunities, normalization of outside utility permits
identifying the inside intercept from yogurt-versus-outside probabilities.
In the purchase-only conditional model, that intercept cancels and is not
identified. Inventing outside counts from a capacity of 14 does not establish
identification of a real participation parameter. Step 4 omits it.

For the draft's first-stage residual $\widehat\eta_{tsj}$, utility adds
$\sigma\widehat\eta_{tsj}$. The rank argument must then include relative price
and relative residual as **separate columns**. Instrument-induced predicted
price variation helps distinguish alpha from sigma. If all price variation is
residual variation after accounting for other regressors, those slopes cannot
be separately identified. An excluded, relevant other-market-price instrument
and a valid control-function error structure are required for the causal
interpretation. The draft's exclusion of common demand shocks across markets
is substantive and cannot be inferred from a first-stage fit. Step 4 currently
does not estimate sigma or perform this first stage.

## 7. Maintained assumptions and a defensible claim

The identification arguments condition on a specified menu and price process,
the logit shock distribution/scale, initial-state treatment, fixed retention,
and a persistent parametric taste distribution. They also require the absence
of unmodeled serial preference shocks that imitate history effects. Treating
purchase dates, stores, and total cups as fixed requires an appropriate
conditional distribution of tastes given those objects, or exogeneity
assumptions; conditioning alone does not remove selection.

Mean-price fallbacks and assuming every category is available preserve the
sample. They are measurement assumptions, not sources of new identifying
variation. Keep their role separate from the theoretical proof.

A defensible presentation to an advisor is: (1) state the population model and
normalizations, (2) prove the fixed-coefficient rank result, (3) state and establish
the additional history-probability rank condition for the proposed parametric
heterogeneity, and (4) assess whether the observed short-panel design supplies
that variation. The first proof alone does not justify claiming that the mixed
model's taste variances or random fatigue distribution have been identified.
