# Worked reading lessons: unpacking the intimidating lines

The file-by-file chapters below are a reference. These short lessons demonstrate
how to slow down and reason through a dense expression. The expanded versions
are teaching equivalents, not suggested edits to the production code.

## Lesson 1: a table pipeline is a sequence of ordinary calls

The estimator contains this expression:

```python
weekly = (lines.groupby(keys + ["flavor"])["quantity"].sum()
          .unstack("flavor", fill_value=0)
          .reindex(columns=FLAVORS, fill_value=0).reset_index())
```

Read left to right after identifying the innermost list expression:

1. `keys + ["flavor"]` concatenates the household/week key list with one more
   column label. The plus sign joins lists; it is not adding column values.
2. `lines.groupby(...)` splits purchase rows into household/week/flavor groups.
3. `["quantity"]` selects the quantity Series from that grouped object.
4. `.sum()` adds package counts within each group. Two separate berry purchases
   in the same household-week become one berry total.
5. `.unstack("flavor", fill_value=0)` moves the flavor part of the grouped index
   into columns. An unobserved flavor within an existing week gets zero count.
6. `.reindex(columns=FLAVORS, fill_value=0)` fixes the columns to other, berry,
   plain, in that order, adding any completely absent flavor with zeros.
7. `.reset_index()` turns household/week index levels into ordinary columns.
8. `weekly = ...` names the resulting table only after the whole chain completes.

You could write intermediate objects instead:

```python
group_columns = keys + ["flavor"]
grouped = lines.groupby(group_columns)["quantity"]
totals = grouped.sum()
wide = totals.unstack("flavor", fill_value=0)
ordered = wide.reindex(columns=FLAVORS, fill_value=0)
weekly = ordered.reset_index()
```

Both forms express the same intended pandas operations. Intermediates are useful
while learning or debugging because you can inspect each type and shape. Chaining
keeps a completed transformation compact once its stages are familiar. It is
not intrinsically more statistically correct, and it is not automatically lazy
just because it is written across several lines.

For a household with berry purchases 2 and 1 and plain purchase 4 in one week,
the final count columns are `(other=0, berry=3, plain=4)`. Index order becomes
important only when those columns are converted to position-based arrays.

## Lesson 2: masks express a sentence one cell at a time

```python
complete = available.any(axis=1) & ((counts == 0) | available).all(axis=1)
```

Suppose one week has counts `[0,2,1]` and availability `[True,True,False]`.
The zero-count comparison gives `[True,False,False]`. OR with availability gives
`[True,True,False]`: the plain purchase has no price. `.all(axis=1)` therefore
gives False for that row. Although `.any(axis=1)` confirms a nonempty menu,
AND combines it with False, so the week is rejected.

If plain count were zero, the OR vector would be all True, so the missing plain
price would be acceptable: plain can be excluded from the choice set. This
distinguishes "unpriced and unpurchased" from "unpriced but purchased."

The expression could be expanded into a row loop and an inner flavor loop:

```python
complete_rows = []
for row_counts, row_available in zip(counts, available):
    nonempty_menu = any(row_available)
    every_purchase_covered = all(
        q == 0 or offered
        for q, offered in zip(row_counts, row_available)
    )
    complete_rows.append(nonempty_menu and every_purchase_covered)
```

The loop uses scalar `or` and `and` because q and offered are individual values.
The production expression uses `|` and `&` because its comparisons concern
whole arrays. Vectorization moves the element loops into NumPy, but the logical
rule remains the same. Parentheses are necessary: do not rely on English-like
operator precedence in an array expression.

## Lesson 3: from named coefficients to a utility matrix

```python
p = dict(zip(parameter_names(model, data["outside"], data["control"]), theta))
beta = np.array([0.0, p["beta_berry"], p["beta_plain"]])
```

The first call obtains parameter labels for this particular model. `zip` pairs
each with the coefficient in the same position; `dict` makes the lookup mapping.
For a naive three-parameter model, `[.4, -.2, 0.]` corresponds to berry intercept
.4, plain intercept -.2, and mean log-minus-alpha zero. It does not mean alpha
is zero: the later transform is `-exp(0) = -1`.

With no gamma or price-control effect and one week's prices `[1,1.2,1.5]`, the
utility calculation is

```python
beta = np.array([0., .4, -.2])
alpha = -1.
prices = np.array([[1., 1.2, 1.5]])
utility = beta + alpha * prices
# utility is [[-1., -.8, -1.7]]
```

Beta has shape `(3,)`, prices `(1,3)`, and the output `(1,3)`. Each flavor's
intercept is added to that flavor's price term. Alpha's negative sign means
higher prices lower utility. The other-flavor intercept is fixed at zero as a
normalization; that does not imply other flavor has zero total utility.

Why not use positions such as `theta[0]` everywhere? That is possible, but a
different model inserts/removes coefficients and can change positions. A single
ordered label helper plus dictionary lookup makes the interpretation visible.
The numeric values remain in an array for efficient optimization.

## Lesson 4: softmax, log probabilities, and zero counts

```python
logprob = utility - logsumexp(utility, axis=1, keepdims=True)
row_ll = (data["counts"] * np.where(data["counts"] > 0, logprob, 0.)).sum(axis=1)
```

For logit, the logarithm of a probability is utility minus the log of the
sum of exponentiated utilities. `axis=1` performs that denominator calculation
across alternatives within each week. `keepdims=True` preserves a W-by-1 column
so it can be subtracted from the W-by-J utility matrix.

An unavailable alternative has utility minus infinity and probability zero.
Its log probability is minus infinity. A zero purchase count should contribute
zero to the log likelihood, but floating point computes `0 * -inf` as NaN.
The `where` replaces that log probability with zero **where count is zero**
before multiplying, protecting the valid zero contribution.

For counts `[0,2,1]`, the row contribution is two times berry log probability
plus one times plain log probability. This is why changing an actual package
count changes the likelihood's weight; it is not just a single flavor label.
Subtracting raw `np.exp(utility).sum()` rather than its log would be the wrong
formula, and computing exp then log naively can overflow. `logsumexp` handles
the same mathematics in a stable numerical form.

## Lesson 5: one household must keep its latent type across weeks

Assume two possible types, equally weighted. Type A gives the household's chosen
flavor probability .9 in each of two weeks; type B gives .1 in each. Persistent
types imply the history probability

```python
likelihood = (.9 * .9 + .1 * .1) / 2
# .41
```

Averaging probabilities independently each week instead gives

```python
wrong_history = ((.9 + .1) / 2) * ((.9 + .1) / 2)
# .25
```

The difference is not rounding: it is whether the household can redraw its type
between weeks. The estimator first adds weekly log likelihoods within household
for each draw, then integrates those full-history likelihoods. `np.bincount`
performs the within-household sum. A list accumulates one household vector per
draw. The final `logsumexp(..., axis=0) - log(R)` integrates the draw dimension
while preserving the household dimension.

This also explains why the gradient uses posterior draw weights. A type assigning
higher probability to the entire observed history receives greater weight in
that household's score. Those weights are calculation intermediates, not new
estimated household coefficients written by the program.

## Lesson 6: understanding the gradient's einsum without memorizing it

```python
gradient = -np.einsum("rh,rhp->p", weights, np.asarray(scores)) / data["n_households"]
```

The quoted letters label axes; they are not Python variable names. `r` is draw,
`h` household, and `p` parameter. Weights has shape `(R,H)`, scores `(R,H,P)`.
Multiply matching draw/household entries, sum over r and h because they do not
appear after the arrow, and retain p. The resulting vector has length P.

Here is the same sum written in a teaching loop:

```python
gradient = np.zeros(P)
for parameter in range(P):
    for draw in range(R):
        for household in range(H):
            gradient[parameter] += (
                weights[draw, household] * scores[draw, household, parameter]
            )
gradient = -gradient / H
```

The negative sign converts a log-likelihood score to a negative-log-likelihood
gradient. Division by H matches the mean loss returned to the optimizer. The
compact NumPy form avoids Python executing three nested numerical loops, while
the teaching loop makes the axis meaning explicit. In this example P/R/H are
dimension labels you would obtain from the actual arrays, not existing global
constants in the source.

## Lesson 7: first-stage least squares does not mean matrix inversion

```python
z_partial = z - controls @ np.linalg.lstsq(controls, z, rcond=None)[0]
```

Starting inside, `lstsq(controls,z,...)` fits z on the control columns. Its return
contains coefficients plus diagnostics, so `[0]` extracts the coefficient vector.
Matrix multiplication produces fitted z, then subtraction leaves the part not
linearly explained by those controls. `rcond=None` requests NumPy's default
singular-value cutoff convention for that calculation.

For a three-row example, let controls be a column of ones and z `[1,2,6]`.
Least squares estimates the mean 3, fitted values `[3,3,3]`, and residual
`[-2,-1,3]`. The production controls are Xi after fixed-effect absorption,
but the projection arithmetic has the same structure.

The alternative formula involving an inverse of `X.T @ X` requires nonsingular
cross-products and can be numerically less stable. `lstsq` handles rank deficiency
using its numerical decomposition. It cannot create excluded-instrument variation
where none exists; the following sum-of-squares check rejects that case.

## Lesson 8: symbolic expressions are not hidden numerical functions

```python
alpha, beta, gamma = sp.symbols("alpha beta gamma", real=True)
price = sp.Symbol("price", real=True)
index = beta + alpha * price + gamma
direct = sp.diff(index, gamma)
```

`direct` is the symbolic number 1. No household data, optimizer, or probability
is involved. If a separate symbol s is introduced, SymPy does not know that s
depends on price unless the dependence is encoded. This is why the supply code
implements explicit total-derivative rules.

`sp.lambdify((alpha,beta,gamma,price), index, "numpy")` builds a callable expecting
four numbers in that order. A call with `(-1,2,.5,1)` gives 1.5. Passing beta
where alpha belongs gives a different value without necessarily raising an error.
Named dictionaries at output do not remove the need to keep this positional
bridge correct. Symbolic algebra improves formula reuse, while numerical tests
at asymmetric states catch argument-order mistakes.

## Lesson 9: closures and the expanding price bracket

```python
f = lambda p: alpha*(p-effective_cost) + 1 + np.exp(
    np.clip(intercept + alpha*p, -745, 700)
)
lo = effective_cost
width = max(2., 2/abs(alpha))
hi = lo + width
while f(hi) > 0:
    width *= 2
    hi = lo + width
```

Only p is written as a lambda parameter; alpha, effective cost, and intercept
come from the enclosing call. Evaluating `f(hi)` substitutes the current hi
without changing those fixed inputs. Under the maintained negative-alpha domain,
f starts positive at lo and eventually becomes negative as p increases.
Doubling width searches for a reliable sign-changing bracket rather than
guessing a universal maximum price.

Once bracketed, Brent's root method solves f(p)=0. It is solving the transformed
FOC, not trying to make the price itself zero. A loop that forgot to update hi
could never change its condition; a loop with `width += 2` would expand more
slowly for far-away roots. The mathematical guide establishes why this unique
FOC root is the optimum for the current unconstrained logit problem.

## Lesson 10: practice before editing the real scripts

For each exercise, predict the result before checking the answer:

1. For model `naive`, outside False, control True, what parameter names are
   returned? Answer: beta_berry, beta_plain, mu_log_minus_alpha, price_control.
2. If a household is selected twice by bootstrap, why change its ID twice?
   Answer: the two sampled histories must remain separate likelihood contributions.
3. What does `.copy()` protect when changing bootstrap IDs? Answer: the original
   household subtable that may be selected again in another draw.
4. Is a lazy query just a synonym for a small table? Answer: no; it is a plan,
   and executing it can still need large intermediate state.
5. In `return fitted, results, bounds`, why do the commas matter? Answer: they
   construct a tuple the caller can unpack; there are three outputs.
6. Why is saving `.tex` not the same as saving `.pdf`? Answer: the former is
   source text requiring a TeX engine to typeset it.
7. Can a zero outside count justify adding missing purchase weeks? Answer: no;
   the implemented sample contains only observed purchase weeks.
8. Does a first-stage F statistic establish exclusion? Answer: no; it describes
   conditional instrument relevance, while exclusion is an economic assumption.

You can learn most of the syntax by running tiny independent arrays and tables.
Read the legacy scripts as text before executing them: several contain top-level
filesystem writes and replacements. The following chapters reproduce that source
without executing it. The numbered listings are fragments for reading; a
fragment beginning inside a function is not necessarily runnable on its own.

# Current empirical workflow: separate switching and habit

These lessons update the earlier combined-gamma examples. Begin with the readable
files under `Code/final_simple/demand/`; the source chapters later in this book
include both these files and their original `final_draft` counterparts. The old
model remains a comparison, rather than the current empirical specification.

## Lesson 11: two coefficients mean two independently estimated effects

The original combined index was

\[
 \Xi_{ijt}=(1-\lambda)N_{ijt}-\lambda S_{ijt},\qquad
 V_{ijt}=\beta_j-a_i p_{ijt}+\gamma_i\Xi_{ijt}+\rho cf_{ijt}.
\]

It forced novelty and own-stock coefficients to move together. The split model
instead estimates

\[
 V_{ijt}=b_0+\beta_j-a_i p_{ijt}+b_{Ni}N_{ijt}+b_{Si}S_{ijt}+\rho cf_{ijt}.
\]

`N` is recent other-flavor purchase shares, `S` is accumulated own-flavor shares,
and `a` is positive price sensitivity. A positive novelty coefficient rewards
switching away from recently purchased rivals. Positive own-stock effects mean
reinforcement; negative ones mean satiation. The name habit does not constrain
its estimated sign. These are purchase histories, not observed consumption.

For example, let berry's N be .75 and S be 2, with b_N=.4 and b_S=-.1. The history
part of berry utility is .4*.75-.1*2=.1. You do not multiply these coefficients by
.3 and -.7 again. Those weights belong to the older combined-index model.

In `final_simple/demand/estimation.py`, choose:

```python
model = "heterogeneous_history"
capacity = 7
```

The estimator already supports this model. Updating distribution exports does
not require changing its likelihood or re-estimating a saved fit. Capacity 7
adds constructed nonpurchase counts; it is a maintained participation convention,
not observed purchase-free weeks. Capacity 0 remains available for conditional
flavor demand, but the present pricing solver requires positive integer capacity.

## Lesson 12: a population distribution is not an estimation confidence interval

The split population model uses independent normal novelty/stock effects and
independent lognormal price sensitivity:

```python
b_n = mean_novelty + sd_novelty * draws[:, 1]
b_s = mean_own_stock + sd_own_stock * draws[:, 2]
a = np.exp(mu_log_minus_alpha + sd_log_minus_alpha * draws[:, 0])
```

Every row of `draws` defines one persistent type across all weeks. The normal
SDs describe variation across consumers, not uncertainty about the estimated
population means. Hessian or bootstrap SEs answer the latter question.

For the previously saved controlled split fit, the exported novelty mean/SD are
about -.4646/.8466 and own-stock mean/SD about -.01703/.09117. These are existing
fit values, not results of a new optimization or new pricing run. Their fitted
normal sign shares are about 70.84% and 57.41% below zero respectively. A negative
novelty coefficient reverses the recent-other-flavor incentive; it need not imply
positive cumulative reinforcement when the two effects are estimated separately.

`export_history_distributions` evaluates each normal's PDF and CDF, stores
quantiles, and draws both marginal densities. With SD zero, it records a point
mass instead of evaluating a normal density with zero denominator. It writes:

- `history_distribution_summary.csv`: parameters, sign shares and quantiles.
- `history_distribution_grid.csv`: density/CDF values for plotting.
- `history_distributions.pdf` and `.png`: the two population marginals.

The readable saved-fit entry point is `demand/history_distributions.py`. Existing
plots are copied under `final_simple/demand/fitted_distributions/`. Their source
fits and interpretation are recorded there. No household table is needed merely
to export these marginal distributions from saved coefficients.

## Lesson 13: recover both household effects, but rank only by switching

A household's posterior type probabilities use its complete history:

\[
 \ell_{id}=\sum_{t,k}n_{ikt}\log P_{ikt}(a_d,b_{Nd},b_{Sd}),\qquad
 w_{id}=\frac{\exp\ell_{id}}{\sum_m\exp\ell_{im}}.
\]

The numerical implementation subtracts log-sum-exp rather than exponentiating
large log likelihoods directly. Add weeks before integrating draws; averaging
weekly probabilities would redraw the person's type every week.

The split pricing script returns a D-by-2 array `history`. Column zero is b_N;
column one is b_S. For one household:

```python
weights = np.array([.75, .25])
history = np.array([[.4, -.1], [-.2, .3]])
mean_type = weights @ history        # [.25, 0.]
```

The posterior mean switching coefficient is .25 and mean stock effect is zero.
The code ranks households by the first mean only, assigns bottom/top quarters,
and keeps both means in utility. It does not set stock effects to zero just
because stock is absent from the ranking variable. Posterior coefficient
uncertainty can be correlated even when population integration draws are independent.

The default firm treats posterior means as known fixed types. The optional
posterior-integrated exercise retains the joint draw weights and updates them
after the first counterfactual basket. A terminal coupon must optimize revenue
averaged over remaining unobserved types; optimizing separately for each true
draw would give the firm information it does not have.

Type estimates use the full observed sample, but the first observed basket alone
initializes counterfactual history. This is a retrospective exercise with known
estimated types, not a prospective holdout forecast. Comparing novelty quartiles
does not isolate novelty causally: their habits, prices and initial states differ.

## Lesson 14: three focal flavors, one inside-versus-outside calculation each

For focal berry, combine other and plain as the outside alternative. Include
constructed nonpurchase when present. If utilities are V_o,V_b,V_p,0, then

\[
 I_{-b}=\log\{e^{V_o}+e^{V_p}+1\},\qquad
 P_b=\operatorname{logistic}(V_b-I_{-b}).
\]

This equals berry's original multinomial-logit probability exactly. The outside
utility is a log-sum-exp, not the arithmetic mean of rival utilities. With all
four utilities zero, each original alternative has probability 1/4; the composite
outside has utility log(3), so the binary formula also gives 1/4, not 1/2.

`focal_flavor = "all"` repeats the seller exercise for other, berry and plain.
Only that run's focal product earns revenue and changes price. These are three
separate counterfactuals with fixed competitors, not three firms solving a joint
equilibrium. The solver internally uses coordinate one for the priced product.
`run_households` swaps the focal product's counts, prices, histories, controls and
intercept into that coordinate together. Household posterior types are recovered
before that swap under the original fitted labels.

Combining rivals in a current choice probability does not erase their histories.
Other and plain baskets can produce different future stocks and inclusive values.
Keep their identities in basket integration. The seller observes the full basket
before the terminal coupon; observing only focal sales would require additional
beliefs about which rival flavors were bought.

## Lesson 15: read backward induction in the order it is solved

There are three choice dates but two pricing decisions:

1. The first observed basket establishes initial h0 at model t=0.
2. The firm chooses p1(h0); a basket is realized and establishes h1.
3. It observes h1 and chooses terminal p2(h1).

For each basket n and each known estimated type, the terminal value is

\[
 W_2(n)=\max_{d_2}\ p(d_2)K P_{f,2}(n,d_2),
 \qquad p(d)=(1-d)p_{\mathrm{posted}}.
\]

The initial action solves

\[
 \max_{d_1}\ p(d_1)K P_{f,1}(d_1)
 +\delta\sum_n M(n\mid d_1)W_2(n).
\]

Read `targeted_coupons` in `final_simple/demand/split_pricing.py`: the outer loop
visits one household. STEP A calculates terminal outcomes for all coupons and
baskets. STEP B calculates first-action basket probabilities and chooses each
basket's best terminal coupon. STEP C picks the best first coupon. The terminal
maximization happens before averaging over baskets. Both focal-purchase and
no-focal-purchase continuation branches count, including all-nonpurchase baskets.

For basket shares x, update S2=S1+x and N2=sum(x)-x. All-nonpurchase means x=0,
leaving stock unchanged and setting next novelty to zero. Capacity 7 yields 120
possible baskets. Updating only at expected shares would generally yield a
different value function because logit demand and optimized prices are nonlinear.

The known-type, observed-state timing matches the supply baseline. Empirical
binary aggregation is exact at a given full state, but it does not reduce the
whole split-stock model to the theory's symmetric +gamma/-gamma one-lag model.
The empirical solver also restricts prices to coupons from 0% through 100%, while
theory allows unrestricted prices and introductory subsidies. Both new split
pricing and theoretical supply default to delta=.92. With zero cost, net revenue
is profit. Consumers follow current-period demand without strategically waiting.

## Lesson 16: reconstruct deleted inputs without estimating demand again

The weekly export was deleted; regenerate it on the server where its underlying
sample lives. `demand/regenerate_weekly_inputs.py` has three conceptual stages:

```python
frame = prepare_data(saved_settings)
frame, first_stage = add_price_controls(frame)  # for a controlled fit
# Compare regenerated counts and the fixed-parameter likelihood with fit.json.
# Write weekly_inputs.csv only after both checks pass.
```

This schematic omits the implementation's guards and temporary-file publication.
It does not call `optimize_demand`. Reconstructing price controls is necessary to
recover the same regressors; it does not replace the saved demand estimates or
standard errors. The procedure restores original years, sample selection, history
weight, capacity, draw seed/count and model from the saved fit. Moved sample or
flavor-reference paths can be overridden explicitly. A mismatch stops rather
than silently pairing old coefficients with a different weekly sample.

The server job is `Code/final_simple/server/run_split_pricing.sh`. Its numbered
steps regenerate the controlled fit's table, export both marginal distributions
from controlled and uncontrolled fits, then run all three focal cases using the
controlled fit. Sync code and install the listed requirements in the server
environment before submitting:

```bash
sbatch Code/final_simple/server/run_split_pricing.sh
```

This command is an instruction for the server, not something the book builder
executes. Reading or rebuilding this book parses source text only. Local
validation used generated inputs with CSV reads blocked; full empirical pricing
has not been run locally. The saved marginal distribution plots are not new
pricing results.

Begin reading pricing outputs at the root `focal_flavor_comparison.csv` and
`run_notes.txt`, then open each focal subfolder's posterior summaries, household
policies, group/aggregate tables and figures. `focal_units` always means the
selected run's product. The manifest maps rival state coordinates back to flavor
names. Keep fit records beside outputs so these consistency checks remain possible.


## Lesson 17: restricting focal sellers does not restrict demand alternatives

For the main comparison, use `final_simple/demand/plain_berry_pricing.py`.
Its selection is:

```python
FOCAL_FLAVORS = ("plain", "berry")
```

This tuple controls the seller problems that are solved, not which products are
in the estimated choice set. The shared solver still uses all three demand
categories and both estimated history coefficients. For focal plain, berry and
other form the composite outside alternative; for focal berry, plain and other
do so. Constructed nonpurchase remains outside too. There is no refit and no
new flavor-specific switching distribution: household coefficients remain shared
across flavors as estimated.

Plain and berry are easier to interpret as focal products than the broad other
aggregate. The three-focal script remains a sensitivity exercise. The two-focal
entry point creates only plain/ and berry/ result folders and a combined
comparison table. It changes neither competitor prices jointly nor the
underlying demand sample. Run it on the server after weekly-input regeneration,
using the same fit directory and weekly table as the three-focal exercise.
