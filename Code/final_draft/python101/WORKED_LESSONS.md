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
