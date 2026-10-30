> Historical October 28 diagnosis, superseded by the October 30 symmetric-switching model. See MODEL_CONVENTIONS.md and the current compiled appendix.

# Why `run_all.py` fails

> **Status update (2026-10-28):** The failures below describe the earlier investigation. The current source contains all four proposed fixes, and a fresh `python3 run_all.py` run exits successfully. See the current status and recommended next steps at the end of this document.

Running `python3 run_all.py` reproduces an assertion failure in `checks/check_focs.py:12`. The runner itself is working: it launches each check with `check=True`, so the first unsuccessful check stops execution and produces the outer `subprocess.CalledProcessError`. That outer exception is a consequence of the failed assertion.

## 1. Immediate cause: missing denominator parentheses

In `model/shares.py:6–11`, all four share expressions have missing parentheses. For example:

```python
sj2_expr = sp.exp(U2) / 1 + sp.exp(U2)
```

Python interprets this as `(exp(U2) / 1) + exp(U2)`, which equals `2 * exp(U2)`. It does not represent the intended logit share. Similarly, `1 / 1 + exp(U2)` equals `1 + exp(U2)`.

The check differentiates profit using this incorrect share, then compares the result with a formula that assumes the logit identity `ds/dp = alpha * s * (1 - s)`. Those expressions disagree.

For `E = exp(U2)`, the actual derivative of the current share is `2 * alpha * E`, while the derivative assumed by the check is `2 * alpha * E * (1 - 2 * E)`. Consequently, the assertion's difference is

```text
4 * alpha * s_j1 * (p2 - c) * exp(2 * U2)
```

which is not identically zero.

The intended definitions are:

```python
sj1_expr = sp.exp(U1) / (1 + sp.exp(U1))
s01_expr = 1 / (1 + sp.exp(U1))
sj2_expr = sp.exp(U2) / (1 + sp.exp(U2))
s02_expr = 1 / (1 + sp.exp(U2))
```

I verified in an isolated symbolic calculation that the period-2 FOC residual becomes exactly zero with the corrected share.

## 2. A second failure is currently hidden

Running the checks individually also reveals an assertion failure in `checks/check_alpha.py:14`.

The file constructs both the total derivative and the direct partial derivative:

```python
dsj1_dalpha_manual = sj1 * s01 * (p1 + alpha * P1_alpha)
dsj1_dalpha_direct = p1 * sj1 * s01
dsj1_dalpha_check = dsj1_dalpha_direct + dsj1_dp1 * P1_alpha
```

But the assertion compares the total derivative with only the direct partial derivative. Their difference is `alpha * P1_alpha * sj1 * s01`, which is generally nonzero because the total derivative includes the response of price to alpha.

The intended assertion appears to be:

```python
assert sp.simplify(dsj1_dalpha_manual - dsj1_dalpha_check) == 0
```

I verified that this corrected comparison simplifies to zero. Fixing only the share expressions will therefore leave `run_all.py` failing later at the alpha check.

## Results from running each check separately

| Command | Observed result |
| --- | --- |
| `python3 -m checks.check_focs` | Fails at the period-2 FOC assertion |
| `python3 -m checks.check_gamma` | Exits successfully; prints expressions and cases |
| `python3 -m checks.check_beta` | Exits successfully; no output |
| `python3 -m checks.check_alpha` | Fails at the derivative assertion |
| `python3 -m checks.check_delta` | Passes its symbolic assertion |

The gamma and beta scripts contain no assertions, so successful execution alone does not establish that their comparative statics are correct. Also, despite its plural name, `check_focs.py` currently checks only the period-2 FOC.

## Additional issues outside the immediate failure

These do not cause the reproduced `run_all.py` failure, but deserve attention before treating the model as validated:

- **`pricing/period1_price.py:4`: missing multiplication operator.** `(delta / alpha)(sj2 / s02)` attempts to call a SymPy expression as a function. Importing this module independently reproduces `TypeError: 'Mul' object is not callable`. The intended term appears to be `(delta / alpha) * (sj2 / s02)`.
- **`pricing/period1_foc.py:6`: incorrect collected FOC.** The expression named `foc1_collected` is not algebraically equal to `foc1_raw`: it drops the standalone `sj1` term and introduces an extra factor of `sj1`. An equivalent expression would be `sj1 + dsj1_dp1 * (p1 - c + delta * sj2 * (p2 - c))`, or, after substituting the share derivative, `sj1 * (1 + alpha * s01 * (p1 - c + delta * sj2 * (p2 - c)))`.

## Suggested next steps

1. Correct the four share definitions.
2. Compare the two total-derivative expressions in the alpha assertion.
3. Rerun `python3 run_all.py`.
4. Correct the two pricing issues and add meaningful checks for those expressions before relying on an “All checks passed” message as model validation.

This investigation changed no Python source files. The proposed fixes were checked with isolated symbolic calculations; they have not been applied to the project.

## Current status and recommended next steps (2026-10-28)

On reinspection, the four share denominators are parenthesized, the alpha assertion compares the two total derivatives, the period-1 price expression includes the multiplication operator, and the collected period-1 FOC retains the standalone `sj1` term. Running `python3 run_all.py` now exits with status 0 and prints “All checks passed!”. The earlier statement that the fixes have not been applied is historical.

The next priority is stronger validation rather than further changes to those repaired expressions:

1. **Protect the period-1 fixes with regression checks.** Extend `checks/check_focs.py` to assert that `foc1_raw - foc1_collected` simplifies to zero. Also derive the period-1 FOC from the profit expression and compare it with the implemented FOC under explicitly stated price dependencies. Add an import check for `pricing.period1_price` and verify its price equation against the FOC after substituting the period-2 pricing relation. Include these checks in `run_all.py` so those modules are actually exercised.
2. **Turn the beta and gamma scripts into checks.** `checks/check_beta.py` currently constructs expressions without asserting anything; `checks/check_gamma.py` prints formulas and cases. Independently differentiate the defining pricing equations and compare the resulting derivatives with the comparative-statics modules. Assert the quotient rule for the price ratio and the gamma cases `(0,0)`, `(1,0)`, and `(0,1)`. Under the documented assumptions, the ratio derivative should be zero, positive, and negative, respectively, for these cases.
3. **Make derivative conventions explicit.** Document which prices respond to each parameter and which quantities are held fixed. Review `model/shares.py`, especially `dsj2_dgamma_total = sj2 * s02`: a total derivative of the stated period-2 logit share would generally include `z2 + alpha * dp2/dgamma`. Establish whether this variable is intended as a coefficient, a partial derivative, or a total derivative before changing it. Likewise, the delta check holds `R2` fixed; document that restriction when interpreting its result.
4. **Check assumptions and numerical consistency.** `model/assumptions.py` records assumptions as strings; it does not impose them on SymPy expressions. Apply suitable assumptions or explicit substitutions when checking signs. At several feasible parameter values, solve the pricing equations, check FOC residuals and second-order conditions, and compare symbolic comparative statics with finite differences of the solved prices. Re-solve prices after each parameter perturbation and ensure prices and shares satisfy the stated feasibility conditions.
5. **Reconcile outputs before regenerating figures.** Compare the modular formulas with `comparative_statics_symbolic.py` and `make_comparative_static_figures.py`, including the negative `(0,1)` ratio derivative documented in `README_comparative_statics.txt`. Once the assertions and numerical comparisons pass, regenerate the figures and inspect their sign labels.

Completion should mean that every check in the runner makes a meaningful assertion, both periods' pricing expressions are covered, and numerical comparisons support the symbolic formulas under documented assumptions. A successful runner exit currently establishes only the assertions it actually executes.

This status update changes documentation only. The runner was executed successfully; the additional checks recommended above have not yet been implemented.

## Recommended work completed (2026-10-28)

All five recommendations above have now been implemented. See `MODEL_CONVENTIONS.md` for derivative conventions, corrected formulas, validation scope, and figure guidance. Both FOCs and pricing equations have regression checks; beta/gamma scripts assert quotient rules and gamma signs; implicit price derivatives and the profit envelope are checked symbolically. Numerical checks re-solve prices, verify feasibility, FOC residuals and negative definite profit Hessians, and compare comparative statics with finite differences at 27 solutions. The alpha, mixed beta-gamma, and profit derivative formulas required corrections. The standalone symbolic script imports the modular formulas. Figures now use re-solved prices and endogenous shares and are exported to `figures/` with CSV source data; the (0,1) ratio slope is negative. Earlier statements that the additional checks remain unimplemented are historical.
