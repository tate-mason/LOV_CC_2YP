# Why `run_all.py` fails

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
|---|---|
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
