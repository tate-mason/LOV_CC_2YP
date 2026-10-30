> Historical October 28 diagnosis, superseded by the October 30 symmetric-switching model. See MODEL_CONVENTIONS.md and the current compiled appendix.

# Derivative changes: previous formulas versus corrected formulas

This document compares the formulas present before the diagnostic follow-up with the corrected formulas. Three existing derivative formulas changed: the period-1 price derivative with respect to alpha, the optimized-profit derivative with respect to gamma, and the mixed beta–gamma derivative of period-1 price. The gamma price derivatives and their price-ratio derivative did **not** change.

## Notation and comparison convention

Write s1 = s_j1, s2 = s_j2, q1 = s_01 = 1-s1, and q2 = s_02 = 1-s2. Let a = alpha < 0 and d = delta in (0,1). Prices and shares below are evaluated at the solution of both pricing FOCs:

```text
F1 = p1 - c + 1/(a*q1) - d*s2/(a*q2) = 0
F2 = p2 - c + 1/(a*q2) = 0.
```

Both optimal prices respond to the parameter being differentiated. The other model parameters, marginal cost, and history indicators are held fixed. Shares respond through utility and price. Profit is

```text
Pi = s1 * [p1 - c + d*s2*(p2 - c)].
```

Numerical comparisons evaluate old and new formulas at the **same solved prices and shares**. The pricing equations themselves were not changed by this follow-up. Thus the reported differences isolate derivative corrections, not a change in the calibration or equilibrium prices.

## 1. Period-1 price derivative with respect to alpha

File: `comparative_statics/alpha_p1.py`.

Previous:

```text
old = [1/q1 - p1*s1 - d*(s2/q2)*(1/a)*(1/q2 - p2*s2)] / a
```

Corrected:

```text
new = 1/a^2 - p1*s1/a + d*q1*s2*p2/a
```

The exact additive correction is

```text
new - old = 1/a^2 - 1/(a*q1)
            + d*q1*s2*p2/a
            + d*s2*(1/q2 - p2*s2)/(a^2*q2).
```

The previous expression did not satisfy implicit differentiation of the two pricing equations. To obtain the corrected expression, first differentiate F2, including the total share response:

```text
dp2/da = 1/a^2 - p2*s2/a.
```

The relevant derivatives of F1 are

```text
F1_p1 = 1/q1
F1_p2 = -d*s2/q2
F1_a  = -1/(a^2*q1) + p1*s1/(a*q1)
        + d*s2/(a^2*q2) - d*p2*s2/(a*q2).
```

Then solve `F1_p1*(dp1/da) + F1_p2*(dp2/da) + F1_a = 0`. This simplifies to the corrected formula. It is not merely a rearrangement of the old expression: the numerical comparison below shows a sign reversal throughout the 27-point validation sample.

## 2. Optimized-profit derivative with respect to gamma

File: `comparative_statics/gamma_profit.py`. The standalone symbolic display now imports this formula instead of maintaining a separate copy.

Previous:

```text
old = -(z1*s1 + d*z2*s2)/a
```

Corrected:

```text
new = -s1*(z1 + d*z2*s2)/a
```

The exact additive correction is

```text
new - old = d*z2*s2*q1/a.
```

The continuation-profit contribution must be multiplied by the probability of a period-1 purchase, s1. Using the envelope theorem, differentiate profit holding decision prices fixed, then substitute the FOCs:

```text
partial Pi / partial gamma
    = z1*s1*q1*[p1-c+d*s2*(p2-c)]
      + d*s1*z2*s2*q2*(p2-c)
    = -z1*s1/a - d*z2*s1*s2/a.
```

Consequences by history:

| History | Effect of correction |
| --- | --- |
| (0,0) | Both formulas are zero. |
| (1,0) | Unchanged: both equal -s1/a. |
| (0,1) | New derivative equals s1 times the old derivative. It remains positive, but is smaller by the fraction 1-s1. |

Thus, for (0,1), the percentage reduction is exactly `100*(1-s1)%`. The size depends on the solved first-period share; it is not universally 98%.

## 3. Mixed beta–gamma derivative of period-1 price

File: `comparative_statics/gamma_p1beta.py`.

The underlying beta derivative is unchanged:

```text
dp1/dbeta = (d*s2*q1 - s1)/a.
```

Define the total gamma share responses and the own-price derivative of the outside share:

```text
A = ds1/dgamma = s1*q1*(z1 + a*dp1/dgamma)
B = ds2/dgamma = s2*q2*(z2 + a*dp2/dgamma)
T = partial q1 / partial p1 = -a*s1*q1.
```

Previous:

```text
old = d*(s2*q1*B*T - s1*A)/a
```

Corrected:

```text
new = [d*(q1*B - s2*A) - A]/a
```

The exact additive correction is

```text
new - old = [d*q1*B*(1-s2*T) + (d*s1-d*s2-1)*A]/a.
```

This follows directly from the product rule applied to `d*s2*q1 - s1`, with `dq1/dgamma = -A`. The old formula multiplied several derivative factors together instead of adding the two product-rule terms, and did not correctly differentiate the standalone `-s1` term. These errors can make the old result very small, so percentage changes can be enormous even when the absolute change is modest.

## 4. Share derivative naming and newly filled formulas

In `model/shares.py`, the old name

```text
dsj2_dgamma_total = s2*q2
```

was replaced by

```text
dsj2_dutility = s2*q2.
```

The quantity s2*q2 is the derivative with respect to the utility index. It is not, in general, a total gamma derivative. The existing correct total derivative in `comparative_statics/gamma_p2.py` remains

```text
ds2/dgamma = s2*q2*(z2 + a*dp2/dgamma) = z2*s2*q2^2.
```

This was a naming clarification in the shared module; it did not change the total share derivative already used by the gamma comparative-statics module. At history (0,0), for example, the actual total gamma derivative is zero, while the utility coefficient s2*q2 is positive.

Two previously empty modules were filled:

```text
comparative_statics/beta_p2.py:
    dp2/dbeta = -s2/a

comparative_statics/beta_p1p2.py:
    d(p1/p2)/dbeta = [p2*(dp1/dbeta) - p1*(dp2/dbeta)]/p2^2.
```

These are additions, so there is no previous numerical formula against which to measure a change.

## 5. Numerical size at the figure calibration

Use a=-2, beta=-1, d=0.7, c=1, and **gamma=1**. Solve both FOCs separately for each history. `Change = new - old`; the last column is `100*(new-old)/abs(old)`, so its sign indicates the direction of the numerical correction. It is undefined when the old value is zero.

| Derivative | History | Previous | Corrected | Change | Change / abs(previous) |
| --- | --- | ---: | ---: | ---: | ---: |
| Alpha price derivative | `(0, 0)` | -0.49878673 | 0.25427446 | +0.75306119 | +150.98% |
| Alpha price derivative | `(1, 0)` | -0.49235357 | 0.27588838 | +0.76824195 | +156.03% |
| Alpha price derivative | `(0, 1)` | -0.50380407 | 0.23988635 | +0.74369042 | +147.62% |
| Profit gamma derivative | `(0, 0)` | 0 | 0 | +0 | — (old = 0) |
| Profit gamma derivative | `(1, 0)` | 0.022924683 | 0.022924683 | +0 | +0.00% |
| Profit gamma derivative | `(0, 1)` | 0.015864261 | 0.00028947178 | -0.015574789 | -98.18% |
| Mixed beta–gamma price derivative | `(0, 0)` | 0 | 0 | +0 | — (old = 0) |
| Mixed beta–gamma price derivative | `(1, 0)` | 0.00066983619 | 0.021128878 | +0.020459042 | +3054.34% |
| Mixed beta–gamma price derivative | `(0, 1)` | -1.9488021e-05 | -0.01390703 | -0.013887542 | -71261.94% |

## 6. Size across the validation sample

The sample contains 27 solutions: the three histories, gamma in {0, 0.7, 1.8}, and parameter tuples `(alpha, beta, delta, c)` equal to `(-2,-1,0.7,1)`, `(-1,0.2,0.4,1)`, and `(-3,1,0.9,0.5)`. These ranges describe this sample, not global bounds. Percentage ranges omit points where the previous derivative is zero and use absolute percentage changes.

| Derivative | Absolute change range | Absolute percentage change range | Strict sign reversals |
| --- | ---: | ---: | ---: |
| Alpha price derivative | 0.500221–2.5706 | 114.87%–366.70% | 27/27 |
| Profit gamma derivative | 0–0.123279 | 0.00%–98.21% | 0/27 |
| Mixed beta–gamma price derivative | 0–0.132439 | 177.61%–249711.60% | 0/27 |

## 7. What did not change, and how the corrections were checked

The existing formulas for dp1/dgamma, dp2/dgamma, dp1/dbeta, dp1/ddelta, and d(p1/p2)/dgamma were retained. In particular, the ratio derivative still has signs zero, positive, and negative for histories (0,0), (1,0), and (0,1), respectively. The follow-up added assertions for these results.

Run `python3 run_all.py` to reproduce the validation. The corrected price derivatives satisfy the differentiated pricing equations symbolically. The profit formula satisfies the envelope identity. At all 27 sample solutions, central finite differences re-solve both prices after parameter perturbations and agree with the corrected derivatives, including the mixed derivative obtained by finite-differencing dp1/dbeta in gamma. The runner also checks FOC residuals, feasibility, and negative definite joint profit Hessians. These checks passed.

The figure generator also changed method: it now re-solves the FOCs at every gamma rather than integrating slopes while holding shares fixed. Consequently, differences between old and new figure curves include this plotting-method correction as well as the profit derivative correction. The derivative tables above deliberately avoid that confounding by comparing formulas at identical solved prices and shares.
