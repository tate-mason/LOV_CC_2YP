# Current presentation exercise: quantify the original theory

Compare common and informed prices, two-period revenues, consumer surplus,
total welfare, and expected binary purchases. Run:

```bash
python3 Code/final_simple/demand/quantified_theory.py \
  --estimates fixed_est/refined_128/point_estimates.csv \
  --output Code/final_draft/seminar_pres/assets/quantified_theory_observed_128
```

This reads aggregate parameter estimates only and can run locally. It runs plain,
berry and other separately. The slides use plain.

The current inputs are the corrected 128-draw observed-shopping-week fit,
including genuine zero-yogurt-purchase weeks. The original November 11 outputs
are preserved separately. Run
`python3 Code/final_draft/seminar_pres/refresh_observed_week_assets.py` afterward
to generate the presentation's demand tables, pricing summary tables and macros.
Then run `python3 Code/final_draft/seminar_pres/rebuild_price_scales.py` to apply
price-axis limits shared within each figure in the presentation. Separate
figures choose their own ranges.

The utility is the original theory's `beta_f + alpha*p + gamma*history_sign`.
`beta_f` is the estimated inside constant plus the flavor coefficient; other is
the reference flavor. `alpha` is minus the population mean price-sensitivity
magnitude, `-exp(mu + sd^2/2)`. Gamma follows the estimated population normal,
represented by nine Gauss-Hermite nodes. The same gamma persists in both periods.
Habit is held at its zero-stock reference. The relative choice index subtracts
the outside utility; welfare restores the theory's outside switching utility.
These are explicit reference choices for quantifying the theory, not an exact
aggregation of fitted multinomial demand.

The informed firm knows gamma. The common-price firm uses its distribution,
with fixed beliefs in both terminal histories, exactly as in the theory code.
Both adapt second prices to focal versus not focal. Revenue comparisons evaluate
the two policies on the same persistent types. Delta is one and cost is zero.

All empirical comparisons integrate over the fitted gamma distribution with its
estimated weights. No low/middle/high probability scenarios are imposed.
Low, middle and high are display groups for price paths only; grouping does not
change the distribution used to optimize uniform prices or aggregate outcomes.
The alternative prior scenarios belong only to the pure-theory exercise.

No observed-price comparison, household context reconstruction, dynamic habit,
price-moment calibration, demand refit, or server job is needed for this exercise.
The previous server runners and long binary walkthrough remain historical work;
they do not generate the current presentation exercise.

## Welfare accounting

Use the original slide utilities with signed price coefficient `alpha < 0`:
`Vj = beta_f + alpha*p + gamma*1{h=0}` and `V0 = gamma*1{h=j}`.
For each type, history and offered price, expected one-period consumer surplus
in dollars is `log(exp(Vj)+exp(V0))/(-alpha)`, up to a common taste-shock
constant. The constant cancels in policy differences. This is the standard
logit expected maximum utility calculation (Train, chapter 3,
https://eml.berkeley.edu/choice2/ch3.pdf).

Consumers make myopic choices, as in the pricing model. We sum their expected
experienced utility across periods using the same discount factor as the firm:
`CS = CS1(h0) + delta*[s1*CS2(j)+(1-s1)*CS2(0)]`. Each policy has its own `s1`
and thus its own future-history probabilities; the same persistent gamma is
used on both branches. Integrate over types only after computing these outcomes.
This is experienced welfare under myopia, not a forward-looking consumer model.

`W = CS + profit`; profit equals revenue at maintained zero marginal cost.
`Q = s1 + delta*[s1*s2(j)+(1-s1)*s2(0)]` counts expected binary purchases, not
package volume. Undiscounted purchases are also exported. Changes always mean
informed minus common, with the same type distribution and initial history.

Do not drop `V0` just because it cancels from within-history choice probabilities:
pricing changes the distribution of future histories, so it does not cancel
from two-period welfare. For transparency, `zero_outside_*` columns report the
alternative cardinal assumption `Vj=v, V0=0`; it leaves quantities/prices
unchanged but changes welfare. The main figures retain the original utilities.
These utility levels are a maintained theory assumption, not separately
identified from relative choice probabilities.

Outputs per flavor: `welfare_comparison.csv` (levels), `welfare_changes.csv`
(changes and type-weighted CS/welfare winner shares), `welfare_by_group.csv`,
`welfare_table.tex`, and `welfare_changes.pdf/png`. `type_results.csv` contains
theoretical quadrature support points, not household records. No standard errors
are attached: these are plug-in results conditional on the estimated parameters.

Recover the matching pure-theory welfare figures from its saved policies:

```bash
python3 Code/final_simple/demand/welfare_reporting.py \
  --theory-output Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty
python3 Code/final_simple/checks/test_binary_welfare.py
```

Third-degree discrimination raises optimized informed profit relative to the
common policy; the signs of consumer-surplus and total-welfare changes are
calculated rather than imposed. The common policy retains the existing
fixed-belief restriction, so its comparison also inherits that restriction.

## First-period comparative statics

For the slides' positive coefficient convention `a=-alpha>0`, write
`q(0)=1`, `q(j)=-1` and `D=pi2(j)-pi2(0)`. The terminal envelope theorem gives
`D' = -(s2(j)+s2(0))/a`. Differentiating the *implicit* first-period price
condition, including this continuation effect, gives

`dp1*/dgamma = [q(h0)*s1 + delta*(1-s1)*(s2(j)+s2(0))]/a`.

The profit envelope derivative is
`dPi*/dgamma = [q(h0)*s1 - delta*s1*s2(j) + delta*(1-s1)*s2(0)]/a`.
For initially outside consumers both derivatives are positive when
`0 <= delta <= 1`. For initially inside consumers both signs are ambiguous.
The checks compare these formulas against numerical derivatives over different
intercepts and novelty values and verify both possible inside signs.
