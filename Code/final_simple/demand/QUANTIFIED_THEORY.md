# Current presentation exercise: quantify the original theory

Only compare common and informed prices and their two-period revenues. Run:

```bash
python3 Code/final_simple/demand/quantified_theory.py \
  --estimates estimates_1111/three/no_control_fit/point_estimates.csv \
  --output Code/final_draft/seminar_pres/assets/quantified_theory
```

This reads aggregate parameter estimates only and can run locally. It runs plain,
berry and other separately. The slides use plain.

The utility is the original theory's `beta_f + alpha*p + gamma*history_sign`.
`beta_f` is the estimated inside constant plus the flavor coefficient; other is
the reference flavor. `alpha` is minus the population mean price-sensitivity
magnitude, `-exp(mu + sd^2/2)`. Gamma follows the estimated population normal,
represented by nine Gauss-Hermite nodes. The same gamma persists in both periods.
Habit is held at its zero-stock reference; outside utility is normalized to zero.
These are explicit reference choices for quantifying the theory, not an exact
aggregation of fitted multinomial demand.

The informed firm knows gamma. The common-price firm uses its distribution,
with fixed beliefs in both terminal histories, exactly as in the theory code.
Both adapt second prices to focal versus not focal. Revenue comparisons evaluate
the two policies on the same persistent types. Delta is one and cost is zero.

The fitted-distribution baseline is accompanied by low-, middle-, and high-heavy
sensitivity scenarios. Display groups use the normal quartile cutoffs; each
favored group receives mass one half and the remaining groups one quarter each.
These sensitivity weights are scenarios, not estimated probabilities.

No observed-price comparison, household context reconstruction, dynamic habit,
price-moment calibration, demand refit, or server job is needed for this exercise.
The previous server runners and long binary walkthrough remain historical work;
they do not generate the current presentation exercise.
