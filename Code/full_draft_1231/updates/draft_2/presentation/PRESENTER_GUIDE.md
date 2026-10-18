# Presenter guide

Six section transitions show the roadmap. Theme: UGA red (#BA0C2F), black, and white. Imported figures retain their source colors.

## 1. Preferences for Product Variety in Targeted Pricing

Introduce the research question: what can purchase histories tell firms about preferences for switching?

## 2. Roadmap

Briefly preview the six sections; demand estimation is the central empirical exercise.

## 3. Roadmap: Motivation and contribution

Locate the section in the overall argument.

## 4. Can firms price the desire for a change?

Use the example of someone who likes berry yogurt but wants something different after repeated purchases. Distinguish a stable taste from fatigue.

## 5. Two exercises, two takeaways

Explain upfront that the empirical demand model and toy supply model are separate exercises.

## 6. Contribution: variety across attributes

Position relative to attribute satiation in McAlister and Lattin, brand dynamics in Chintagunta, and personalized pricing in Dube and Misra. References are in the backup slides.

## 7. Roadmap: Data and descriptive evidence

Locate the section in the overall argument.

## 8. Why yogurt, and whose purchases?

Single-person households help with interpretation but do not eliminate measurement issues or purchases for others.

## 9. The estimation sample contains 998 households

Report these as the draft table statistics. Do not infer a trip frequency from the eight-week observation description; the aggregation needs reconciliation before doing so.

## 10. Three flavor groups simplify a rich product space

Emphasize that changing from vanilla to peach is invisible in the other category. This can suppress measured variety seeking.

## 11. Many households concentrate on one flavor group

The draft reports that over 70 percent are nearly loyal to one group. Concentration is at the aggregate flavor level, not necessarily at the individual flavor or brand level.

## 12. Other flavors dominate across purchase volumes

Other is around 60 percent across volume groups. Discuss how the uneven breadth of the categories affects interpretation.

## 13. Roadmap: Monte Carlo motivation

Locate the section in the overall argument.

## 14. Monte Carlo: simulation procedure

The procedure summarizes the simulation described in the manuscript. Its distance-based mechanism differs from the empirical flavor-fatigue index. Simulation code is not re-estimated here.

## 15. Monte Carlo: simulated choice probabilities

Walk through product 1 and product 5, then the disappearance of the middle products. Values are taken directly from the source table.

## 16. Roadmap: Demand estimation and results

Locate the section in the overall argument.

## 17. Demand combines stable tastes and purchase history

The next slides define the full manuscript-style history index and show its update. Distinguish the intended fatigue motivation from the cross-flavor construction. Demand history coefficients and supply switching rewards have different sign conventions.

## 18. Defining the consumption-history index $\Xi_{ijt}$

This makes the manuscript recursion causal by summing only over observed past periods rather than the printed fixed upper limit of eight. It matches the history function in est_step2.py: past index values, cross-flavor purchase shares, equal rho weights, and zero initialization. This is the earlier implementation consistent with the manuscript form, not the newer own-flavor stock used in est_step4.py. Exact provenance of the reported tables has not been established.

## 19. How $\Xi_{ijt}$ changes: a two-period example

Work through the first update, then the memory term. The distinction matters: a negative coefficient on this cross-flavor index cannot alone be interpreted as disutility from repeatedly consuming the same flavor. The prior slides describe the manuscript motivation; this slide spells out what its stated index actually measures.

## 20. What separates tastes from fatigue?

A switching pattern alone does not identify fatigue. Discuss initial conditions, serially correlated shocks, and the amount of within-household variation required.

## 21. Demand estimation: first-stage price control function

This is the Petrin--Train control-function approach used in the draft. First-stage coefficients are written explicitly. The draft does not provide instrument strength diagnostics and is inconsistent about the number of other markets.

## 22. Demand estimation: complete utility specification

This expands the estimation formula using the flavor-specific parameters actually reported in the draft tables. The preceding slides define the manuscript-style history recursion using only past observed periods and explain its cross-flavor construction. The reference category normalizations match the reported tables.

## 23. Demand estimation: panel likelihood and mixed logit

Show that persistent household heterogeneity requires integrating the product over a household history, rather than multiplying separately integrated marginal probabilities. This is the full panel mixed-logit estimation target; the manuscript does not document the draw count or coefficient distribution, so neither is asserted here.

## 24. Demand estimation: four specifications

The distribution and numerical integration details should be checked against the estimation implementation before presenting this likelihood as a verified account of the code. National common demand shocks can threaten the price instrument. Results follow immediately.

## 25. Baseline demand estimates barely move

Flavor coefficients are relative to other; the intercept locates inside utility relative to the outside option. The largest percentage change is in magnitude. These estimates are reported as significant in the source tables.

## 26. Berry fatigue is negative; plain is imprecise

A negative coefficient means history lowers utility under the fatigue index interpretation. Do not confuse a significant mean with precise identification of its heterogeneity.

## 27. Heterogeneity exposes an estimation problem

These values contradict the prose claim of little taste heterogeneity. The repeated standard deviations and large berry value should be audited against estimation code. Treat them as reported estimates, not validated structural distributions.

## 28. Dollar interpretations inherit parameter uncertainty

These are magnitudes, not positive willingness to pay for a disliked flavor. The fatigue ratio is per unit of the history index, not automatically the discount needed after one repeated purchase. Avoid treating the large mixed-logit ratio as a precise policy target.

## 29. Roadmap: Pricing and learning

Locate the section in the overall argument.

## 30. From empirical demand to a pricing mechanism

Make the separation explicit. Supply uses a positive reward for switching, whereas demand uses a negative coefficient on a fatigue stock.

## 31. A retailer prices two switching types

Use s for a probability or share, not a sum of probabilities over consumers. Population size multiplies per-consumer profits.

## 32. Prices affect both current and future profits

Here j includes all first-period outcomes, including no purchase, with zero outside margin and an explicitly defined continuation state. This compact objective avoids reproducing the draft long first-order conditions, whose outside-option treatment is ambiguous.

## 33. Without type information, purchases update beliefs

A posterior above the prior conditional on a purchase is not inherently biased or an overestimate. Bayes plausibility requires averaging over all outcomes, including no purchase. The simulation table reports only the inside-good posteriors.

## 34. The information constraint changes the pricing problem

This objective expresses the mechanism without relying on the source algebra for the learning derivative. The sign of a price response is not universal; focus on the reported simulation. Include outside histories when implementing the objective.

## 35. Simulation parameters are illustrative

No empirical calibration is claimed. The source does not fully document the profit denominator or the initial-history implementation; preserve the table values and label the normalization cautiously.

## 36. The profit gap peaks in the mixed population

Chart is redrawn from the value-of-information column, not recalculated from rounded profit columns. Retain the reported normalization; the draft calls it per unit but the objective is written per consumer over two periods. This requires clarification before quantitative policy use.

## 37. Pricing exercise: full results

Prices shown are first-period limited-information prices. Lower posted price alone does not establish a larger discount without a common reference price. Conditional posteriors above priors should not be described as systematic overestimation.

## 38. Roadmap: Conclusions and discussion

Locate the section in the overall argument.

## 39. What does this say about privacy policy?

Distinguish the illustrated cost of an information constraint from a causal estimate of an actual regulation. A policy conclusion needs consumer welfare, collection costs, and an explicit policy mapping.

## 40. Conclusions and next steps

Return to the research question. Stable baseline coefficients are a finding about this specification, not evidence that variety never matters.

## Source and editorial choices

Base: `../draft_1010.tex`. The original manuscript is unchanged. Figures are copied into `assets/` so the deck can be moved as a folder. Tables are shortened and the information-value chart is redrawn directly from reported values.

The slides favor source tables over conflicting prose: the $7.35 fatigue magnitude belongs to berry; the Monte Carlo comparison grid ends at gamma=9; and the random coefficient estimates do not support a blanket claim of little taste heterogeneity. Conditional purchase posteriors are not labeled biased estimates of the population share.

Demand probabilities use deterministic utility after integrating extreme-value shocks. Supply is presented through compact objectives rather than reproducing ambiguous first-order conditions. Outside-option histories must be accounted for in a full implementation. The history index is explicitly defined using a past-only version of the manuscript recursion, cross-flavor purchase shares, and zero initialization, matching the earlier `../../est_step2.py` history function. The newer `../../est_step4.py` uses a different own-flavor stock. Exact provenance of the reported tables remains unverified.

Before quantitative policy use, reconcile observation/trip aggregation, the number of markets used in the instrument, initial conditions/history construction, supply optimization details, and the denominator of reported profits. No new empirical estimates or welfare results are introduced.

## Build

Run from this folder:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error variety_pricing_slides.tex
```

Speaker notes are embedded in the TeX and hidden in the audience PDF. Add `\setbeameroption{show notes}` in the preamble to include them.


## Estimation detail added in this revision

The deck writes the full panel mixed-logit target, integrating a household’s product of conditional probabilities over persistent random coefficients. This expands the paper’s abbreviated likelihood exposition. Distributional assumptions, simulation draw count, and initial-condition implementation are not supplied in the manuscript and are not asserted in the slides. The numerical results remain those reported in the source tables.


The pricing results slide now includes the exact full-paper table body in `assets/pricing_exercise_table.tex`, preserving the paper’s population labels, prices, posterior beliefs, profits, and numeric precision, with the redundant percentage-high and percentage-low columns removed. The abbreviated backup profit table has been removed.
