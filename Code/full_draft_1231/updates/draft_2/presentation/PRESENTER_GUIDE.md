# Presenter guide

The main presentation contains 24 slides. Supply simulation parameters, the demand simulation, utility specification, and panel likelihood are in the appendix. Buttons on Supply results, Demand model, and Specifications open the relevant backup slides; each has a return button. Repeated section roadmaps have been removed. Slide wording is unchanged.

## Main presentation

### 1. Dynamic Optimal Pricing with Love of Variety

Introduce the research question: what can purchase histories tell firms about preferences for switching?

### 2. Introduction

Use the example of someone who likes berry yogurt but wants something different after repeated purchases. Distinguish a stable taste from fatigue.

### 3. Motivation and Contribution

Explain upfront that the empirical demand model and toy supply model are separate exercises.

Position relative to attribute satiation in McAlister and Lattin, brand dynamics in Chintagunta, and personalized pricing in Dube and Misra. References are in the backup slides.

### 4. Supply exercise

Introduce the illustrative supply exercise before turning to demand. Supply uses a positive reward for switching, whereas demand uses a negative coefficient on a fatigue stock.

### 5. Supply model

Explain the equation from the consumer's perspective: baseline product appeal, a reward for choosing a different product from last period, and a negative price effect. V is deterministic current-period utility, not a forward-looking consumer Bellman value. Adding independent type-I extreme-value shocks gives logit choice probabilities including an outside option. The retailer uses these probabilities in its pricing problem. Use s for a probability or share, not a sum of probabilities over consumers. Population size multiplies per-consumer profits.

### 6. Full information

Here j includes all first-period outcomes, including no purchase, with zero outside margin and an explicitly defined continuation state. This compact objective avoids reproducing the draft long first-order conditions, whose outside-option treatment is ambiguous.

### 7. Learning

A posterior above the prior conditional on a purchase is not inherently biased or an overestimate. Bayes plausibility requires averaging over all outcomes, including no purchase. The simulation table reports only the inside-good posteriors.

### 8. Limited information

This objective expresses the mechanism without relying on the source algebra for the learning derivative. The sign of a price response is not universal; focus on the reported simulation. Include outside histories when implementing the objective.

### 9. Supply results

Chart is redrawn from the value-of-information column, not recalculated from rounded profit columns. Retain the reported normalization; the draft calls it per unit but the objective is written per consumer over two periods. This requires clarification before quantitative policy use.}
\par\smallskip
\hyperlink{supply-parameters}{\beamerbutton{Simulation parameters}

### 10. Pricing results

Table from the full paper with the redundant percentage-high and percentage-low columns removed; population mixes, prices, posterior beliefs, profits, and reported information values are retained.

### 11. Data

Single-person households help with interpretation but do not eliminate measurement issues or purchases for others.

### 12. Sample

Report these as the draft table statistics. Do not infer a trip frequency from the eight-week observation description; the aggregation needs reconciliation before doing so.

### 13. Flavor groups

Emphasize that changing from vanilla to peach is invisible in the other category. This can suppress measured variety seeking.

### 14. Flavor concentration

The draft reports that over 70 percent are nearly loyal to one group. Concentration is at the aggregate flavor level, not necessarily at the individual flavor or brand level.

### 15. Flavor shares

Other is around 60 percent across volume groups. Discuss how the uneven breadth of the categories affects interpretation.

### 16. Demand model

The next slides define the full manuscript-style history index and show its update. Distinguish the intended fatigue motivation from the cross-flavor construction. Demand history coefficients and supply switching rewards have different sign conventions.}
\par\smallskip
\hyperlink{simulation-procedure}{\beamerbutton{Simulation procedure}}\quad
\hyperlink{simulation-results}{\beamerbutton{Simulation results}}\quad
\hyperlink{utility-specification}{\beamerbutton{Utility specification}

### 17. Identification

A switching pattern alone does not identify fatigue. Discuss initial conditions, serially correlated shocks, and the amount of within-household variation required.

### 18. Price control function

This is the Petrin--Train control-function approach used in the draft. First-stage coefficients are written explicitly. The draft does not provide instrument strength diagnostics and is inconsistent about the number of other markets.

### 19. Specifications

The distribution and numerical integration details should be checked against the estimation implementation before presenting this likelihood as a verified account of the code. National common demand shocks can threaten the price instrument. Results follow immediately.}
\par\smallskip
\hyperlink{panel-likelihood}{\beamerbutton{Panel likelihood}

### 20. Demand estimates

Flavor coefficients are relative to other; the intercept locates inside utility relative to the outside option. The largest percentage change is in magnitude. These estimates are reported as significant in the source tables.

### 21. LOV coefficients

A negative coefficient means history lowers utility under the fatigue index interpretation. Do not confuse a significant mean with precise identification of its heterogeneity.

### 22. Heterogeneity

These values contradict the prose claim of little taste heterogeneity. The repeated standard deviations and large berry value should be audited against estimation code. Treat them as reported estimates, not validated structural distributions.

### 23. Implied WTP

These are magnitudes, not positive willingness to pay for a disliked flavor. The fatigue ratio is per unit of the history index, not automatically the discount needed after one repeated purchase. Avoid treating the large mixed-logit ratio as a precise policy target.

### 24. Policy discussion

Distinguish the illustrated cost of an information constraint from a causal estimate of an actual regulation. A policy conclusion needs consumer welfare, collection costs, and an explicit policy mapping.

## Appendix

### 1. Simulation parameters

No empirical calibration is claimed. The source does not fully document the profit denominator or the initial-history implementation; preserve the table values and label the normalization cautiously.}
\par\smallskip
\hyperlink{supply-results}{\beamerreturnbutton{Supply results}

### 2. Simulation procedure

The procedure summarizes the simulation described in the manuscript. Its distance-based mechanism differs from the empirical flavor-fatigue index. Simulation code is not re-estimated here.}
\par\smallskip
\hyperlink{simulation-results}{\beamerbutton{Simulation results}}\quad \hyperlink{demand-model}{\beamerreturnbutton{Demand model}

### 3. Simulation results

Walk through product 1 and product 5, then the disappearance of the middle products. Values are taken directly from the source table.}
\par\smallskip
\hyperlink{simulation-procedure}{\beamerbutton{Simulation procedure}}\quad \hyperlink{demand-model}{\beamerreturnbutton{Demand model}

### 4. Utility specification

This expands the estimation formula using the flavor-specific parameters actually reported in the draft tables. The preceding slides define the manuscript-style history recursion using only past observed periods and explain its cross-flavor construction. The reference category normalizations match the reported tables.}
\par\smallskip
\hyperlink{demand-model}{\beamerreturnbutton{Demand model}

### 5. Panel likelihood

Show that persistent household heterogeneity requires integrating the product over a household history, rather than multiplying separately integrated marginal probabilities. This is the full panel mixed-logit estimation target; the manuscript does not document the draw count or coefficient distribution, so neither is asserted here.}
\par\smallskip
\hyperlink{specifications}{\beamerreturnbutton{Specifications}

### 6. Discussion

Optional discussion prompts.

### 7. Flavor shares over time

Present the supporting figure or reference.

### 8. Marginal distributions

Present the supporting figure or reference.

### 9. Joint distributions

Present the supporting figure or reference.

### 10. References

Present the supporting figure or reference.

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
