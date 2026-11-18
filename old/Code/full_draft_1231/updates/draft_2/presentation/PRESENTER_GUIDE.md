# Presenter guide

The active model freely optimizes prices in both periods with observed random initial histories. All former coupon restrictions and no-initial-history claims have been removed from the active pricing slides. The empirical section retains its source-data caveats.

## Pricing slides

1. Title: dynamic monopoly pricing with love of variety.
2. Introduction: distinguish stable tastes and switching preferences.
3. Motivation: the model information gain is an illustrative firm-profit comparison.
4. Consumer problem: define types, products, periods, initial state, choices, prices, switching utility, shocks, and logit shares. Initial states j, k, and 0 are equally likely and independent of type. Initial 0 has no switching reward. Consumers are myopic; choosing outside in period 1 exits permanently.
5. Producer problem: define the complete menu P, cost c, initial-state probability mu, type probability q, firm discount factor delta, terminal profit w, and total type-conditional value pi. Period-two prices can use both initial state and first purchase. Initial states are observed. Common menus cannot directly condition on type but account for endogenous selection into observed histories.
6. Known-type optimal prices: solve backward from period 2. Terminal product margins equal -1/(alpha times terminal outside share). In period 1, add discounted continuation profit to the current margin before applying the corresponding markup condition. Define all terms explicitly. A button links to the unobserved-type conditions in the appendix, with a return button. This slide precedes the free-pricing calibration slide.
7. Free pricing: the calibration retains beta=(2.2,2), costs=(1.2,0.8), alpha=-1.5, and delta=.95. Both prices in both periods are free to rise or fall. Known-type interior terminal margins are equal across products, so the former higher-margin coupon claim does not apply.
8. Full prices: one table with shared policy headers and two panels. Period 1 shows all three initial-state period-one price pairs. Period 2 shows all six full-history period-two price pairs. All three policies are shown; no terminal histories are omitted. Prices are transaction prices. Common 50/50 means common across types, not across products or histories.
9. Period results: show period-one profit, discounted period-two profit, and total value for each type under common and observed-type prices. Every profit is per arriving consumer; terminal profit already includes the probability of reaching period 2.
10. Population profits: optimize a common menu at every q. Observed-type profit averages the two separately optimized type values. The difference measures the value of direct type information in this model.
11. Fixed menus: compare three menus held fixed as q changes. Each generates an affine line. Freely reoptimizing prices weakly dominates this restricted set of choices.
12. Mispricing: hold true type fixed in each panel while varying the firm's assumed probability q=Pr(H). Both x-axes use q. Correct certainty is q=1 for H and q=0 for L. Expected two-period firm profit is expected period-one profit plus delta times expected period-two profit, averaging over initial states and choice shocks, including permanent initial exit. This is the meaning of customer value on the graphs.
13. Who is most valuable: fix gamma_L=1 and q=.5; vary gamma_H from 1.5 to 10 in steps of 0.5 and reoptimize the common menu. Show common-menu type values, known-type values, current profit, and discounted harvesting separately. In the simulated range H value increases and exceeds L for gamma_H>gamma_L. L's common-menu value can change despite fixed gamma_L, because prices change.
14. Induction and harvesting: known-type benchmark comparing optimized static current profit with dynamic current profit and discounted terminal profit. The difference between static and dynamic current earnings measures profit sacrificed for continuation. It does not identify a coupon effect.

## Empirical slides

### 15. Data

Single-person households help with interpretation but do not eliminate measurement issues or purchases for others.

### 16. Sample

Report these as the draft table statistics. Do not infer a trip frequency from the eight-week observation description; the aggregation needs reconciliation before doing so.

### 17. Flavor groups

Emphasize that changing from vanilla to peach is invisible in the other category. This can suppress measured variety seeking.

### 18. Flavor concentration

The draft reports that over 70 percent are nearly loyal to one group. Concentration is at the aggregate flavor level, not necessarily at the individual flavor or brand level.

### 19. Flavor shares

Other is around 60 percent across volume groups. Discuss how the uneven breadth of the categories affects interpretation.

### 20. Demand model

The next slides define the full manuscript-style history index and show its update. Distinguish the intended fatigue motivation from the cross-flavor construction. Demand history coefficients and supply switching rewards have different sign conventions.

### 21. Identification

A switching pattern alone does not identify fatigue. Discuss initial conditions, serially correlated shocks, and the amount of within-household variation required.

### 22. Price control function

This is the Petrin--Train control-function approach used in the draft. First-stage coefficients are written explicitly. The draft does not provide instrument strength diagnostics and is inconsistent about the number of other markets.

### 23. Specifications

The distribution and numerical integration details should be checked against the estimation implementation before presenting this likelihood as a verified account of the code. National common demand shocks can threaten the price instrument. Results follow immediately.

### 24. Demand estimates

Flavor coefficients are relative to other; the intercept locates inside utility relative to the outside option. The largest percentage change is in magnitude. These estimates are reported as significant in the source tables.

### 25. LOV coefficients

A negative coefficient means history lowers utility under the fatigue index interpretation. Do not confuse a significant mean with precise identification of its heterogeneity.

### 26. Heterogeneity

These values contradict the prose claim of little taste heterogeneity. The repeated standard deviations and large berry value should be audited against estimation code. Treat them as reported estimates, not validated structural distributions.

### 27. Implied WTP

These are magnitudes, not positive willingness to pay for a disliked flavor. The fatigue ratio is per unit of the history index, not automatically the discount needed after one repeated purchase. Avoid treating the large mixed-logit ratio as a precise policy target.

### 28. Conclusion

Distinguish the illustrated cost of an information constraint from a causal estimate of an actual regulation. A policy conclusion needs consumer welfare, collection costs, and an explicit policy mapping.

## Pricing appendix

The linked unobserved-type slide defines type weights omega, current margin m, first-period effective margin M, and initial-state conditional value v. Its terminal condition weights each type by its probability of reaching the first purchase; the first-period condition includes continuation value.

The parameters/timing slide states the initial-history and pricing information assumptions. Known-type first-order conditions follow backward from terminal equal margins; the first-period effective margin includes continuation value. They cannot be applied to pooled shares under heterogeneous types. The true-gamma comparison uses fixed benchmark menus and separately optimized known-type menus. Mispricing losses compare each true type with its own optimum, and both axes use Pr(H).

The simulation and empirical appendix material retains its original source limitations. No empirical parameters are re-estimated in this task.

## Sources and rebuilding

Active code: `../../../toy_model.py`. Derivation: `../../toy_model/toy_model.tex`. Current data: `../../toy_model/results/`, especially `gamma_H_sweep.csv`, `value_distribution.csv`, and `prices_50_50.csv`. Former coupon/newborn files in the results folder are historical and not current deck inputs. The older fixed-price and no-initial-state unrestricted exercises are retained in their existing historical directories.

From the repository root:

```sh
.venv/bin/python toy_model.py
MPLCONFIGDIR=/tmp/toy-mpl .venv/bin/python updates/draft_2/presentation/build_pricing_figures.py
.venv/bin/python -m unittest test_toy_model -v
latexmk -pdf -cd updates/draft_2/presentation/variety_pricing_slides.tex
latexmk -pdf -cd updates/toy_model/toy_model.tex
```

The figure builder generates numerical macros, the complete two-panel price table, and period-profit results from the active model. The known-type solution uses backward interior equations; heterogeneous common-price optimization uses three unconstrained BFGS starts per initial state and analytic gradients. Tests include independent derivative-free optimization, finite-difference gradients, profit and survival accounting, initial switching, known-type price conditions, and information dominance.
