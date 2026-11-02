The current estimator changes the economic specification, the purchase sample,
and the construction of prices and instruments relative to
`full_draft_1231/est_step4.py`. The current results are from
`shell_files/res_hess`. Step 4 is the main historical comparison because it was
the last full estimator in that folder; some problems in the earlier step-2
estimator had already been corrected by step 4.

Both versions aggregate purchases across stores into household-week quantities,
use zero initial histories, and integrate persistent household types over the
entire household purchase history. Both current capacity-7 results and step 4
construct outside quantity as max(7 minus inside quantity purchased, 0), retaining
inside quantities above seven. Neither inserts outside-only observations for
unobserved purchase weeks. These are retained features, rather than new sources
of improvement.

The biggest economic change is the history variable. Step 4 used a depreciating
own-flavor stock, with depreciation 0.9 per calendar week, and entered its
centered log transformation into utility. Each flavor had a separate mean
history coefficient and, in the full model, a separate random history
coefficient. The current estimator instead constructs

    S_ijt = sum of own-flavor purchase shares over all earlier observed weeks
    N_ijt = sum of other-flavor shares in the immediately preceding calendar week
    X_ijt = 0.3 N_ijt - 0.7 S_ijt.

There is no depreciation, logarithm, or global centering. A purchase gap sets the
recent-other-flavor component to zero but does not erase accumulated own-flavor
history. The coefficient gamma_i is common across flavors within a household.
Positive gamma now combines a penalty for accumulated own-flavor purchases with
a reward for alternatives different from the preceding week's purchases;
negative gamma reverses those effects. This changes the sign interpretation:
a negative old coefficient on own stock represented satiation, while a negative
current gamma represents reinforcement or familiarity in this specified index.
Old and new gamma magnitudes cannot be compared directly because the regressor,
scale, and coefficient restrictions all changed. The 0.7 tradeoff is fixed, so
the two history effects are not separately estimated.

The second major change is where heterogeneity enters utility. Step 4's full
model had random berry and plain intercepts and three random history slopes,
with a common price coefficient: five random dimensions and 13 demand
parameters. The current model fixes flavor intercepts across households and
uses two random dimensions: a positive lognormal price sensitivity and a normal
history coefficient shared across flavors. With the common inside intercept
needed for capacity 7, there are eight demand parameters. In the paper's sign
convention,

    V_ijt = b_0 + beta_j - alpha_i p_ijt + gamma_i X_ijt + rho vhat_ijt,
    log(alpha_i) ~ N(mu_alpha, sigma_alpha^2),
    gamma_i ~ N(mu_gamma, sigma_gamma^2).

The two type components are independent. This reduces the dimension of the
simulation and removes the old competition between random flavor intercepts and
random flavor-specific history slopes. It also places a stronger restriction
on persistent flavor preferences. The clearer estimates therefore reflect a
more parsimonious model as well as more data; they are not evidence that
persistent unobserved flavor tastes have been fully controlled for.

Sample construction also changed substantially. Both versions restrict demand
to single-person household purchase records. Step 4 additionally restricted
serving codes and selected product modules from its original merged file. The
current estimator reads the prepared matched-price yogurt purchase table for
2006–2015, uses resolved saved flavor labels, and does not reapply the old
serving-code restriction. It requires at least two usable purchase weeks after
menu and instrument exclusions. The documented old production run contained
550 households and 3,452 household-weeks; res_hess contains 15,105 households
and 281,695 household-weeks, averaging about 18.6 weeks per household. This is
a much larger and longer sample, but the increase cannot be assigned to any
one processing change without a sample audit. Single-person selection itself
is a restored continuity with the old design, not a newly introduced restriction.

The price exposure construction now uses distinct observed store-week-UPC quotes
from the prepared data. Within a household-week, the exposure store is the one
with the most quantity purchased, with deterministic store-code tie breaking.
Flavor prices average distinct covered UPC quotes at that store in that week.
Step 4 instead used transaction-based prices and filled missing store-week
flavor prices with week-flavor and then overall-flavor averages. The current
model does not use those broad fallbacks. A flavor lacking a valid local price
is excluded from the menu; a week is excluded if a purchased flavor lacks price
coverage. Interpreting price coverage as availability remains an assumption.
All-household purchase records remain available for constructing price menus
and instruments even though demand records are restricted to size one.

The control function has also been redesigned. Step 4 used a same-UPC,
same-week other-market instrument and a transaction-level regression containing
a numeric brand code and week effects. Residuals were standardized and missing
values could be filled with zero. The current instrument is a same-week mean
of distinct store-week-UPC quotes outside the consumer's DMA, pooled across
flavors. The same instrument enters three separate flavor price equations.
Each equation controls for all three history indices and flavor-specific week
and consumer-DMA effects. Residuals enter demand in price units, without
standardization. First-stage diagnostics use household-clustered uncertainty.
In res_hess, instrument F statistics are about 1,288 for other, 578 for berry,
and 86 for plain; partial R-squared values are 0.0213, 0.0192, and 0.0076.
These establish relevance in the specified first stages, not the exclusion
restriction or exogeneity of purchase history.

Computationally, both versions use simulated maximum likelihood with fixed
antithetic normal integration draws and a type that persists across household
weeks. The current code supplies an analytic demand gradient and averages the
negative log likelihood over households for optimization. The saved current
run used 32 draws and one start, compared with 128 draws and three starts in
the saved step-4 settings. The current run converged with no parameters at
bounds. That is encouraging, but a smaller draw count and one start do not by
themselves demonstrate greater numerical stability. Step 4's recorded profile
and draw checks revealed competing optima; comparable checks remain useful for
the new specification.

Current uncertainty uses the inverse observed-information Hessian, scaled for
the mean-household objective, with normal 95% intervals. The old code supported
household sandwich errors conditional on the first stage, although SEs were
disabled in the saved step-4 run. The current Hessian errors are model-based,
condition on estimated price controls and fixed simulation draws, and omit
first-stage estimation uncertainty. The retained household bootstrap refits
both stages and can replace them for the next run.

The current gamma estimates are mu_gamma = -0.05880 and sigma_gamma = 0.06343.
Under the fitted normal distribution, approximately 17.7% of households have
positive gamma and 82.3% have negative gamma. These are population shares
implied by the fitted distribution, not classifications of observed households.
The model therefore estimates average reinforcement alongside a positive-gamma
minority consistent with switching in the specified history index. This is a
clearer economic summary than three separate flavor-specific random slopes,
while depending on the new restrictions on intercepts and history.

The strongest description of the progress is that the new estimator combines
a substantially larger single-person panel with a simpler two-dimensional
heterogeneity structure, an explicitly signed history index, and revised local
price menus and control functions. The changes jointly produce the current
results. They do not isolate which change caused the sharper estimates, and
raw likelihood values across the old and new samples are not comparable as
measures of improvement.

Sources checked: full_draft_1231/est_step4.py, est_step4.md, and
EST_STEP4_HANDOFF.md; shell_files/step4_output/settings.json and
random_beta_gamma.csv; final_draft/est_code/est.py, price_control.py,
hessian_se.py, and README.md; shell_files/res_hess/fit.json, estimates.csv,
and first_stage.csv. The earlier step-2 integration, stock-recursion,
store-aggregation, and SD-reporting corrections are described in the handoff;
they should not be attributed anew to the transition from step 4 to this model.
