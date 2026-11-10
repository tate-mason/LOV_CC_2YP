# Demand and pricing: current reading guide

Start with [the main workflow](../README.md). All empirical input/output is server-only. The current after-data job fits **three and five categories separately**, both with uncontrolled heterogeneous split-history demand. It retains the original share novelty and cumulative stock.

## What is estimated

For each inside flavor j:

```
V_ijt = inside_constant + beta_j + alpha_i*p_ijt + gamma_i*N_ijt + phi_i*H_ijt
N_ijt = 1 - w_ij,t-1     # if preceding calendar purchase week is observed; else 0
H_ijt = sum(w_ijs for observed s < t)
w_ijt = packages of j / all yogurt packages in that purchase week
```

Other is the reference flavor. Expanding the categories adds `beta_other_fruits` and `beta_desserts`, with their own choice probabilities, shares, menus and stocks. Gamma/phi remain household coefficients shared across flavor alternatives; the expansion does not add flavor-specific random coefficients. Types persist across observed weeks.

Alpha is negative lognormal; gamma and phi are independent normal population coefficients. Positive gamma favors flavors less represented last week; positive phi means reinforcement and negative phi means satiation. These are purchase-based proxies. Mixed baskets receive gradual novelty scores. There is no first-ever-consumption claim and no lagged switching-event predictor in the current jobs.

The outside utility is zero. Seven opportunities imply outside count `max(0,7-Q)` and total count `max(7,Q)` for estimation. Weeks with no observed purchase are not inserted as observed outside weeks. Forecast pricing uses seven opportunities. Histories are constructed before likelihood exclusions and updated after each current predictor is recorded.

The estimator retains optional controlled fits and older model choices as sensitivities. The main jobs pass `--no-price-control`, `--model heterogeneous_history`, `--novelty-measure previous_share`, and `--flavor-spec three` or `five` explicitly. `flavor_spec` is also editable in the Parameters section. Keep each fit with its actual specification metadata.

## Nu and the T1EV epsilon

In the price-control notation, nu is the error in the first-stage price equation:
`p_ijt = fitted_price(IV, history controls, week FE, DMA FE) + nu_ijt`.
`add_price_controls` estimates its residual `nu_hat` in dollars and stores it as
`cf_<flavor>`. Controlled utility adds `price_control * nu_hat` to V. This term
allows the part of price unexplained by the instruments and controls to be
associated with demand. Nu is not assumed to be T1EV. Current uncontrolled fits
omit this term.

Epsilon is a different error: `U_ijt = V_ijt + epsilon_ijt`, the independent
Type-I extreme-value utility shock for each choice opportunity and alternative,
conditional on household coefficients and observed controls. Its scale is
normalized to one. Integrating epsilon analytically gives the softmax/logit
probabilities in `household_loglike`; it is not a price residual or an estimated
household coefficient. Persistent normal taste draws are called `z` in this
implementation (sometimes denoted nu in mixed-logit notation); those draws
generate alpha/gamma/phi and are integrated across the entire household history,
whereas epsilon represents transitory choice variation.

## Read the code in this order

1. `../flavor_specs.py`: descriptor precedence and ordered category lists.
2. `estimation.py::prepare_data`, `add_stocks`, `parameter_names`: menus, histories, parameter-vector meaning.
3. `household_loglike`, `demand_objective_gradient`: integrate complete persistent-type likelihoods and their analytic gradients.
4. `optimize_demand`, `fit_model`, `implied_wtp`: optimizer starts, inference, saved settings, and unit interpretation.
5. `regenerate_weekly_inputs.py`: reconstruct the original table with the saved flavor specification and verify household/week totals and likelihood. This does not reestimate demand.
6. `split_pricing.py::posterior`, `compositions`, `forecast_context`, `targeted_coupons`: type recovery and exact basket backward induction.
7. `gamma_calibration.py`: separate binary theory calibration and information evaluation.
8. `plain_berry_pricing.py` and `../server/run_after_data.py`: assemble the main counterfactuals and pipeline.

## Main server command

From the server checkout:

```bash
sbatch Code/final_simple/server/run_after_data.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

This skips the completed data build, reruns summaries and estimation for both category specifications, then runs plain and berry pricing for each. Supply theory simulations remain separate.

For pricing alone from an existing uncontrolled original-share fit:

```bash
sbatch Code/final_simple/server/run_split_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

The pricing-only job expects that run's `fits/no_control`; the after-data job creates separate specification directories. A five-category fit needs a weekly table reconstructed from that same fit. Never pair it with a three-category or controlled table.

## How pricing uses the expanded model

`plain_berry_pricing.py` always chooses two flavor sellers: plain and berry. In each counterfactual only the flavor product's price changes and only its sales earn revenue. All other demand categories and constructed nonpurchase remain alternatives. These are fixed-rival counterfactuals, not equilibrium prices from competing firms.

For each plain/berry exercise, **flavor** means that selected flavor and **outside**
means everything except it, including other yogurt flavors and nonpurchase.
`P(outside) = 1 - P(flavor)` using the unchanged multinomial probabilities.
This pricing outside is broader than the demand estimator's nonpurchase
alternative, whose normalized utility is zero. A basket classified as outside
contains zero units of the selected flavor; mixed baskets containing that flavor
belong to the flavor-purchase branch.

The flavor choice probability can be written as logistic of its utility minus the log-sum-exp of all rival utilities and nonpurchase. That probability aggregation does not collapse the rivals' separate history states. The basket enumeration retains every category: `choose(K+J,J)` states for J inside flavors and K opportunities, hence 120 or 792 at K=7.

Pricing defaults to `delta = 1`: revenue is first-period revenue plus expected
terminal revenue, with no time discounting. Coupon percentages still reduce
offered prices. `pricing_theory_comparison.pdf/png` uses the supply theory's
three-panel layout: p1-to-p2 price paths, expected two-period revenue, and revenue
gain. It shows the existing low/middle/high groups and retains their probabilities.
The CLI selector is `--flavor`; comparison exports use `flavor_comparison.csv`
and a `flavor` column.

An observed basket sets h0; first-period offers induce a full basket h1; terminal offers condition on h1. Low/high novelty quartiles get coupons and the middle receives none by design. Groups rank gamma only, but also differ in posterior price sensitivity and habit. Group differences are not a causal effect of gamma holding everything else fixed.

The default `posterior-mean` treatment makes the estimated coefficient vector known to the firm. `posterior-integrated` is a separate sensitivity with posterior updating after counterfactual baskets; it is not the fixed-belief auxiliary experiment.

## Fixed beliefs and persistent types

In the auxiliary binary exercise, the uninformed firm uses unchanged low/middle/high type probabilities when choosing offers after either purchase outcome. Actual consumers retain their type across both periods. Revenue is computed along each persistent type's full path and averaged afterward. The informed policy is evaluated over the same initial population.

Use `persistent_type_revenue` for policy comparisons and `value_of_information` for known minus uninformed revenue. `profit` is retained as the optimization objective and can differ from the persistent-type evaluation. The information value is distinct from personalized-pricing revenue gains relative to common pricing.

`information_calibration_status.json` distinguishes complete from rejected calibrations. Rejection exports aggregate price moments and optimizer diagnostics; coupon pricing continues, but no invalid information-value charts are produced. Bounds/identification checks are retained.

## Outputs and legacy scripts

Copy only `RESULTS_ONLY`, with `run_status.json`, `pricing_provenance.json`, population estimates/WTP, grouped revenue and benchmark CSVs, and aggregate figures. Full fit settings, weekly inputs and internal manifests stay on the server. The exporter verifies fit/parameter hashes and `price_control=false` for both flavor sellers.

`split_pricing.py --flavor all` selects all categories of its saved fit for optional separate-seller cases. Main pricing remains plain/berry. `pricing.py` supports only the earlier three-group combined-gamma model (delta=1 default); it explicitly rejects five-group fits. `EMPIRICAL_TARGETING.md` documents that legacy exercise and is not the current submission guide.

Pricing figures and public policy tables use **Common pricing** and **Personalized pricing**. Price paths show only `p1 → E[p2]`. The posted-price policy remains an internal numerical diagnostic; public gains are relative to common pricing. `Code/pricing_presentation.py` supplies the shared presentation rules.
