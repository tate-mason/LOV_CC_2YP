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

`plain_berry_pricing.py` always chooses two focal sellers: plain and berry. In each counterfactual only the focal product's price changes and only its sales earn revenue. All other demand categories and constructed nonpurchase remain alternatives. These are fixed-rival counterfactuals, not equilibrium prices from competing firms.

The focal choice probability can be written as logistic of its utility minus the log-sum-exp of all rival utilities and nonpurchase. That probability aggregation does not collapse the rivals' separate history states. The basket enumeration retains every category: `choose(K+J,J)` states for J inside flavors and K opportunities, hence 120 or 792 at K=7.

An observed basket sets h0; first-period offers induce a full basket h1; terminal offers condition on h1. Low/high novelty quartiles get coupons and the middle receives none by design. Groups rank gamma only, but also differ in posterior price sensitivity and habit. Group differences are not a causal effect of gamma holding everything else fixed.

The default `posterior-mean` treatment makes the estimated coefficient vector known to the firm. `posterior-integrated` is a separate sensitivity with posterior updating after counterfactual baskets; it is not the fixed-belief auxiliary experiment.

## Fixed beliefs and persistent types

In the auxiliary binary exercise, the uninformed firm uses unchanged low/middle/high type probabilities when choosing offers after either purchase outcome. Actual consumers retain their type across both periods. Revenue is computed along each persistent type's full path and averaged afterward. The informed policy is evaluated over the same initial population.

Use `persistent_type_revenue` for policy comparisons and `value_of_information` for known minus uninformed revenue. `profit` is retained as the optimization objective and can differ from the persistent-type evaluation. The information value is distinct from personalized-coupon revenue gains relative to no coupons.

`information_calibration_status.json` distinguishes complete from rejected calibrations. Rejection exports aggregate price moments and optimizer diagnostics; coupon pricing continues, but no invalid information-value charts are produced. Bounds/identification checks are retained.

## Outputs and legacy scripts

Copy only `RESULTS_ONLY`, with `run_status.json`, `pricing_provenance.json`, population estimates/WTP, grouped revenue and benchmark CSVs, and aggregate figures. Full fit settings, weekly inputs and internal manifests stay on the server. The exporter verifies fit/parameter hashes and `price_control=false` for both focal sellers.

`split_pricing.py --focal-flavor all` selects all categories of its saved fit for optional separate-seller cases. Main pricing remains plain/berry. `pricing.py` supports only the earlier three-group combined-gamma model (discount default .95); it explicitly rejects five-group fits. `EMPIRICAL_TARGETING.md` documents that legacy exercise and is not the current submission guide.
