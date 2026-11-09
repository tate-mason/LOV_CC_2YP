# Current workflow: where to return after a long break

This chapter describes the current `final_simple` code. The source chapters that follow also retain earlier implementations; their existence is not an instruction to run them instead of the current job.

## The shortest reading path

1. Read `final_simple/flavor_specs.py` to see what each category means.
2. Read `data/summaries.py::run_summaries` and `run` to see how the completed matched data become summaries and labeled estimation inputs.
3. Read `demand/estimation.py::add_stocks`, `parameter_names`, `household_loglike`, and `demand_objective_gradient` for the estimated model.
4. Read `demand/plain_berry_pricing.py::run_comparison`, then `split_pricing.py::targeted_coupons` for the seller's two-period decisions.
5. Read `demand/gamma_calibration.py` for the separate binary information exercise.
6. Read `server/run_after_data.py::run_pipeline` and `export_uncontrolled_pricing.py::export_results` for execution order and recoverable outputs.

The complete production source from `final_simple` is included in this edition, with function and statement annotations. The main job is `server/run_after_data.sh`; the Python driver names its steps explicitly.

## Keep the completed data build

The expensive build converts raw panel/retail records, cleans and combines years, merges tables, and matches purchases with prices. These steps remain in `data/build_data.py`. The current request is to reuse their completed outputs and begin again with summaries. No new local data copy is needed or permitted.

On the research server, from the checkout:

```bash
sbatch Code/final_simple/server/run_after_data.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

The optional second argument selects the full descriptor reference. A recovered reference from the old build is preferred when available. An old three-category mapping alone lacks the detail needed for the new groups.

The driver creates a fresh `after_data_NEW_JOB_ID` child. It estimates both specifications before attempting either pricing stage. The original build and earlier fits stay intact.

## Two specifications are two separate analyses

| Name | Category order |
|---|---|
| `three` | other, berry, plain |
| `five` | other, berry, plain, other fruits, desserts |

Category order matters: the same order must determine count columns, price columns, history arrays, fixed-effect names, and basket coordinates. `categories(specification)` returns an ordered list, not a set. `other` is the reference flavor in both fits, although what remains in that group changes.

The five-group classification gives plain/unflavored first priority, then desserts, then berries, then other fruits, then residual other. Strawberry cheesecake is dessert; peach is other fruit. Cherry/black cherry moves from the legacy berry rule to other fruits in `five`. Descriptor patterns and numeric checks document these choices. Classification rules are not estimates of consumer substitution.

Both runs use uncontrolled heterogeneous split-history demand, the original novelty measure, seven-choice capacity, 128 integration draws, three starts and Hessian inference. Extra categories add fixed effects and alternatives, not extra random novelty/habit coefficients.

## What novelty means here

Let w be flavor j's share of last week's yogurt packages. The original predictor is N = 1 - w. Last week's 75% plain / 25% berry basket gives this week's plain N=.25, berry N=.75, and absent categories N=1. The predictor is known before choosing the current basket.

It measures novelty relative to recent exposure. All plain → berry gives berry a full novelty score, and an all-berry current basket removes that score next week. Mixed baskets adjust novelty gradually. This is not a strict one-time reward for every mixed-basket switch. The proposed lagged-switch-away predictor was not adopted.

H sums prior observed weekly own-flavor shares without decay. Positive H coefficients mean reinforcement; negative ones mean satiation. N and H precede current purchases, which update the next week's state. Missing purchase weeks reset next-week N to zero but do not erase H. Purchase records are proxies for exposure, not observed consumption.

## What the firm chooses and what is held fixed

The main entry point solves two seller problems, plain and berry. Focal sellers and demand alternatives are different lists: all three or five categories stay in demand even though only plain/berry sellers are reported. Rival prices stay fixed; these are separate counterfactuals, not market equilibrium.

The firm uses recovered household posterior mean price, novelty and habit coefficients. Low/high novelty quartiles receive individually optimized coupons; the middle half receives none by design. These groups also differ in price sensitivity and stock preferences. Coupons range from 0% to 100% in 1% steps, with discount factor .92.

An observed initial basket establishes h0. The firm chooses p1, observes basket h1, and then chooses p2 conditional on h1. Exact basket integration gives 120 states for three inside categories and 792 for five at capacity seven. Every rival's own history remains distinct.

## Fixed beliefs describe decisions; persistent types describe consumers

The auxiliary binary theory exercise is separate from coupon pricing. Its uninformed firm uses the same low/middle/high type probabilities after either first-period purchase outcome. It does not learn. Actual consumer types remain fixed over both periods.

For a fixed offer policy, persistent-type evaluation first computes each type's full path revenue, including the probability that this type reaches each second-period state. It then averages across initial types. The informed and uninformed policies must be evaluated over the same initial population.

`profit` retains the optimization objective. `persistent_type_revenue` reports the common policy evaluation. `value_of_information` is known-type revenue minus uninformed-policy revenue under persistent types. This is a different comparison from personalized coupons versus no coupons.

## Read failure states as results of checks

The screenshot's exception came from calibrating the separate binary supply model to observed price moments. The original message grouped convergence, boundary and rank failures together. The revised code tries a larger numerical budget, selects the best admissible start, and records the rejection reason and numerical diagnostics. It retains bounds and identification checks.

If that auxiliary calibration remains unsupported, coupon pricing still finishes. The job exports aggregate calibration moments and a failed status; it does not produce information-value figures from a rejected fit. Demand convergence or primary-pricing errors remain failures, and `run_status.json` identifies which specification failed.

## Recover only the result boundary

Read the new job's `RESULTS_ONLY/run_status.json` first. Each specification contains summary tables/figures, population estimates and WTP, aggregate pricing and policy benchmarks, provenance, and successfully calibrated information figures. Cross-spec CSVs place estimates and benchmark revenues alongside each other.

Only `RESULTS_ONLY` may be copied locally. Prepared samples, weekly likelihood tables, household histories, reference mappings, data partitions and logs remain on the server. The exporter uses explicit allowed names and verifies parameter hashes and `price_control=false`. Rebuilding this textbook reads source code only and does not cross the empirical boundary.
