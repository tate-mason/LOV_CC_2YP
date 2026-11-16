# Estimation observation-calendar audit — November 16, 2026

## Finding

All recent empirical **weekly yogurt demand** implementations inspected before
today's correction build household-week observations from yogurt purchase lines.
They do not join an observed full-shopping-panel calendar. Thus their
capacity-seven versions omit observed weeks with zero yogurt purchases. This
is executable sample construction, not merely inaccurate documentation.

The issue is shared across coefficient specifications, controlled/uncontrolled
fits, flavor aggregations, standard errors, WTP transformations, and downstream
pricing that reuses those fits. Pure theory, simulations, and descriptive
statistics explicitly conditional on purchases are separate cases below.

Scope: source inspection of project estimation implementations, wrappers,
archived code copies, server submission scripts, and allowed aggregate result
provenance. No research records, weekly inputs, or empirical logs were opened
locally. No empirical job was run locally. An absent historical server fit or
weekly table cannot be certified from its exported parameter table alone.

## Implementations and evidence

| Implementation | Observation construction | Conclusion |
|---|---|---|
| `Code/final_draft/est_code/est.py` | `prepare_data` at line 174 reads the matched purchase sample. Lines 215–223 require positive quantities and group existing purchase lines into household-week/flavor counts. Line 296 adds outside counts only to those rows. | Purchase weeks only. |
| The same folder's `est_naive.py`, `est_gamma.py`, `est_het_naive.py`, `est_het_gamma.py`, `est_no_price_control.py`, `est_separate_history.py`, `est_het_history.py`, `est_mixture_gamma.py` | Each imports `main` from `est`; these are different parameterizations of the same preparation code. | All inherit the omission when estimating weekly participation. |
| `Code/final_simple/demand/estimation.py`, before today's correction | The same matched-purchase grouping, originally at lines 393–394, and `outside=max(capacity-Q,0)`, originally line 471. No full-panel calendar input existed. | All seven model choices inherited the omission: naive, homogeneous, mixed_naive, heterogeneous, mixture, separate_history, heterogeneous_history. |
| `Code/final_draft/writeup/documentation_revision/Code/final_simple/demand/estimation.py` | Lines 385–386 group purchase lines; line 460 constructs outside counts. | Archived copy still purchase-week only. |
| `Code/final_draft/writeup/two_spec_revision/Code/final_simple/demand/estimation.py` | Lines 384–385 group purchase lines; line 459 constructs outside counts. | Archived three/five-flavor implementation still purchase-week only. |
| `Code/final_draft/seminar_pres/pricing_walkthrough/source_snapshot/data.py` | `prepare_data` at line 345; purchase-line grouping at line 394; outside count at line 471. | Walkthrough snapshot retains the old omission; it is not the corrected runner. |
| `Code/full_draft_1231/est_step4.py` | Lines 182–200 select inside categories, retain positive quantities, and group only those rows into weeks. Counts are formed by iterating that weekly table. | Purchase weeks only. Real-data fixed-effect and mixed specifications share this preparation. Simulated recovery is a separate complete-calendar calculation. |
| `Code/full_draft_1231/estimation_step1.py` | Lines 194–200 group inside purchases; line 247 derives `hh_weeks` from `weekly_purchases`. | Purchase weeks only. |
| `Code/full_draft_1231/est_step2.py` | Lines 201–207 group inside purchases; line 254 derives `hh_weeks` from those rows. | Purchase weeks only, including the naive/random-coefficient/history variants in this file. |
| `Code/full_draft_1231/est_step3.py` | Lines 246–255 construct observed category purchases; line 299 derives `hh_weeks` from them. | Purchase weeks only, including the detailed-flavor variant. |
| `Code/full_draft_1231/est_hh_demo.py` | Lines 206–212 construct inside purchases; line 255 derives `hh_weeks` from them. | Purchase weeks only, including pairwise/demographic specifications. |
| `Code/full_draft_1231/est_ind_flav.py` | Lines 205–211 construct inside purchases; line 256 derives `hh_weeks` from them. | Purchase weeks only for plain-versus-flavored estimation. |
| `Code/full_draft_1231/inter_intra_est.py` | Lines 234–240 group inside purchases; line 284 derives `hh_weeks` from them; lines 320–323 explicitly skip zero-total baskets. | Positive-purchase baskets only. This is a sequential within-basket estimator, not the all-observed-week participation model. |

Older scripts did **not all use the same outside-count formula**. Step 1,
step 2, step 3, household-demographic, and individual-flavor scripts default to
capacity 14 and calculate `max(14,Q+1)-Q`, forcing at least one outside count
in a retained purchase week. Step 4 and the recent pipelines instead use the
seven-threshold rule `max(7-Q,0)`. Neither formula supplies missing weeks.

## Earlier exploratory code: distinguish trip models from weekly models

- `Code/est_yogurt/est_struc_framework.py` retains `full_panel` before dropping
  unmatched prices (line 106), constructs a roster of trips for yogurt-buying
  households (lines 168–189), and attempts an outside-choice likelihood branch
  (lines 295–322). This is a **trip-level attempt to include nonpurchase**, not
  seven opportunities in every observed shopping week. It cannot be described
  as the same weekly construction. Its use of `if occ.yogurt_buy` is also unsafe:
  merged missing purchases are NaN, NaN is truthy, and an attempted NaN UPC match
  can skip the observation rather than contribute an outside likelihood. I have
  not verified any historical execution or linked result bundle for this file.
- `Code/full_draft_0901/estimation.py` is another trip-level model. It drops
  missing IV residuals at lines 220–222 before copying into `trip_level`, then
  models observed trips and maps unclassified flavors to outside. Its input is
  already restricted by size, price matching and residual availability. A
  missing flavor label is not proof of a genuine nonpurchase. It does not build
  an all-observed-week calendar and is not a clean counterexample to the issue.
- `Code/est_yogurt/est_struc_refactor.py` filters to yogurt modules and drops
  unmatched prices at lines 108–109; it ends with price-IV preparation, rather
  than a complete demand estimator. `dat_check.py` is a merge diagnostic.
- `Code/est_yogurt/est_yogurt.py` contains exploratory purchase-line switching
  regressions/classification. It is not the current weekly participation
  likelihood. Nonpurchase summary variables earlier in a script do not mean
  nonpurchase weeks enter a subsequent demand fit.

## Saved result families and downstream dependencies

| Saved result or use | Finding / strength of provenance |
|---|---|
| `estimates_1111/three/no_control_fit` | Confirmed current presentation source: `seminar_pres/assets/demand_1111_three/provenance.json` names it. Its aggregate fit summary reports 15,105 households, 281,695 weeks, capacity 7, average stock, heterogeneous history. The generating three/five server pipeline uses the purchase-week preparation described above. |
| `estimates_1111/five/no_control_fit` | Same generating pipeline with five flavor categories. Its fit summary reports 280,727 weeks, capacity 7, average stock, heterogeneous history. Changing flavor grouping does not add calendar weeks. |
| `estimates_1111/three/pricing` and `/five/pricing` | `pricing_provenance.json` identifies the reused demand fits. Reusing these estimates propagates the sample problem into empirical pricing; a pricing-only rerun does not correct demand. |
| `seminar_pres/assets/quantified_theory/{plain,berry,other}` | Settings name the November 11 three-flavor point estimates and explicitly say `demand_refitted=false`. All-flavor pricing, revenue, consumer-surplus, welfare and quantity numbers computed from these fitted inputs inherit the issue. The theoretical equations remain a separate matter. |
| `Code/final_draft/writeup/1101_writeup/assets/fit.json` | Confirmed saved fit settings: matched yogurt sample, 281,695 weeks, capacity 7, controlled heterogeneous model. This older presentation/writeup fit uses the same purchase-week construction in `final_draft/est_code/est.py`. |
| `Results/archive/res1108` and `Results/archive/res_pricing1108`, controlled and uncontrolled | The November 8 results-update document says their demand estimate tables match. The repository's controlled/uncontrolled rebuild commands both use the same pre-correction `final_simple` weekly sample builder. Treat these empirical results and their dependent figures as affected. Full original fit settings and weekly totals are not included in these exports, so exact historical execution cannot be independently reconstructed locally. |
| `Results/pricing_no_disc` and `Results/five_flav_res` | Earlier three/five empirical exports. Their pricing provenance records reused capacity-seven demand fits, and documented figure builders reuse these exports. The available weekly demand implementations for these models omit zero weeks. Treat as affected; original full fit settings/weekly records are not present locally for an independent execution-level reconstruction. |
| `Code/final_draft/RESULTS_ONLY/no_control_fit` | An aggregate export without full original fit metadata. It is not evidence of a separate corrected calendar. Exact lineage remains unverified from this export alone. |
| Seminar assets under `pricing`, `aligned_pricing`, `information_pricing` and the November 8 writeup | Provenance names the older result exports above, or identifies the same fit hash as `Results/pricing_no_disc`. Refreshing plots or changing the pricing solver did not refit demand. |

Parameter distributions, their sign shares, WTP and transformed price
coefficients, Hessian/bootstrap inference, likelihood/AIC/BIC and posterior-type
pricing from an affected fit all require reassessment. Standard errors do not
repair sample selection. Differences across old result bundles do not establish
robustness to the observation-calendar correction.

## What this issue does not establish

- It does not invalidate pure theoretical comparative statics, symbolic
  derivations or numerical examples with imposed parameters.
- `Code/MonteCarlo/`, `Code/est_new/random_coefficients_logit.py`, toy-model
  calculations and step 4's simulated recovery use generated choices. Their
  success does not validate selection in real yogurt data, but they do not
  acquire this real-data omission merely by using the same likelihood.
- Purchase-week descriptive quantities—flavor composition, HHI, basket changes,
  purchase gaps—can still describe the stated purchase sample. They are not
  unconditional weekly purchase/nonpurchase rates. Overall full-panel summaries
  and matched-yogurt summaries have different household samples/denominators.
- Vendored BDG MATLAB replication code concerns a different research model;
  no evidence links it to these yogurt estimate exports. It is not classified
  as affected by this specific yogurt-week construction.

## Current corrected run and remaining limitations

The new `server/run_observed_week_refit.py` explicitly sets
`observation_mode=observed_panel` and supplies the full-shopping `panel` input.
It adds genuine zero-yogurt weeks and seven outside counts for those weeks.
The UPC-catalog patch also recognizes retail-known yogurt UPCs when calculating
pre-match totals, instead of relying exclusively on household-panel module codes.

The user-reported server aggregate audit is:

- Observed panel weeks: 3,014,425.
- Unmatched/invalid yogurt weeks excluded: 423,281.
- Genuine zero-purchase weeks: 2,348,144.
- Complete matched purchase weeks: 243,000.
- Zero weeks are 90.6219% of the 2,591,144 eligible weeks before price checks.
- Those zero weeks represent 16,437,008 additional outside choices before
  price-coverage/household filters.

These are supplied aggregate counts, not a completed corrected fit. The later
`Retained calendar` output is needed to measure final coverage. In particular,
zero weeks require covered flavor prices, with an explicit same-week DMA
fallback; incompletely matched yogurt weeks remain excluded; households without
membership in the original prepared yogurt sample are not newly added.

Only the main `final_simple` pipeline and the new runner are corrected today.
`final_draft/est_code`, historical copies and old submission scripts must not be
mistaken for corrected alternatives. Ordinary new capacity-seven runs now require
`--panel`; old launchers lacking that argument will fail rather than silently
produce another purchase-week-only participation fit. Legacy reconstruction
explicitly requests `purchase_weeks` to reproduce old estimates; that is not a
correction or a new fit.

The current job's long silence after `Using merged prices directly` is explained
by the existing `add_stocks` per-household/per-row Python loop, before optimization.
Its duration is not evidence of convergence or optimizer progress. The corrected
sample is roughly an order of magnitude larger in weeks before coverage checks.
No claim of completion within the presentation deadline is justified yet.

## Interpretation limit

The omission mechanically overstates inside-choice frequency relative to the
intended observed-week population. It does not imply that only the inside
intercept moves. Price sensitivity, history coefficients, heterogeneity, WTP,
and empirical policy comparisons may also change. Neither their signs nor
their magnitudes can be certified before inspecting the corrected estimates.
