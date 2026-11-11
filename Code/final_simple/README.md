# Readable research code: current workflow

Empirical reads and writes are research-server-only. `Code/privacy/server_data_policy.py` enforces that boundary; local path overrides cannot authorize empirical processing. Copy back only the job's allowlisted `RESULTS_ONLY` folder. Never copy raw panels, purchase histories, household tables, weekly inputs, data partitions, or logs.

The main empirical analysis uses **uncontrolled split-history demand** with the original novelty measure. Two flavor specifications run alongside each other, with separate outputs:

| Specification | Demand categories |
|---|---|
| `three` | other, berry, plain; original classification retained |
| `five` | other, berry, plain, other fruits, desserts |

`flavor_specs.py` defines category order and classification once for summaries and demand. In the expanded specification, dessert cues beat fruit cues; plain/unflavored keeps first priority. Strawberry cheesecake is dessert; peach is other fruit. Cherry moves from the original berry rule to other fruits only in `five`. Coffee/honey without other classification cues remains residual other.

## Restart after the completed data build

From the research-server checkout, after syncing source:

```bash
sbatch Code/final_simple/server/run_after_data.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_ORIGINAL_JOB_ID
```

An optional second argument selects the descriptor reference. Otherwise the job prefers the original run's recovered reference CSV, then the configured Nielsen descriptor reference. It reuses `data/merged_panel.parquet` and the matched `data/yogurt_estimation_sample.parquet`; it does not repeat conversion, cleaning, merging, or matching.

For each specification it reruns **summaries → uncontrolled estimation → pricing**. Both fits finish before pricing begins. Plain and berry remain the flavor sellers in separate pricing counterfactuals; all selected demand categories remain in the model and basket histories.

New outputs live in `run_ORIGINAL_JOB_ID/after_data_NEW_JOB_ID/`. Only its `RESULTS_ONLY/` directory should be recovered. It has separate `three/` and `five/` results, cross-spec estimate/revenue comparison CSVs, and `run_status.json`. Existing fits/results are not replaced. Check status before interpreting results.

## Reading order

| File | Read it to understand |
|---|---|
| `flavor_specs.py` | Shared classification and ordered three/five category lists |
| `data/build_data.py` | Expensive raw-data build; skipped by the after-data job |
| `data/summaries.py` | Descriptive statistics and specification-specific labeled demand input |
| `demand/estimation.py` | Histories, likelihood, gradients, optimization, inference and WTP |
| `demand/regenerate_weekly_inputs.py` | Reconstruct matching weekly inputs from saved settings without reestimating demand |
| `demand/history_distributions.py` | Population distributions from saved estimates; no household input needed |
| `demand/plain_berry_pricing.py` | Main plain/berry comparison using the uncontrolled fit |
| `demand/split_pricing.py` | Shared persistent-type recovery and exact basket backward induction |
| `demand/gamma_calibration.py` | Auxiliary binary supply calibration and fixed-belief offers |
| `server/run_after_data.py` | Visible two-spec pipeline and aggregate comparison exports |
| `server/export_uncontrolled_pricing.py` | Allowed export filenames and fit provenance checks |
| `supply/` | Separate theoretical mechanism exercises |

The textbook is [Python 101](../final_draft/code_guides_pdf/python101_walkthrough.pdf). Its current-workflow chapter and lessons 18–22 explain the revised code; source chapters cover every production Python module in `final_simple`. Historical `final_draft` listings remain labeled as earlier implementations.

## Economic specification

For flavor j, `N = 1 - last week's own-flavor share`, or zero when the preceding calendar purchase week is unavailable. `H` is the average of all prior observed purchase-week own-flavor shares, with zero for empty prehistory. Missing calendar weeks add no observations to its denominator. `--stock-measure decaying` or `cumulative` remain available for sensitivity runs and saved-fit reconstruction. Predictors precede current purchases. `N` measures novelty relative to last week's basket: binary for single-flavor baskets and gradual for mixed baskets. It does not promise a one-time bonus for mixed baskets. The rejected lagged-switch experiment is not selected by the current jobs.

Utility includes `gamma_i*N + phi_i*H + alpha_i*price` plus constants. Novelty and stock coefficients are independently estimated and unrestricted in sign. Positive stock means reinforcement; negative stock means satiation. Alpha is negative lognormal. A unit of history is a weekly package share, not an additional package. WTP for N refers to a full unit increase in N.

The main job sets `model=heterogeneous_history`, `capacity=7`, `no_price_control=True`, `novelty_measure=previous_share`, 128 draws, three optimizer starts, and Hessian inference. Ordinary estimator settings still support controlled and earlier-model sensitivity fits; label those outputs separately. `capacity=0` is conditional flavor demand and is not the participation model used by current pricing.

## Pricing and information are different comparisons

Coupons range from 0% to 100% in one-percentage-point steps; the revenue discount factor is 1 (no time discounting). Posterior mean types are treated as known by default. Low/high novelty quartiles get personalized coupons; the middle half stays at posted prices by design. Rival prices stay fixed. Exact state counts at capacity seven are 120 for three inside categories and 792 for five.

The primary exercise compares common pricing and personalized pricing. The separate binary type-information exercise chooses uninformed offers using fixed beliefs, but evaluates both informed and uninformed policies under the same persistent consumer types. `profit` retains the fixed-belief objective; `persistent_type_revenue` is the common evaluation, and `value_of_information` is its known-minus-uninformed difference.

A rejected auxiliary calibration does not invalidate completed coupon pricing. Read `information_calibration_status.json`: if calibration is unsupported, aggregate moment/optimizer diagnostics are exported and information-value figures are omitted. No substitute calibration is silently used. Demand/pricing failures remain failures.

## Other entry points

`server/run_split_pricing.sh EXISTING_REBUILD_RUN_DIR` reruns only pricing using a completed original-share uncontrolled fit. It regenerates matching server inputs and checks the saved likelihood. The after-data job is needed when summaries or grouping change.

`demand/split_pricing.py --flavor all` is a sensitivity launcher: all means every category in its saved fit, not a hardcoded three. The main job intentionally calls only plain and berry. `demand/pricing.py` is the earlier three-group combined-gamma berry model with delta=1 by default; it is not the new split-history workflow. `server/run_novelty_estimation.sh` is the unadopted lagged-novelty experiment, not the current job.

Each calculation file retains editable settings and explicit run steps. Command-line flags override settings for server submissions. Keep separate output directories for separate models; do not edit saved fit metadata to change specifications.

## Validation and rebuilding documentation

Local checks may test pure numerical kernels and source/policy behavior. Do not run the older empirical-I/O fixture suite locally merely because it is called synthetic. Use `python3 Code/privacy/check_policy.py` for the boundary check. Current pure three/five numerical checks are in `checks/check_flavor_specifications.py`; they use descriptor strings, aggregate count arrays, and theoretical price moments, without household panels or empirical I/O.

From `Code/`, rebuild the textbook with `python3 final_draft/build_python101.py`. This parses source text; it does not run research code. `--markdown-only` updates the readable source and coverage manifest without typesetting the PDF. See `checks/VALIDATION.md` for verification limits; passing numerical checks is not a completed empirical rerun.

Pricing figures and public policy tables use **Common pricing** and **Personalized pricing**. Price paths show only `p1 → E[p2]`. The posted-price policy remains an internal numerical diagnostic; public gains are relative to common pricing. `Code/pricing_presentation.py` supplies the shared presentation rules.
