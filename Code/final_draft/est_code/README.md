# Yogurt demand estimation

The main script is `est.py`. It uses the revised sample construction in
`../data_code`, the model discussion in `../../full_draft_1231/updates/draft_2/draft_1010.tex`,
and the weekly aggregation lessons in `../../full_draft_1231/EST_STEP4_HANDOFF.md`.
The current instructions supersede the older draft: beta is common across
households; gamma and alpha are heterogeneous across households; gamma is common across flavors; delta is one; own-flavor history
includes all earlier periods.

## Four model specifications

Beta is common across households in all four models. Each entry point accepts
all the same data, price-control, optimization, and bootstrap options:

| Entry point | `--model` | Gamma | Alpha |
| --- | --- | --- | --- |
| `est_naive.py` | `naive` | Absent | Common |
| `est_gamma.py` | `homogeneous` | Common across flavors | Common |
| `est_het_naive.py` | `mixed_naive` | Absent | Varies across households |
| `est_het_gamma.py` | `heterogeneous` | Varies across households | Varies across households |

For example, run each entry point with `--flavors /path/to/flavors.csv` and a
separate `--output` directory. Equivalently use `est.py --model MODEL`.
The default for `est.py` remains `heterogeneous`. Mixed models integrate a
persistent household alpha; the gamma model also integrates persistent gamma.
All four use the same sample requiring at least two usable purchase weeks.

## Run on the server

Install packages in your usual Python environment:

```bash
python3 -m pip install -r requirements.txt
```

Start with a small sample and separate output directory:

```bash
python3 est.py \
  --max-households 100 --draws 32 --starts 2 --se-method none \
  --output results_small
```

Then run the full sample, increasing draws if estimates change materially:

```bash
python3 est.py \
  --draws 128 --starts 5 --bootstrap-reps 49 --output results_full
```

Defaults point to `/scratch/dtm63837/Kilts_Panel/Data/`:

- `--sample`: `summary_outputs/yogurt_estimation_sample.parquet`, the full prepared
  purchase table saved at the end of `summ_stats.py`.
- Years: 2006–2015. Change with `--first-year` and `--last-year`.

Run `summ_stats.py` first. Estimation uses the saved `flavor` labels and merged
`price` column directly; it does not reload RMS or rematch its UPCs. `price_rms`
is accepted if `price` is absent. Price controls use the consumer `dma_cd`
(`--household-dma-column` overrides the name). DMA codes must be positive integers.
The optional flavor CSV is only used for older input tables without saved labels.

Households must have at least two distinct usable purchase weeks after menu/IV
exclusions. Weeks need not be consecutive. Demand purchases are restricted to
`household_size == 1` before histories and weekly quantities are constructed.
Missing or other household sizes are excluded; a missing `household_size` column
is an error. This uses size on each purchase record, so only single-person
periods contribute when household composition changes. All matched purchase
records remain available for store price menus and retail instruments.
The serving-code filter is not applied. The development
limit takes the lowest household IDs deterministically; it is not a random sample.

## Xi: exact definition

A period is a calendar week ending Saturday. Let `q_ijt` be the number of packages
of flavor j bought by household i across **all stores** in week t. Let
`x_ijt = q_ijt / sum_k q_ikt` be its share among inside purchases.

```text
S_ijt  = sum over all observed s < t of x_ijs
N_ijt  = sum over k != j of x_ik,t-1
Xi_ijt = (1 - lambda) * N_ijt - lambda * S_ijt
```

`lambda=0.7`; adjust using `--habit-weight`. Delta is fixed at one, with no
window, decay parameter, log transformation, or recursive sum of previous Xi.
Both S and N are zero on the first observed week. Current purchases enter S only
after current Xi has been recorded. Each household starts its own history.

N uses **only the immediately preceding calendar week**, not all past other-flavor
purchases and not the last purchase week across a gap. If the preceding week has
no matched purchase, N is zero. S retains all observed prior purchases even across
gaps and years. Missing matched purchases therefore mean missing history: zero
recent purchases is a maintained measurement assumption, not proof of no purchase.

Example: after a week buying only berry, next week's berry alternative gets
`N=0`, while other and plain get `N=1`. Berry's accumulated S rises by one.
If the next purchase is two weeks later, all three N terms are zero, while the
berry stock remains one.

Positive gamma penalizes repeated own-flavor purchases and rewards recent
other-flavor purchases. Negative gamma reverses those effects and represents
habit/familiarity. Gamma is not constrained to be positive.

This is a **specified index**, not two independently estimated habit and novelty
coefficients. Gamma scales both pieces at the fixed lambda tradeoff. The positive
N term rewards an alternative that differs from last week's flavors; it does not
by itself imply an untried product. S measures cumulative purchase shares rather
than observed consumption or physical inventory. With delta=1 it grows throughout
the panel, so check sensitivity to panel length and the unknown initial history.

## Utility and household heterogeneity

For each inside flavor j:

```text
V_ijt = beta_j + gamma_i * Xi_ijt + alpha_i * price_jt + sigma * residual_jt
beta_other = 0
beta_berry, beta_plain are common across households

gamma_i  = mean_gamma + sd_gamma * z_i_gamma
alpha_i  = -exp(mu_log_minus_alpha + sd_log_minus_alpha * z_i_alpha)
```

Two independent standard normal draws represent one price coefficient and
one history coefficient common to all flavors. Independence is a simplifying
restriction; correlated gamma/alpha are not estimated. Beta never varies with
draws. Alpha is always negative. The population mean alpha is
`-exp(mu_log_minus_alpha + sd_log_minus_alpha**2 / 2)`, not
`mu_log_minus_alpha`. The standard deviation parameter is on log(-alpha).

A draw represents a persistent household type across **all** its weeks. For each
draw the script first sums count-weighted log probabilities across the household,
then integrates household likelihoods using log-sum-exp. Averaging probabilities
separately each week would incorrectly redraw household types.

Mixed-flavor baskets remain count vectors. The model treats package counts as
repeated choices with the same weekly probabilities, conditional on quantity.
It is a count logit inspired by history-dependent choice, not a replication of
Dube's bundle/quantity model. Purchases within a week do not update Xi until the
next week. Differing package sizes are not converted into consumption units.

## Outside choices and scope

The default `--capacity 0` estimates flavor choice **conditional on purchasing
yogurt**. There is no outside alternative or common inside constant, because
purchase lines alone do not identify category participation.

To reproduce the older handoff's constructed outside-count convention, explicitly
set `--capacity 7`. The script then adds:

```text
outside_count = max(7 - total_inside_package_count, 0)
V_outside = 0
```

It also estimates a common inside constant. Counts above seven are retained and
get zero outside count. Only observed yogurt purchase weeks enter; missing weeks
are never inserted as seven outside choices. This convention assumes packages
proxy consumption occasions and is an assumption, not an observed nonpurchase
outcome. Fits under different capacities have different outcomes and likelihoods.

## Prices and sample audits

Prices use the merged `price` column directly, without a `prmult` adjustment.
Repeated observations of a store/week/UPC quote are deduplicated, then prices
are averaged across the observed UPCs of each flavor at that store/week.
The exposure store is the household's store with most purchased packages that
week; ties use the lowest store code. Price coverage is limited to the records
in the merged table, rather than a separate retail assortment file.

`summ_stats.py` drops yogurt purchase lines without finite positive matched
prices, then requires at least two distinct purchase weeks before saving the
prepared table. Full-panel demographic summaries retain the full panel.

Choice sets contain the flavors with finite positive prices at the exposure
store/week. An unpriced, unpurchased flavor is excluded from the logit denominator;
it does not cause the household-week to be dropped. Every purchased flavor must
have a covered price. Weeks failing that condition are excluded after updating
history; the two-usable-week rule is then reapplied. No prices are imputed.
Conflicting prices or flavors for the same store/week/UPC stop estimation.

The script recomputes Saturday weeks from purchase dates. `panel_week.py` can map
Sunday purchases to the preceding Saturday; because `master_dat.py` already
inner-joined using that variable, recomputing dates cannot recover purchases lost
upstream. Audit and rebuild that merge before treating the sample as final.


## Other-DMA price control (on by default)

For each week, pool valid merged price quotes across **all flavors** and compute
one overall mean outside the consumer's `dma_cd`. Repeated purchases of the same
store/week/UPC quote within a DMA count once. Each remaining quote has equal
weight; this is not a mean of DMA means or a flavor-specific instrument.
Prices come from the full merged input even when `--max-households` limits demand
estimation. At least one other-DMA quote is required.

The same `iv_overall` is stored as `iv_other`, `iv_berry`, and `iv_plain` for the
three existing price first stages. Missing instruments exclude that week's
likelihood after updating its purchase history.

Estimate each flavor's OLS first stage only on retained household-weeks where
that flavor has a covered price. Unavailable alternatives have no first-stage
observation and their utilities are masked out of the likelihood:

```text
price_ijt = pi_j * instrument_it + coefficients_j * (Xi_other, Xi_berry, Xi_plain)
            + week fixed effects_j + consumer DMA fixed effects_j + residual_ijt
```

Each household-week has equal first-stage weight. Multiple households with the
same exposure therefore repeat its price observation. Fixed effects are removed
by repeated demeaning rather than a large dummy-variable matrix. Xi controls
condition the pricing equation on the observed state variables. Rank-deficient
Xi controls are handled by least squares; an instrument with no independent
variation after these controls stops estimation.

The residual enters its flavor's utility in dollars, with one common estimated
`sigma` (`price_control` in the output). It has no outside-option counterpart.
This is a linear residual control, not a fully flexible control-function model.
`--no-price-control` explicitly disables both the first stage and residual term.

`first_stage.csv` gives each flavor's IV slope, household-clustered IV SE/F
(with the household-count correction), partial R-squared, and residual SD.
These are relevance diagnostics; no F threshold establishes instrument validity.
The other-DMA exclusion requires prices outside the consumer's DMA to be
uncorrelated with their unobserved demand shocks after conditioning. National
advertising or common demand shocks can violate this assumption. Construction
of the instrument does not establish causal identification.

## Files and results

- `est.py`: data preparation, Xi, likelihood, and optimization, in that order.
- `price_control.py`: other-DMA instruments and first-stage regressions.
- `bootstrap_se.py`: household resampling and two-stage standard errors.
- `est_naive.py`: common alpha/beta, no Xi effect.
- `est_gamma.py`: common alpha/beta and common flavor-specific gamma.
- `est_het_gamma.py`: same full model as `est.py`, random gamma and alpha.

Use separate `--output` directories for each model. `weekly_inputs.csv` contains
counts, exposure stores, prices, stocks, novelty, and Xi for inspection.
`estimates.csv` reports coefficients and boundary flags. `fit.json` records
settings, convergence, and every starting solution's parameter vector and NLL.
The multinomial combinatorial constant is omitted from NLL because it does not
depend on the parameters. NLL comparisons require identical data and outcomes.

## Standard errors (on by default)

With `--se-method bootstrap`, the default is 49 household-cluster bootstrap replications. Each replication
samples households with replacement and retains each entire history. Repeated
households receive different IDs, so their likelihoods remain separate. Existing
Xi values remain attached to the original histories, including the contribution
of purchase weeks excluded for missing menus or instruments.

**Both stages are re-estimated:** refit all three price equations, recompute
residuals, and re-estimate demand. This includes uncertainty in the estimated
control residuals. The underlying RMS observations, constructed instruments,
and integration draws stay fixed. Thus inference is conditional on that retail
price universe and does not capture retail-price sampling uncertainty, common
DMA/time shocks across households, or numerical simulation error.

The reported SE is the sample SD of usable bootstrap estimates. The 95% intervals
are bootstrap percentiles; no normal-Wald p-values are supplied. Random-coefficient
SDs can be on the zero boundary, so percentile intervals also need care there.
Increase `--bootstrap-reps` (for example to 499) for final inference.

Each bootstrap fit starts at the main estimate and uses `--bootstrap-starts 2`
local starts by default. A nonconverged main fit skips inference. Nonconverged
or unidentifiable bootstrap fits are logged and excluded. SEs are withheld if
fewer than 20 or less than 80% of requested replications succeed. Inspect failures
rather than treating the accepted distribution as automatically valid. Bootstrap
inference can be slow: `--se-method none` disables it for development runs.

`point_estimates.csv` is saved before bootstrapping. `bootstrap_draws.csv` is
checkpointed after every replication, including failures and parameter vectors.
`covariance.csv` contains the bootstrap covariance. `estimates.csv` contains point
estimates, SEs, boundary flags, and percentile limits. `fit.json` records the
inference status; missing SEs mean inference was disabled or failed its checks.

Before inference, check convergence, bounds, broader starts, integration draws,
and draw seeds. Normal random types independent of zero initial stocks remain
assumptions; common beta does not eliminate initial-conditions identification
issues. The bootstrap does not fix model misspecification or weak instruments.

Local checks need no raw data:

```bash
python3 -m unittest -v test_est.py test_price_control.py
```

References: [Lattin, balanced choice](https://www.gsb.stanford.edu/faculty-research/working-papers/model-balanced-choice-behavior)
and [Dube, multiple discreteness](https://doi.org/10.1287/mksc.1030.0041).


## Summary statistics

`../data_code/summ_stats.py` produces demographic tables for the full panel and
matched yogurt sample, before final menu/IV exclusions. Matched yogurt summaries
require at least two distinct purchase weeks within the selected years; this
restriction applies to demographics, coverage, and purchase dynamics. Full-panel
summaries retain all households. Run it with the same
flavor reference and year range. It writes CSV and LaTeX, plus optional figures:

```bash
python3 ../data_code/summ_stats.py --flavors /path/to/Latest_Flavor_2010.csv
```

Each household counts once; its earliest available nonmissing demographic value
is used. The main race table treats Hispanic origin code 1 as a Hispanic race
category, following the original script. The reported race is preserved in
`race_original` internally. Income and head-age categories remain codebook codes
until you fill the label dictionaries at the top of that file. No income-code
or age-code means are interpreted as dollars or years. All shares are unweighted,
and missing demographics appear explicitly in the denominator.

Trip counts use trip IDs, not purchase dates. Annual participation uses distinct
observed panel years rather than filling gaps. Trips per approximate observed
week assumes full-year exposure in each observed year, not verified enrollment.
Flavor shares and HHI use purchased quantities. Switching/spells are defined for
adjacent calendar weeks with single-flavor baskets; mixed baskets and gaps break
the sequence. Transition denominators include stays. Commented examples cover
loyalty by income and summaries/selection for final estimation inputs.

Control-function reference: [Petrin and Train](https://emlab.berkeley.edu/~train/papers/petrintrain.pdf).

The optional bootstrap uses 49 replications (override with `--bootstrap-reps`).
The main demand fit retains 128 simulation draws.


Estimation now uses an analytic gradient of the same household-integrated
likelihood. It reports progress every ten iterations and saves completed starts
in `starts_checkpoint.json`. The full model has seven parameters with the price
control (eight with the constructed outside option). Existing running jobs must
be restarted with the updated code to use these changes; old flavor-specific
parameter vectors are incompatible.

For development, use `--draws 32 --starts 1 --se-method none`; confirm the final
fit with more draws and starts before running the bootstrap. Reducing draws
changes simulation accuracy. Additional allocated CPUs alone do not parallelize
this implementation.

Summary figures now include `flavor_hhi_histogram.pdf` and `flavor_hhi_cdf.pdf`
in the main summary directory for detailed, unaggregated flavor labels, and in
`aggregated_flavors/` for the three groups. HHI sums squared household quantity
shares across all observed flavors over the selected years; each household has
equal weight. `--no-figures` disables these along with the existing figures.

Gamma models save `gamma_distribution.pdf` immediately after the main fit,
before bootstrapping. It shows the fitted population normal density with its
mean, SD, and implied share below zero; negative gamma represents habit under
the specified Xi index. Common gamma or zero estimated SD appears as a point
mass. This is the distribution across household types, not a confidence interval
or the distribution of household posterior estimates. Nonconverged or boundary
fits are marked provisional. Naive models have no gamma graph.


Summary runtime: purchases are now written once to temporary household hash
partitions, reused for demographic and flavor summaries. Each demographic
history is deduplicated and sorted once across all demographic fields; exact
trip/UPC counts still use all purchase lines. Partitioning requires Polars >= 2.0
(see requirements.txt). Keep `--household-batches 16` initially; this now controls
physical partition size rather than repeated scans of the original input.
Temporary partitions require disk space in the output directory and are removed
at exit. Logs report partitioning and per-batch elapsed times. Existing running
summary jobs must restart with the new code to use this change.

To regenerate only the HHI figures from saved household summaries, without
re-reading purchases or rerunning demographics:

```bash
python3 ../data_code/summ_stats.py --hhi-only --output /path/to/summary_outputs
```

This needs `yogurt_household_flavors.csv` in that directory. It also regenerates
the grouped versions if `aggregated_flavors/yogurt_household_flavors.csv` exists.
The ordinary summary command generates these figures by default; `--no-figures`
disables them. Each saved HHI graph is now named in the log.


## Fast Hessian standard errors (current default)

`--se-method hessian` is now the default for all model entry points. After a
converged fit, it differentiates the analytic gradient with central differences
(2 evaluations per parameter), inverts the observed information, and scales by
the number of households because the objective is the mean household negative
log likelihood. It saves `covariance.csv` and normal 95% intervals in
`estimates.csv`; `fit.json` identifies the method and limitations.

These are model-based SEs conditional on the estimated price
controls: they omit first-stage estimation uncertainty and are not sandwich
robust. Inference is withheld for unconverged fits, parameters near bounds, or
nonpositive/ill-conditioned information. Bootstrap remains available unchanged:

```bash
python3 est.py --draws 32 --starts 1 --se-method hessian --output results_single_person_hessian
python3 est.py --se-method bootstrap --bootstrap-reps 49 --output results_single_person_bootstrap
```

Both commands enforce the single-person purchase restriction. Existing saved
all-household results must be rerun; changing the code does not update them.


The summary script now restricts both full-panel and matched-yogurt descriptive
statistics to `household_size == 1` purchase records, before checking repeated
purchase weeks. This includes demographics, detailed/grouped flavor shares,
HHI figures, switching, and spells. Rerun `summ_stats.py` to regenerate them.
The saved prepared estimation parquet retains all-household records for retail
price menus and instruments; `est.py` applies the single-person demand filter.


### Comparing single-normal and mixture gamma

The existing `heterogeneous` model and `est_het_gamma.py` are unchanged.
Use `--model mixture` (or `est_mixture_gamma.py`) for a two-normal gamma
mixture with the same shared lognormal price sensitivity, independent of gamma
and component membership. Membership is fixed over each household history.
The mixture adds three parameters relative to the single normal:
`mean_gamma` is component 1's mean, component 2's mean is
`mean_gamma + gamma_mean_gap`, `sd_gamma` and `sd_gamma_2` are component SDs,
and `mixture_weight` is component 1's population share. Means are ordered to
prevent label switching. The mean gap is bounded to [0, 40], SDs to [0, 3],
and the share to [0.001, 0.999]. Equal means and SDs recover the single normal.

Run both with identical sample, capacity, price controls, history settings,
draw count and seed, saving to separate output directories. For example:

```sh
python3 est.py --model heterogeneous --starts 10 --output normal_output
python3 est.py --model mixture --starts 10 --output mixture_output
```

`--draws` is the number of integration draws **per component**; the mixture
therefore costs about twice as much per likelihood evaluation. Use multiple
starts and check convergence, bounds, and stability with more draws before
interpreting differences. `gamma_distribution.pdf` shows the mixture and its
weighted components; zero-SD components are displayed as point masses.
All fits now include `parameters`, `aic`, and `bic` in `fit.json`; BIC uses the
number of independent households. Compare these only on identical observations.
Lower AIC/BIC favors a model, but this is a sensitivity comparison, not a formal
normality test. The usual chi-squared likelihood-ratio reference is invalid for
this mixture comparison. No formal normality-test p-value is produced.
Hessian inference uses the existing conditioning and singularity checks;
near-coincident components can make component-level inference unreliable.


Every fit saves `implied_wtp.csv`, with mean WTP and population SD in the
same currency/unit as the input price. For any utility coefficient c,
WTP = c/(-alpha). Under the model's independence assumption,
E[WTP] = E[c] exp(-mu_log_minus_alpha + sd_log_minus_alpha^2/2).
This averages household ratios, rather than dividing by mean price sensitivity.
The population SD describes preference heterogeneity, not sampling uncertainty;
no WTP standard errors or confidence intervals are reported.
Beta rows are premiums relative to the omitted other flavor; the inside
constant is other versus outside at zero history/control residual. Gamma is
per unit Xi; novelty and own-stock rows multiply gamma WTP by (1-lambda)
and -lambda respectively. Mixture fits also include conditional component
moments and weight them for the overall population moments. Nonconverged or
boundary fits are flagged provisional.
