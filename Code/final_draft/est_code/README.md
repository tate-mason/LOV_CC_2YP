# Yogurt demand estimation

The main script is `est.py`. It uses the revised sample construction in
`../data_code`, the model discussion in `../../full_draft_1231/updates/draft_2/draft_1010.tex`,
and the weekly aggregation lessons in `../../full_draft_1231/EST_STEP4_HANDOFF.md`.
The current instructions supersede the older draft: beta is common across
households; gamma and alpha are heterogeneous; delta is one; own-flavor history
includes all earlier periods.

## Four model specifications

Beta is common across households in all four models. Each entry point accepts
all the same data, price-control, optimization, and bootstrap options:

| Entry point | `--model` | Gamma | Alpha |
| --- | --- | --- | --- |
| `est_naive.py` | `naive` | Absent | Common |
| `est_gamma.py` | `homogeneous` | Common, flavor-specific | Common |
| `est_het_naive.py` | `mixed_naive` | Absent | Varies across households |
| `est_het_gamma.py` | `heterogeneous` | Varies across households and flavors | Varies across households |

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
  --flavors /path/to/Latest_Flavor_2010.csv \
  --max-households 100 --draws 32 --starts 2 --se-method none \
  --output results_small
```

Then run the full sample, increasing draws if estimates change materially:

```bash
python3 est.py \
  --flavors /path/to/Latest_Flavor_2010.csv \
  --draws 128 --starts 5 --bootstrap-reps 199 --output results_full
```

Defaults point to `/scratch/dtm63837/Kilts_Panel/Data/`:

- `--sample`: `yogurt_estimation_sample.parquet` (matched HMS purchases).
- `--rms`: `merged_rms.parquet` (retail menus, including unchosen alternatives).
- Years: 2006–2015. Change with `--first-year` and `--last-year`.

Check the RMS path on the server: some upstream code writes it under `Data/RMS/`.
The flavor CSV needs `upc`, `flavor_descr`, and `product_descr`, and must cover
purchased UPCs **and** yogurt UPCs in the retained RMS stores.
For price controls, the panel also needs `dma_cd`, and RMS needs `dma_code`.
Override names with `--household-dma-column` and `--retail-dma-column`.
DMA codes must be positive integers; missing DMA codes stop processing. The script stops
on missing matches or conflicting flavor labels. A blank description is assigned
"other", following `summ_stats.py`; inspect the reference before using this rule.

Households must have at least two distinct usable purchase weeks after menu/IV
exclusions. Weeks need not be consecutive. The old single-person
household and serving-code filters are not carried forward. The development
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
V_ijt = beta_j + gamma_ij * Xi_ijt + alpha_i * price_jt + sigma * residual_jt
beta_other = 0
beta_berry, beta_plain are common across households

gamma_ij = mean_gamma_j + sd_gamma_j * z_ij
alpha_i  = -exp(mu_log_minus_alpha + sd_log_minus_alpha * z_i_alpha)
```

Four independent standard normal draws represent one price coefficient and
three flavor-specific history coefficients. Independence is a simplifying
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

Retail prices are `price / prmult`, then equal-weight averages across distinct
UPCs of each flavor at a store in that week. This corrects the quoted multiple
price but does not standardize package sizes. The exposure store is the household's
store with most purchased packages that week; ties use the lowest store code.
This is an exposure proxy, not an estimated store-choice model.

All three flavor prices must be observed; there is no price imputation. Weeks
with incomplete menus are excluded from the likelihood after their purchases
have contributed to future history. The script prints exclusion counts. RMS
movement is an availability proxy, not an observed complete shelf assortment.

The script recomputes Saturday weeks from purchase dates. `panel_week.py` can map
Sunday purchases to the preceding Saturday; because `master_dat.py` already
inner-joined using that variable, recomputing dates cannot recover purchases lost
upstream. Audit and rebuild that merge before treating the sample as final.
Duplicate RMS store/week/UPC rows stop estimation for upstream inspection.

## Other-DMA price control (on by default)

For each UPC and week, average valid `price/prmult` across stores in each DMA.
The instrument for consumer DMA d is the equal-weight average of these DMA means
across **every available DMA other than d**. There is no selected-market list.
Exclude the consumer's DMA, even if their exposure store is in a different DMA.
If the consumer's DMA has no retail observation for that UPC/week, all observed
DMAs are eligible. At least one other DMA is required for each UPC.

Next average the UPC instruments within the exposure store's flavor assortment,
using the same equal UPC weights as the local flavor price. This avoids comparing
one local assortment with a different national assortment. If any UPC lacks an
other-DMA instrument, the entire flavor instrument is missing, and that week's
likelihood is excluded. Its purchases still contribute to history. Retail prices
come from all DMAs, including stores never visited by selected households.

Estimate three separate OLS first stages on the retained household-week exposures:

```text
price_ijt = pi_j * instrument_ijt + coefficients_j * (Xi_other, Xi_berry, Xi_plain)
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

The default is 199 household-cluster bootstrap replications. Each replication
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
