# est_step4.py: code and mathematical guide

This file extends the organization of `est_step2.py`. It does not import or modify that file. All estimates and diagnostics go to `step4_output/` relative to the working directory. Importing the module does not read data or create output directories.

## Choices made for this version

- One observation is one household-week, combining purchases across all stores.
- Only weeks containing purchases in the selected yogurt sample enter the real-data estimation. Unobserved weeks are not automatically classified as outside-only weeks. Calendar gaps still decay stock.
- Initial own-flavor stock is exactly zero, as requested.
- Stock inputs are flavor shares among inside purchases.
- Outside count is `max(7 - Q, 0)`, with actual inside counts retained. Thus three yogurts imply four outside choices; seven imply zero; eight imply zero. Total modeled counts are `max(7, Q)`. Seven is an outside threshold, **not a hard cap on choice occasions**. This is a working likelihood conditional on the constructed totals, not a model of why purchases exceed seven. It cannot simultaneously represent at most seven consumption occasions and eight observed consumption choices. Check whether quantity denotes individual yogurts or packages in your source data.
- Three mean gamma coefficients are estimated, including “other.” Only the baseline flavor intercept is normalized.
- Gamma signs are unrestricted. A negative own-flavor gamma is consistent with satiation; the code does not enforce this conclusion.

## Running and switches

Run `python3 est_step4.py` on the machine holding `MERGED_PATH`. Change that constant if the parquet lives elsewhere. The file retains step2's household-size, product-module, and serving-code sample restrictions, flavor classification, and first-stage price control function.

`RUN_WITHIN_FIXED_EFFECTS`, `RUN_WITHIN_DIAGNOSTICS`, `RUN_HOMOGENEOUS`, `RUN_RANDOM_BETA`, and `RUN_RANDOM_BETA_GAMMA` default to true. The three model switches work independently; enabled stages use the last enabled fit as a starting point. `RUN_PROFILES`, `RUN_RECOVERY`, and `RUN_DRAW_SENSITIVITY` default to false because they trigger many extra fits. `COMPUTE_STANDARD_ERRORS` controls numerical curvature and household sandwich errors. `USE_CONTROL_FUNCTION=False` drops its coefficient rather than leaving an unidentified zero regressor.

`N_DRAWS`, `N_STARTS`, and `MAX_ITER` control simulation and optimization. Set `MAX_HOUSEHOLDS` for development; this selects households in sorted identifier order and is not a random representative subsample. `OUTSIDE_RULE="strict"` stops if quantities exceed seven; it never silently drops those weeks. Output files overwrite files with the same names on reruns.

`python3 est_step4.py --simulate-only` runs optional parameter recovery without accessing the parquet. This still performs expensive optimization; adjust the recovery constants first.

## Stock: exactly what enters utility

Let $q_{hjt}$ be flavor counts, $Q_{ht}=\sum_jq_{hjt}$, and $x_{hjt}=q_{hjt}/Q_{ht}$ when $Q_{ht}>0$, otherwise zero. Let $d_t$ be calendar weeks since the previous observed week. The pre-choice stock is

$$S_{hj1}=0,\qquad S_{hjt}=\delta^{d_t}(S_{hj,t-1}+x_{hj,t-1}),\quad \delta=0.9.$$

This discounts purchases once rather than repeatedly adding past stocks. Repeating one flavor every week produces limiting stock $\delta/(1-\delta)=9$, rather than explosive growth. Current purchases never enter current stock. Stock is held fixed across all counts within the current week; this is an inter-week model, not a sequential within-week satiation model.

For irregular gaps, the previous share is treated as arriving at the previous observation and decays across the whole gap. Unknown purchases during missing weeks are not filled in. Zero initial stock is an assumption about unobserved prehistory; initial weeks can strongly influence slope estimates. Sensitivity to excluding early likelihood contributions would be a useful later extension, while retaining those purchases in stock construction.

Define $z_{hjt}=\log(1+S_{hjt})-c_j$, where $c_j$ is the pooled household-week mean of the log stock. There is no standardization by the stock's potentially distorted sample standard deviation. Global centering changes parameter coordinates, not identifying information.

## Utility, beta, gamma, and normalization

Outside utility is zero. Mean inside utilities are

$$v_{hjt}=a+b_j+g_j z_{hjt}+\alpha p_{hjt}+\kappa r_{hjt},\qquad b_{other}=0.$$

The three mean zero-stock intercepts are $a-g_{other}c_{other}$, $a+b_{berry}-g_{berry}c_{berry}$, and $a+b_{plain}-g_{plain}c_{plain}$. Consequently the zero-stock berry preference relative to other is $b_{berry}-g_{berry}c_{berry}+g_{other}c_{other}$, with an analogous formula for plain. Reported betas refer to each flavor's centered stock reference; compare zero-stock coefficients only after this conversion.

The homogeneous model has fixed $b,g$. The random-beta model adds independent normal deviations to berry/plain intercepts. The full model also adds independent normal deviations to the three gamma coefficients. Random deviations are anchored at **zero stock**, so the complete full-model utility is

$$v_{hjt}^{(r)}=a+b_j+g_jz_{hjt}+\eta_{hj}^{(r)}+\xi_{hj}^{(r)}(z_{hjt}+c_j)+\alpha p_{hjt}+\kappa r_{hjt}.$$

Here $\eta_{other}=0$, $\eta_{berry}=s_{b,berry}\epsilon_1$, $\eta_{plain}=s_{b,plain}\epsilon_2$, and $\xi_j=s_{g,j}\epsilon_{j+3}$, with independent standard normal draws. The `beta += deviations * reference` line preserves this anchoring when globally centering. Without it, centering would inadvertently change the assumed random-coefficient distribution. Standard deviations are fitted and reported directly, not exponentiated. Bounds include zero and an upper bound of five; hitting either is reported.

Price is constrained to be nonpositive. There is no automatic positive willingness-to-pay interpretation if price is at zero. For a utility improvement $\Delta v$, compensating price willingness to pay is $-\Delta v/\alpha$ when $\alpha<0$. This version deliberately reports utility parameters rather than step2's inconsistent WTP signs.

## Household integration and counts

Conditional probabilities use log-sum-exp for numerical stability:

$$\log P_{hjt}^{(r)}=v_{hjt}^{(r)}-\log\sum_k\exp(v_{hkt}^{(r)}).$$

The household likelihood integrates **after** summing all its weeks and count-weighted log probabilities:

$$\ell_h=\operatorname{LSE}_r\left[\sum_{t,j}n_{hjt}\log P_{hjt}^{(r)}\right]-\log R.$$

The objective minimizes $-\sum_h\ell_h$. A household's type remains fixed across its weeks. Multinomial combinatorial constants are omitted because they do not depend on parameters; objective comparisons require identical observations and counts. Draws are fixed across evaluations and shared as integration nodes across households, which does not impose a shared realized household type. Antithetic normal draws pair each draw with its negative. Prefer even `N_DRAWS` for balanced pairs.

## What “within household” means here

The stock varies within household, and persistent random tastes are integrated over the complete household history. The diagnostics explicitly demean states within each household, then measure how much state variation survives household intercepts, calendar-week effects, prices, and control residuals. `within_household.csv` records state SDs, ranges, and changes in observed flavor shares. `residual_state_variation.csv` records residual state SDs by flavor. Zero residual variation is a warning that these controls can explain all state variation; nonzero variation is necessary evidence, not proof of structural identification.

**The three homogeneous/random-coefficient stages are not household fixed-effect estimators.** Global centering does not eliminate between-household information, and a random-intercept distribution is not equivalent to removing unrestricted household preferences. Therefore this implementation does not claim that gamma is identified exclusively by within-household variation. Household fixed effects or correlated random effects would be an additional specification, with incidental-parameter and initial-condition questions for short panels. A price control function also does not instrument lagged choices or automatically resolve state endogeneity.

## Data construction and retained limitations

Across-store counts are summed before packing the household panel; no household-week outcome is repeated for each store. The reference store is the one with most selected yogurt units that week; ties use store code order. This is a transparent price-exposure approximation, not a structural model of store choice. Store-week flavor prices fall back to week-flavor prices, then overall flavor means. The diagnostic reports the fallback fraction. A missing category price is not evidence that the category was unavailable.

The upstream price construction is intentionally familiar from step2: transaction-based leave-other-market prices, a first-stage OLS with numeric `brand_cd` and week effects, standardized residuals, and missing residuals replaced by zero. It is not a newly validated IV design. Transaction averages can reflect selected product mix, numeric brand codes impose an artificial linear ordering, and missing-residual zero filling is an assumption. Revisit those separately before a structural interpretation. The code validates final numeric inputs, but has not established the instrument's relevance or exclusion restriction.

Because the real estimation includes only purchase weeks, the outside option does not identify unconditional participation across all calendar weeks. Reconstructing a reliable household observation calendar would be needed for that extension. The dynamic simulation includes outside-only weeks; it validates the likelihood on a complete simulated calendar, not purchase-week selection or the above-seven recording rule.

## Optimization, uncertainty, and optional checks

Multiple starts perturb shared starting values, and bounds are applied before optimization. The best successful fit is selected; if none succeeds, the best attempted fit is returned with failure status. Always inspect convergence, start objectives, and boundary flags. L-BFGS-B success is not proof of a global optimum.

The numerical Hessian $H$ is the curvature of negative log likelihood. Household score rows are $s_h=\partial\ell_h/\partial\theta$. The reported conditional household sandwich is

$$\widehat V=H^{-1}\left(\sum_hs_hs_h'\right)H^{-1}.$$

Errors are left as NaN for boundary solutions or nonpositive Hessian eigenvalues. The script does not disguise negative curvature through a pseudoinverse. These errors condition on the generated first-stage residuals and fixed simulation draws; a household bootstrap that reestimates the first stage would be needed for fuller uncertainty. Normal tests of a zero SD are nonstandard, so no automatic SD p-values are printed. Numerical derivatives use SciPy's private `_numdiff` helper, which can require adjustment after SciPy upgrades.

Profiles fix each gamma on a grid and reoptimize every other coefficient, including beta. `gamma_profiles.csv` records the optimized objective and compensating betas; flat profiles or large beta changes with little objective change reveal weak separation. A negative objective difference indicates the profile found an improvement over the nominal optimum. The profile grid is in utility-per-log-stock units and can be adjusted with `PROFILE_WIDTH`.

Draw sensitivity refits using a new seed and twice as many draws. Recovery generates persistent household types, exogenous prices, choices, and updated endogenous stocks sequentially. It centers the simulated stocks and transforms the true mean intercepts to match. `recovery.csv` records truth, estimate, error, and convergence for each replication. Three replications are a smoke diagnostic, not a reliable Monte Carlo study; increase them to assess bias and dispersion.

## Direct within-household benchmark

`RUN_WITHIN_FIXED_EFFECTS=True` fits an additional benchmark before the three main stages. It replaces all common flavor intercepts by unrestricted household-by-flavor intercepts:

$$v_{hjt}=a_{hj}+g_jz_{hjt}+\alpha p_{hjt}+\kappa r_{hjt}.$$

Because $a_{hj}$ absorbs every time-invariant household flavor preference, the common slopes rely on changes within households. The three inside intercepts remain meaningful relative to outside utility zero. This benchmark does not recover a population beta distribution. Centering is immaterial to its fit because the intercepts absorb the shifts.

For fitted probabilities $P$ and count total $N$, the negative-log-likelihood utility derivative is $e_{hjt}=N_{ht}P_{hjt}-n_{hjt}$. The analytic gradient sums $e z$ for gamma, $e p$ for price, $e r$ for the control function, and $e$ for each household-flavor intercept. The code passes this gradient to L-BFGS-B, avoiding numerical differentiation across thousands of intercepts.

The CSV reports common slopes, and the JSON reports convergence. No fixed-effect standard errors are claimed. Short-panel incidental-parameter bias can be material, and always-zero flavor or outside counts can drive intercepts to infinity. Interpret this as a sensitivity benchmark alongside the main stages; inspect convergence and compare slope signs/magnitudes. Neither fixed effects nor panel integration resolves time-varying unobserved demand shocks correlated with histories.

`state_reference.csv` saves the log-stock centering constants required to convert reported mean coefficients back to zero-stock units.

## Line-by-line statement index

The index below covers every executable statement, function header, and control-flow header. Multiline expressions are explained as one statement with a line interval. Blank lines and standalone comments are omitted; comments immediately above statements provide additional explanations in the source. Source line numbers refer to this delivered version.

| Lines | Code | Explanation |
| --- | --- | --- |
| 2 | `from pathlib import Path` | Import the named dependency used in this part of the implementation. |
| 3 | `import argparse` | Import the named dependency used in this part of the implementation. |
| 4 | `import json` | Import the named dependency used in this part of the implementation. |
| 5 | `import numpy as np` | Import the named dependency used in this part of the implementation. |
| 6 | `import pandas as pd` | Import the named dependency used in this part of the implementation. |
| 7 | `import polars as pl` | Import the named dependency used in this part of the implementation. |
| 8 | `import scipy as sp` | Import the named dependency used in this part of the implementation. |
| 9 | `from scipy.optimize import minimize` | Import the named dependency used in this part of the implementation. |
| 10 | `from scipy.special import logsumexp` | Import the named dependency used in this part of the implementation. |
| 11 | `from rich.console import Console` | Import the named dependency used in this part of the implementation. |
| 12 | `from rich.table import Table` | Import the named dependency used in this part of the implementation. |
| 13 | `import statsmodels.formula.api as smf` | Import the named dependency used in this part of the implementation. |
| 15 | `console = Console()` | Compute or initialize `console` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 16 | `MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"` | Compute or initialize `MERGED_PATH` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 17 | `OUTPUT_DIR = Path("step4_output")` | Compute or initialize `OUTPUT_DIR` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 18 | `CATEGORIES = ["other", "berry", "plain", "outside"]` | Compute or initialize `CATEGORIES` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 19 | `cat_map = {name: j for j, name in enumerate(CATEGORIES)}` | Compute or initialize `cat_map` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 22 | `RUN_WITHIN_DIAGNOSTICS = True` | Compute or initialize `RUN_WITHIN_DIAGNOSTICS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 23 | `RUN_WITHIN_FIXED_EFFECTS = True` | Compute or initialize `RUN_WITHIN_FIXED_EFFECTS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 24 | `RUN_HOMOGENEOUS = True` | Compute or initialize `RUN_HOMOGENEOUS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 25 | `RUN_RANDOM_BETA = True` | Compute or initialize `RUN_RANDOM_BETA` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 26 | `RUN_RANDOM_BETA_GAMMA = True` | Compute or initialize `RUN_RANDOM_BETA_GAMMA` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 27 | `RUN_PROFILES = False` | Compute or initialize `RUN_PROFILES` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 28 | `RUN_RECOVERY = False` | Compute or initialize `RUN_RECOVERY` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 29 | `RUN_DRAW_SENSITIVITY = False` | Compute or initialize `RUN_DRAW_SENSITIVITY` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 30 | `COMPUTE_STANDARD_ERRORS = True` | Compute or initialize `COMPUTE_STANDARD_ERRORS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 31 | `USE_CONTROL_FUNCTION = True` | Compute or initialize `USE_CONTROL_FUNCTION` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 34 | `WEEKLY_CAPACITY = 7` | Compute or initialize `WEEKLY_CAPACITY` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 35 | `OUTSIDE_RULE = "threshold"  # threshold: max(7-Q, 0); strict: reject Q > 7` | Compute or initialize `OUTSIDE_RULE` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 36 | `DELTA_DISCOUNT = 0.9` | Compute or initialize `DELTA_DISCOUNT` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 37 | `MIN_WEEKS = 2` | Compute or initialize `MIN_WEEKS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 38 | `N_DRAWS = 128` | Compute or initialize `N_DRAWS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 39 | `N_STARTS = 3` | Compute or initialize `N_STARTS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 40 | `MAX_ITER = 600` | Compute or initialize `MAX_ITER` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 41 | `SEED = 306` | Compute or initialize `SEED` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 42 | `MAX_HOUSEHOLDS = None  # Set an integer for a reproducible development sample.` | Compute or initialize `MAX_HOUSEHOLDS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 43 | `PROFILE_POINTS = 9` | Compute or initialize `PROFILE_POINTS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 44 | `PROFILE_WIDTH = 1.0` | Compute or initialize `PROFILE_WIDTH` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 45 | `RECOVERY_HOUSEHOLDS = 60` | Compute or initialize `RECOVERY_HOUSEHOLDS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 46 | `RECOVERY_WEEKS = 24` | Compute or initialize `RECOVERY_WEEKS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 47 | `RECOVERY_REPLICATIONS = 3` | Compute or initialize `RECOVERY_REPLICATIONS` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 50 | `def make_draws(n_draws=N_DRAWS, seed=SEED):` | Generate fixed normal integration nodes paired with their negatives. |
| 52 | `half = np.random.default_rng(seed).standard_normal(((n_draws + 1) // 2, 5))` | Compute or initialize `half` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 53 | `return np.vstack([half, -half])[:n_draws]` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 59 | `def load_and_preprocess():` | Construct the selected purchase-week household panel, prices, count vectors, and histories. |
| 60 | `schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()` | Compute or initialize `schema_names` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 62–69 | `exprs = [ pl.col("quantity").cast(pl.Float64), pl.col("head_age").cast(pl.Int64), pl.col("household_income").cast(pl.Int64), pl.col("household_size").cast(pl.Int64), pl.col("serving_per_container_cd").cast(pl.Int64), pl.col("price").cast(pl.Float64), ]` | Compute or initialize `exprs` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 71 | `if "product_module_code_hms" in schema_names:` | Execute this branch only when the displayed condition holds. |
| 72 | `exprs.append(pl.col("product_module_code_hms").cast(pl.Int64))` | Execute this call to print, save, validate, or update the object shown. |
| 74 | `scan_q = pl.scan_parquet(MERGED_PATH).with_columns(exprs)` | Compute or initialize `scan_q` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 76 | `if "product_module_code_hms" in schema_names:` | Execute this branch only when the displayed condition holds. |
| 77 | `scan_q = scan_q.filter(pl.col("product_module_code_hms").is_in([3612, 3603]))` | Compute or initialize `scan_q` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 79–84 | `merged_df = ( scan_q.filter(pl.col("household_size") == 1) .filter(pl.col("serving_per_container_cd").is_in([67181961, 65622705])) .collect() .to_pandas() )` | Compute or initialize `merged_df` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 86 | `merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)` | Compute or initialize `merged_df['flavor_str']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 87–89 | `merged_df["flavor_cd"] = pd.to_numeric( merged_df["flavor_cd"], errors="coerce" ).fillna(0)` | Compute or initialize `merged_df['flavor_cd']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 91 | `merged_master = merged_df.copy()` | Compute or initialize `merged_master` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 93–97 | `descr_cols = [ c for c in merged_df.columns if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"]) ]` | Compute or initialize `descr_cols` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 99 | `merged_df["full_text"] = ""` | Compute or initialize `merged_df['full_text']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 100 | `for c in descr_cols:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 101 | `merged_df["full_text"] += " " + merged_df[c].fillna("").astype(str)` | Update `merged_df['full_text']` using the displayed increment; this accumulates rather than replaces its existing value. |
| 102 | `merged_df["full_text"] = merged_df["full_text"].str.lower()` | Compute or initialize `merged_df['full_text']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 104 | `berry_regex = r"berry\|straw\|blue\|rasp\|black\|cran\|cherry\|wildberry"` | Compute or initialize `berry_regex` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 105 | `plain_regex = r"plain\|unflavored"` | Compute or initialize `plain_regex` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 107 | `is_plain = merged_df["full_text"].str.contains(plain_regex, na=False)` | Compute or initialize `is_plain` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 108 | `is_berry = merged_df["full_text"].str.contains(berry_regex, na=False) & (~is_plain)` | Compute or initialize `is_berry` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 110 | `merged_master["flavor"] = np.select([is_plain, is_berry], [2, 1], default=0)` | Compute or initialize `merged_master['flavor']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 112–116 | `market_price = ( merged_master.groupby(["upc", "week_end", "market_name"])["price"] .mean() .reset_index() )` | Compute or initialize `market_price` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 117–122 | `totals = ( market_price.groupby(["upc", "week_end"])["price"] .agg(["sum", "count"]) .reset_index() .rename(columns={"sum": "price_sum_all", "count": "n_markets_all"}) )` | Compute or initialize `totals` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 124 | `market_price = market_price.merge(totals, on=["upc", "week_end"], how="left")` | Compute or initialize `market_price` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 125–127 | `market_price["price_iv"] = ( market_price["price_sum_all"] - market_price["price"] ) / (market_price["n_markets_all"] - 1)` | Compute or initialize `market_price['price_iv']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 128–130 | `market_price["price_iv"] = market_price["price_iv"].replace( [np.inf, -np.inf], np.nan )` | Compute or initialize `market_price['price_iv']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 132–136 | `merged_master = merged_master.merge( market_price[["upc", "week_end", "market_name", "price_iv"]], on=["upc", "week_end", "market_name"], how="left", )` | Compute or initialize `merged_master` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 138–140 | `merged_master = merged_master[ (merged_master["price"] > 0.1) \| (merged_master["price"].isna()) ]` | Compute or initialize `merged_master` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 141–143 | `iv_res = smf.ols( "price ~ price_iv + brand_cd + C(week_end)", data=merged_master, missing="drop" ).fit()` | Compute or initialize `iv_res` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 145 | `merged_master["iv_res"] = np.nan` | Compute or initialize `merged_master['iv_res']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 146 | `merged_master.loc[iv_res.model.data.row_labels, "iv_res"] = iv_res.resid` | Compute or initialize `merged_master.loc[iv_res.model.data.row_labels, 'iv_res']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 148 | `valid_resids = merged_master["iv_res"].dropna()` | Compute or initialize `valid_resids` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 149 | `if len(valid_resids) > 0:` | Execute this branch only when the displayed condition holds. |
| 150 | `res_mean = valid_resids.mean()` | Compute or initialize `res_mean` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 151 | `res_std = valid_resids.std()` | Compute or initialize `res_std` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 152–154 | `merged_master["iv_res"] = (merged_master["iv_res"] - res_mean) / ( res_std if res_std > 0 else 1.0 )` | Compute or initialize `merged_master['iv_res']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 156 | `merged_master["iv_res"] = merged_master["iv_res"].fillna(0.0)` | Compute or initialize `merged_master['iv_res']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 158 | `def map_flavor_category(flavor):` | Define this local helper for the enclosing calculation. |
| 159 | `if pd.isna(flavor):` | Execute this branch only when the displayed condition holds. |
| 160 | `return "outside"` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 161 | `elif flavor == 0:` | Execute this branch only when the displayed condition holds. |
| 162 | `return "other"` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 163 | `elif flavor == 1:` | Execute this branch only when the displayed condition holds. |
| 164 | `return "berry"` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 165 | `elif flavor == 2:` | Execute this branch only when the displayed condition holds. |
| 166 | `return "plain"` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 168 | `merged_master["category"] = merged_master["flavor"].map(map_flavor_category)` | Compute or initialize `merged_master['category']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 169 | `inside_df = merged_master[merged_master["category"] != "outside"].copy()` | Compute or initialize `inside_df` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 171 | `quantities = inside_df["quantity"].to_numpy(dtype=float)` | Compute or initialize `quantities` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 172 | `if not np.isfinite(quantities).all() or np.any(quantities < 0) or not np.allclose(quantities, np.round(quantities)):` | Execute this branch only when the displayed condition holds. |
| 173 | `raise ValueError("Source quantities must be finite nonnegative integer yogurt units.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 174 | `inside_df = inside_df.loc[inside_df["quantity"] > 0].copy()` | Compute or initialize `inside_df` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 176 | `weekly = inside_df.groupby(["household_code", "week_end", "category"])["quantity"].sum().unstack("category", fill_value=0)` | Compute or initialize `weekly` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 177 | `weekly = weekly.reindex(columns=CATEGORIES[:3], fill_value=0).reset_index()` | Compute or initialize `weekly` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 178 | `weekly["week_end"] = pd.to_datetime(weekly["week_end"])` | Compute or initialize `weekly['week_end']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 181 | `stores = inside_df.groupby(["household_code", "week_end", "store_code_uc"], as_index=False)["quantity"].sum()` | Compute or initialize `stores` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 182 | `stores["week_end"] = pd.to_datetime(stores["week_end"])` | Compute or initialize `stores['week_end']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 183 | `stores = stores.sort_values(["household_code", "week_end", "quantity", "store_code_uc"], ascending=[True, True, False, True])` | Compute or initialize `stores` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 184 | `stores = stores.drop_duplicates(["household_code", "week_end"])` | Compute or initialize `stores` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 185 | `weekly = weekly.merge(stores[["household_code", "week_end", "store_code_uc"]], validate="one_to_one")` | Compute or initialize `weekly` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 188 | `inside_df["week_end"] = pd.to_datetime(inside_df["week_end"])` | Compute or initialize `inside_df['week_end']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 189 | `store_prices = inside_df.groupby(["store_code_uc", "week_end", "category"])[["price", "iv_res"]].mean()` | Compute or initialize `store_prices` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 190 | `week_prices = inside_df.groupby(["week_end", "category"])[["price", "iv_res"]].mean()` | Compute or initialize `week_prices` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 191 | `overall = inside_df.groupby("category")[["price", "iv_res"]].mean()` | Compute or initialize `overall` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 192 | `households = []` | Compute or initialize `households` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 193 | `for hh_id, group in weekly.groupby("household_code", sort=True):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 194 | `group = group.sort_values("week_end")` | Compute or initialize `group` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 195 | `if len(group) < MIN_WEEKS:` | Execute this branch only when the displayed condition holds. |
| 196 | `continue` | Skip this household because it does not meet the minimum panel length. |
| 197 | `prices, resids, counts, dates, fallback = [], [], [], [], []` | Compute or initialize `(prices, resids, counts, dates, fallback)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 198 | `for row in group.itertuples(index=False):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 199 | `q = np.array([row.other, row.berry, row.plain], dtype=float)` | Compute or initialize `q` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 200 | `if np.any(q < 0) or not np.allclose(q, np.round(q)):` | Execute this branch only when the displayed condition holds. |
| 201 | `raise ValueError("Quantities must be nonnegative integer yogurt units.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 202 | `if OUTSIDE_RULE == "strict" and q.sum() > WEEKLY_CAPACITY:` | Execute this branch only when the displayed condition holds. |
| 203 | `raise ValueError(f"HH {hh_id}: weekly units exceed capacity; inspect units or use threshold rule.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 204 | `if OUTSIDE_RULE not in ("strict", "threshold"):` | Execute this branch only when the displayed condition holds. |
| 205 | `raise ValueError("Unknown OUTSIDE_RULE")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 206 | `counts.append(np.append(q, max(WEEKLY_CAPACITY - q.sum(), 0)))` | Execute this call to print, save, validate, or update the object shown. |
| 207 | `p, r, flags = [], [], []` | Compute or initialize `(p, r, flags)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 208 | `for cat in CATEGORIES[:3]:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 209 | `key = (row.store_code_uc, row.week_end, cat)` | Compute or initialize `key` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 210 | `wk_key = (row.week_end, cat)` | Compute or initialize `wk_key` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 211 | `if key in store_prices.index:` | Execute this branch only when the displayed condition holds. |
| 212 | `values, flag = store_prices.loc[key], 0` | Compute or initialize `(values, flag)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 213 | `elif wk_key in week_prices.index:` | Execute this branch only when the displayed condition holds. |
| 214 | `values, flag = week_prices.loc[wk_key], 1` | Compute or initialize `(values, flag)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 215 | `elif cat in overall.index:` | Execute this branch only when the displayed condition holds. |
| 216 | `values, flag = overall.loc[cat], 2` | Compute or initialize `(values, flag)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 218 | `raise ValueError(f"No price observations for {cat}")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 219 | `p.append(values["price"])` | Execute this call to print, save, validate, or update the object shown. |
| 220 | `r.append(values["iv_res"] if USE_CONTROL_FUNCTION else 0.0)` | Execute this call to print, save, validate, or update the object shown. |
| 221 | `flags.append(flag)` | Execute this call to print, save, validate, or update the object shown. |
| 222 | `prices.append(p)` | Execute this call to print, save, validate, or update the object shown. |
| 223 | `resids.append(r)` | Execute this call to print, save, validate, or update the object shown. |
| 224 | `dates.append(row.week_end)` | Execute this call to print, save, validate, or update the object shown. |
| 225 | `fallback.append(flags)` | Execute this call to print, save, validate, or update the object shown. |
| 226 | `choices = np.asarray(counts)` | Compute or initialize `choices` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 227 | `stock = compute_satiation_state(choices[:, :3], dates)` | Compute or initialize `stock` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 228–230 | `households.append(dict(household_code=str(hh_id), prices=np.asarray(prices), resids=np.asarray(resids), choices=choices, dates=pd.DatetimeIndex(dates), stock=stock, fallback=np.asarray(fallback)))` | Execute this call to print, save, validate, or update the object shown. |
| 231 | `if MAX_HOUSEHOLDS is not None and len(households) >= MAX_HOUSEHOLDS:` | Execute this branch only when the displayed condition holds. |
| 232 | `break` | Stop collecting households at the configured development-sample limit. |
| 233 | `if not households:` | Execute this branch only when the displayed condition holds. |
| 234 | `raise ValueError("No eligible households after preprocessing.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 235 | `for hh in households:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 236 | `if not np.isfinite(hh["prices"]).all() or not np.isfinite(hh["resids"]).all():` | Execute this branch only when the displayed condition holds. |
| 237 | `raise ValueError("Nonfinite price/control-function inputs.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 238 | `center_states(households)` | Execute this call to print, save, validate, or update the object shown. |
| 239 | `return households` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 242 | `def compute_satiation_state(inside_counts, dates):` | Build strictly lagged, calendar-decayed own-flavor shares with zero initial stock. |
| 244 | `stock = np.zeros_like(inside_counts, dtype=float)` | Compute or initialize `stock` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 245 | `for t in range(1, len(dates)):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 246 | `elapsed = (pd.Timestamp(dates[t]) - pd.Timestamp(dates[t - 1])).days / 7.0` | Compute or initialize `elapsed` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 247 | `if elapsed <= 0:` | Execute this branch only when the displayed condition holds. |
| 248 | `raise ValueError("Household dates must be unique and increasing.")` | Stop with an explicit input-validation error instead of silently changing the sample or model. |
| 249 | `previous = inside_counts[t - 1]` | Compute or initialize `previous` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 250 | `shares = previous / previous.sum() if previous.sum() > 0 else np.zeros(3)` | Compute or initialize `shares` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 251 | `stock[t] = DELTA_DISCOUNT ** elapsed * (stock[t - 1] + shares)` | Compute or initialize `stock[t]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 252 | `return stock` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 255 | `def center_states(households, reference=None):` | Compute log stocks and subtract a common flavor-specific reference. |
| 257 | `if reference is None:` | Execute this branch only when the displayed condition holds. |
| 258 | `reference = np.concatenate([np.log1p(h["stock"]) for h in households]).mean(axis=0)` | Compute or initialize `reference` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 259 | `for hh in households:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 260 | `hh["z"] = np.log1p(hh["stock"]) - reference` | Compute or initialize `hh['z']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 261 | `hh["reference"] = np.asarray(reference)` | Compute or initialize `hh['reference']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 262 | `return np.asarray(reference)` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 268 | `def diagnose_within(households, output_dir):` | Measure usable within-household history variation and report construction diagnostics. |
| 269 | `rows = []` | Compute or initialize `rows` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 270 | `for hh in households:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 271 | `z = hh["z"]` | Compute or initialize `z` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 272 | `shares = hh["choices"][:, :3] / hh["choices"][:, :3].sum(axis=1, keepdims=True)` | Compute or initialize `shares` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 274 | `for j, cat in enumerate(CATEGORIES[:3]):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 275–277 | `rows.append(dict(household=hh["household_code"], flavor=cat, weeks=len(z), within_sd=float(z[:, j].std()), state_range=float(np.ptp(z[:, j])), share_changes=int(np.count_nonzero(np.diff(shares[:, j])))))` | Execute this call to print, save, validate, or update the object shown. |
| 278 | `frame = pd.DataFrame(rows)` | Compute or initialize `frame` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 279 | `frame.to_csv(output_dir / "within_household.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 280 | `console.print(frame.groupby("flavor")[["within_sd", "state_range", "share_changes"]].agg(["mean", "median"]))` | Execute this call to print, save, validate, or update the object shown. |
| 282 | `dates = np.concatenate([h["dates"].asi8 for h in households])` | Compute or initialize `dates` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 283 | `_, week_idx = np.unique(dates, return_inverse=True)` | Compute or initialize `(_, week_idx)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 284 | `blocks = []` | Compute or initialize `blocks` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 285 | `for j, cat in enumerate(CATEGORIES[:3]):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 286 | `target = np.concatenate([h["z"][:, j] for h in households])` | Compute or initialize `target` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 287 | `price = np.concatenate([h["prices"][:, j] for h in households])` | Compute or initialize `price` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 288 | `resid = np.concatenate([h["resids"][:, j] for h in households])` | Compute or initialize `resid` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 289 | `sizes = [len(h["z"]) for h in households]` | Compute or initialize `sizes` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 290 | `hh_idx = np.repeat(np.arange(len(sizes)), sizes)` | Compute or initialize `hh_idx` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 292 | `from scipy.sparse import coo_matrix, hstack` | Import the named dependency used in this part of the implementation. |
| 293 | `from scipy.sparse.linalg import lsqr` | Import the named dependency used in this part of the implementation. |
| 294 | `n = len(target)` | Compute or initialize `n` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 295 | `hd = coo_matrix((np.ones(n), (np.arange(n), hh_idx)), shape=(n, len(sizes))).tocsr()` | Compute or initialize `hd` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 296 | `wd = coo_matrix((np.ones(n), (np.arange(n), week_idx))).tocsr()` | Compute or initialize `wd` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 297 | `design = hstack([hd, wd, coo_matrix(np.column_stack([price, resid]))]).tocsr()` | Compute or initialize `design` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 298 | `residual = target - design @ lsqr(design, target, atol=1e-10, btol=1e-10)[0]` | Compute or initialize `residual` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 299 | `within = np.concatenate([h["z"][:, j] - h["z"][:, j].mean() for h in households])` | Compute or initialize `within` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 300 | `blocks.append(dict(flavor=cat, within_sd=float(within.std()), residual_sd=float(residual.std())))` | Execute this call to print, save, validate, or update the object shown. |
| 301 | `pd.DataFrame(blocks).to_csv(output_dir / "residual_state_variation.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 302 | `console.print(pd.DataFrame(blocks))` | Execute this call to print, save, validate, or update the object shown. |
| 303 | `all_choices = np.concatenate([h["choices"] for h in households])` | Compute or initialize `all_choices` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 304 | `console.print(f"Households: {len(households)}; purchase weeks: {len(all_choices)}; weeks above seven: {(all_choices[:, :3].sum(axis=1) > 7).sum()}")` | Execute this call to print, save, validate, or update the object shown. |
| 305 | `console.print(f"Price fallback fraction: {(np.concatenate([h['fallback'] for h in households]) > 0).mean():.3f}")` | Execute this call to print, save, validate, or update the object shown. |
| 311 | `def estimate_within_fixed_effects(households, output_dir):` | Define this local helper for the enclosing calculation. |
| 314 | `n_hh = len(households)` | Compute or initialize `n_hh` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 315 | `n_common = 5 if USE_CONTROL_FUNCTION else 4` | Compute or initialize `n_common` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 316 | `def objective_gradient(params):` | Define this local helper for the enclosing calculation. |
| 317 | `gamma, alpha = params[:3], params[3]` | Compute or initialize `(gamma, alpha)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 318 | `intercepts = params[n_common:].reshape(n_hh, 3)` | Compute or initialize `intercepts` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 319 | `gradient = np.zeros_like(params)` | Compute or initialize `gradient` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 320 | `value = 0.0` | Compute or initialize `value` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 321 | `for h, hh in enumerate(households):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 322 | `inside = intercepts[h] + gamma * hh["z"] + alpha * hh["prices"]` | Compute or initialize `inside` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 323 | `if USE_CONTROL_FUNCTION:` | Execute this branch only when the displayed condition holds. |
| 324 | `inside += params[4] * hh["resids"]` | Update `inside` using the displayed increment; this accumulates rather than replaces its existing value. |
| 325 | `u = np.column_stack([inside, np.zeros(len(inside))])` | Compute or initialize `u` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 326 | `lp = u - logsumexp(u, axis=1, keepdims=True)` | Compute or initialize `lp` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 327 | `value -= np.sum(hh["choices"] * lp)` | Update `value` using the displayed increment; this accumulates rather than replaces its existing value. |
| 328 | `error = (hh["choices"].sum(axis=1, keepdims=True) * np.exp(lp) - hh["choices"])[:, :3]` | Compute or initialize `error` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 329 | `gradient[:3] += np.sum(error * hh["z"], axis=0)` | Update `gradient[:3]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 330 | `gradient[3] += np.sum(error * hh["prices"])` | Update `gradient[3]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 331 | `if USE_CONTROL_FUNCTION:` | Execute this branch only when the displayed condition holds. |
| 332 | `gradient[4] += np.sum(error * hh["resids"])` | Update `gradient[4]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 333 | `gradient[n_common + 3*h:n_common + 3*h + 3] = error.sum(axis=0)` | Compute or initialize `gradient[n_common + 3 * h:n_common + 3 * h + 3]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 334 | `return float(value), gradient` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 335 | `initial = np.zeros(n_common + 3*n_hh)` | Compute or initialize `initial` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 336 | `initial[3] = -0.1` | Compute or initialize `initial[3]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 337 | `bounds = [(None, None)] * len(initial)` | Compute or initialize `bounds` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 338 | `bounds[3] = (None, 0.0)` | Compute or initialize `bounds[3]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 339–340 | `fit = minimize(objective_gradient, initial, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5))` | Compute or initialize `fit` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 341 | `names = ["gamma_other", "gamma_berry", "gamma_plain", "price"] + (["control_function"] if USE_CONTROL_FUNCTION else [])` | Compute or initialize `names` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 342 | `pd.DataFrame(dict(parameter=names, estimate=fit.x[:n_common])).to_csv(output_dir / "within_fixed_effects.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 343–344 | `metadata = dict(success=bool(fit.success), message=str(fit.message), fun=float(fit.fun), warning="Short-panel incidental-parameter bias and separated HH flavor intercepts are possible; no bias correction or SEs.")` | Compute or initialize `metadata` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 345 | `(output_dir / "within_fixed_effects.json").write_text(json.dumps(metadata, indent=2))` | Execute this call to print, save, validate, or update the object shown. |
| 346 | `console.print("Within-household fixed-effect benchmark", dict(zip(names, fit.x[:n_common])), metadata)` | Execute this call to print, save, validate, or update the object shown. |
| 347 | `return fit` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 350 | `def parameter_names(stage):` | Define the exact coefficient order for the selected stage. |
| 351 | `names = ["constant", "beta_berry", "beta_plain", "gamma_other", "gamma_berry", "gamma_plain", "price"]` | Compute or initialize `names` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 352 | `if USE_CONTROL_FUNCTION:` | Execute this branch only when the displayed condition holds. |
| 353 | `names += ["control_function"]` | Update `names` using the displayed increment; this accumulates rather than replaces its existing value. |
| 354 | `if stage in ("random_beta", "random_beta_gamma"):` | Execute this branch only when the displayed condition holds. |
| 355 | `names += ["sd_beta_berry", "sd_beta_plain"]` | Update `names` using the displayed increment; this accumulates rather than replaces its existing value. |
| 356 | `if stage == "random_beta_gamma":` | Execute this branch only when the displayed condition holds. |
| 357 | `names += ["sd_gamma_other", "sd_gamma_berry", "sd_gamma_plain"]` | Update `names` using the displayed increment; this accumulates rather than replaces its existing value. |
| 358 | `return names` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 361 | `def utility_draws(params, hh, stage, draws):` | Construct inside and outside utility for every week and integration draw. |
| 362 | `n = 1 if stage == "homogeneous" else len(draws)` | Compute or initialize `n` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 363 | `beta = np.tile([0.0, params[1], params[2]], (n, 1))` | Compute or initialize `beta` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 364 | `gamma = np.tile(params[3:6], (n, 1))` | Compute or initialize `gamma` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 365 | `offset = 8 if USE_CONTROL_FUNCTION else 7` | Compute or initialize `offset` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 366 | `if stage != "homogeneous":` | Execute this branch only when the displayed condition holds. |
| 367 | `beta[:, 1:] += draws[:, :2] * params[offset:offset + 2]` | Update `beta[:, 1:]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 368 | `if stage == "random_beta_gamma":` | Execute this branch only when the displayed condition holds. |
| 369 | `deviations = draws[:, 2:5] * params[offset + 2:offset + 5]` | Compute or initialize `deviations` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 370 | `gamma += deviations` | Update `gamma` using the displayed increment; this accumulates rather than replaces its existing value. |
| 372 | `beta += deviations * hh.get("reference", np.zeros(3))` | Update `beta` using the displayed increment; this accumulates rather than replaces its existing value. |
| 373 | `base = params[0] + params[6] * hh["prices"]` | Compute or initialize `base` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 374 | `if USE_CONTROL_FUNCTION:` | Execute this branch only when the displayed condition holds. |
| 375 | `base += params[7] * hh["resids"]` | Update `base` using the displayed increment; this accumulates rather than replaces its existing value. |
| 376 | `inside = base[:, None, :] + beta[None, :, :] + gamma[None, :, :] * hh["z"][:, None, :]` | Compute or initialize `inside` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 377 | `return np.concatenate([inside, np.zeros((len(inside), n, 1))], axis=2)` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 380 | `def household_loglikes(params, households, stage, draws):` | Aggregate the conditional panel likelihood before integrating household types. |
| 381 | `ll = []` | Compute or initialize `ll` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 382 | `for hh in households:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 383 | `u = utility_draws(params, hh, stage, draws)` | Compute or initialize `u` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 384 | `log_probs = u - logsumexp(u, axis=2, keepdims=True)` | Compute or initialize `log_probs` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 386 | `conditional = np.sum(hh["choices"][:, None, :] * log_probs, axis=(0, 2))` | Compute or initialize `conditional` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 387 | `ll.append(logsumexp(conditional) - np.log(len(conditional)))` | Execute this call to print, save, validate, or update the object shown. |
| 388 | `return np.asarray(ll)` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 391 | `def total_objective(params, households, stage, draws):` | Return the negative summed household log likelihood for minimization. |
| 392 | `value = -household_loglikes(params, households, stage, draws).sum()` | Compute or initialize `value` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 393 | `return float(value) if np.isfinite(value) else 1e100` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 396 | `def numerical_hessian(function, params):` | Numerically differentiate the objective twice to diagnose curvature. |
| 398 | `from scipy.optimize._numdiff import approx_derivative` | Import the named dependency used in this part of the implementation. |
| 399 | `gradient = lambda x: approx_derivative(lambda y: np.array([function(y)]), x, method="3-point").ravel()` | Compute or initialize `gradient` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 400 | `matrix = approx_derivative(gradient, params, method="3-point")` | Compute or initialize `matrix` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 401 | `return (matrix + matrix.T) / 2` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 404 | `def estimate_model(households, stage, draws, warm_start=None, starts=N_STARTS, compute_se=COMPUTE_STANDARD_ERRORS):` | Fit a stage from multiple starts and calculate conditional household sandwich uncertainty. |
| 405 | `names = parameter_names(stage)` | Compute or initialize `names` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 406 | `x0 = np.zeros(len(names))` | Compute or initialize `x0` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 407 | `x0[3:6] = -0.1` | Compute or initialize `x0[3:6]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 408 | `x0[6] = -0.1` | Compute or initialize `x0[6]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 409 | `for i, name in enumerate(names):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 410 | `if name.startswith("sd_"):` | Execute this branch only when the displayed condition holds. |
| 411 | `x0[i] = 0.2` | Compute or initialize `x0[i]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 412 | `if warm_start is not None and name in warm_start["names"]:` | Execute this branch only when the displayed condition holds. |
| 413 | `x0[i] = warm_start["params"][warm_start["names"].index(name)]` | Compute or initialize `x0[i]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 415 | `bounds = [(0.0, 5.0) if name.startswith("sd_") else (None, 0.0) if name == "price" else (None, None) for name in names]` | Compute or initialize `bounds` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 416 | `objective = lambda p: total_objective(p, households, stage, draws)` | Compute or initialize `objective` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 417 | `results = []` | Compute or initialize `results` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 418 | `rng = np.random.default_rng(SEED)` | Compute or initialize `rng` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 419 | `for k in range(starts):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 420 | `initial = x0.copy() if k == 0 else x0 + rng.normal(0, 0.15, len(x0))` | Compute or initialize `initial` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 421 | `for i, (lo, hi) in enumerate(bounds):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 422 | `initial[i] = np.clip(initial[i], -np.inf if lo is None else lo, np.inf if hi is None else hi)` | Compute or initialize `initial[i]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 423–424 | `results.append(minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5)))` | Execute this call to print, save, validate, or update the object shown. |
| 425 | `converged = [r for r in results if r.success]` | Compute or initialize `converged` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 426 | `best = min(converged or results, key=lambda r: r.fun)` | Compute or initialize `best` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 427 | `se = np.full(len(names), np.nan)` | Compute or initialize `se` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 428 | `eigenvalues = np.array([])` | Compute or initialize `eigenvalues` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 429 | `condition = None` | Compute or initialize `condition` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 430–431 | `boundary = [name for name, p, (lo, hi) in zip(names, best.x, bounds) if (lo is not None and abs(p - lo) < 1e-5) or (hi is not None and abs(p - hi) < 1e-5)]` | Compute or initialize `boundary` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 432 | `if compute_se:` | Execute this branch only when the displayed condition holds. |
| 433 | `from scipy.optimize._numdiff import approx_derivative` | Import the named dependency used in this part of the implementation. |
| 434 | `hessian = numerical_hessian(objective, best.x)` | Compute or initialize `hessian` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 435 | `eigenvalues = np.linalg.eigvalsh(hessian)` | Compute or initialize `eigenvalues` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 436 | `condition = float(np.linalg.cond(hessian))` | Compute or initialize `condition` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 437 | `if eigenvalues.min() > 0 and not boundary:` | Execute this branch only when the displayed condition holds. |
| 439 | `scores = approx_derivative(lambda p: household_loglikes(p, households, stage, draws), best.x, method="3-point")` | Compute or initialize `scores` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 440 | `bread = np.linalg.inv(hessian)` | Compute or initialize `bread` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 441 | `covariance = bread @ (scores.T @ scores) @ bread` | Compute or initialize `covariance` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 442 | `se = np.sqrt(np.maximum(0, np.diag(covariance)))` | Compute or initialize `se` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 443–446 | `return dict(stage=stage, names=names, params=best.x, se=se, fun=float(best.fun), success=bool(best.success), message=str(best.message), boundary=boundary, hessian_eigenvalues=eigenvalues, hessian_condition=condition, starts=[dict(fun=float(r.fun), success=bool(r.success)) for r in results])` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 449 | `def display_results(result, output_dir, suffix=""):` | Print coefficients and save machine-readable results and optimizer diagnostics. |
| 450 | `table = Table(title=result["stage"] + suffix)` | Compute or initialize `table` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 451 | `for name in ["Parameter", "Estimate", "HH sandwich SE"]:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 452 | `table.add_column(name)` | Execute this call to print, save, validate, or update the object shown. |
| 453 | `for name, value, se in zip(result["names"], result["params"], result["se"]):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 454 | `table.add_row(name, f"{value:.5f}", f"{se:.5f}")` | Execute this call to print, save, validate, or update the object shown. |
| 455 | `console.print(table)` | Execute this call to print, save, validate, or update the object shown. |
| 456 | `console.print(f"Converged: {result['success']}; negative LL: {result['fun']:.4f}; {result['message']}")` | Execute this call to print, save, validate, or update the object shown. |
| 457 | `console.print(f"Boundary parameters: {result['boundary']}; Hessian condition: {result['hessian_condition']}")` | Execute this call to print, save, validate, or update the object shown. |
| 458 | `pd.DataFrame(dict(parameter=result["names"], estimate=result["params"], se=result["se"])).to_csv(output_dir / f"{result['stage']}{suffix}.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 459 | `serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}` | Compute or initialize `serial` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 460 | `(output_dir / f"{result['stage']}{suffix}.json").write_text(json.dumps(serial, indent=2))` | Execute this call to print, save, validate, or update the object shown. |
| 466 | `def profile_parameters(result, households, draws, output_dir):` | Fix gamma grid values and reoptimize nuisance parameters to expose compensating betas. |
| 467 | `names, fitted = result["names"], result["params"]` | Compute or initialize `(names, fitted)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 468 | `rows = []` | Compute or initialize `rows` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 469 | `for index, name in enumerate(names):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 470 | `if not name.startswith("gamma_"):` | Execute this branch only when the displayed condition holds. |
| 471 | `continue` | Skip this household because it does not meet the minimum panel length. |
| 473 | `keep = np.arange(len(fitted)) != index` | Compute or initialize `keep` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 474 | `def expand(free, fixed):` | Define this local helper for the enclosing calculation. |
| 475 | `p = fitted.copy()` | Compute or initialize `p` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 476 | `p[keep], p[index] = free, fixed` | Compute or initialize `(p[keep], p[index])` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 477 | `return p` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 478 | `bounds = [(0, 5) if n.startswith("sd_") else (None, 0) if n == "price" else (None, None) for i, n in enumerate(names) if i != index]` | Compute or initialize `bounds` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 479 | `for fixed in np.linspace(fitted[index] - PROFILE_WIDTH, fitted[index] + PROFILE_WIDTH, PROFILE_POINTS):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 480–481 | `fit = minimize(lambda free: total_objective(expand(free, fixed), households, result["stage"], draws), fitted[keep], method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10))` | Compute or initialize `fit` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 482 | `p = expand(fit.x, fixed)` | Compute or initialize `p` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 483–484 | `rows.append(dict(parameter=name, fixed=fixed, nll=float(fit.fun), delta_nll=float(fit.fun - result["fun"]), beta_berry=p[1], beta_plain=p[2], success=bool(fit.success)))` | Execute this call to print, save, validate, or update the object shown. |
| 485 | `pd.DataFrame(rows).to_csv(output_dir / "gamma_profiles.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 488 | `def simulate_panel(seed, n_hh=RECOVERY_HOUSEHOLDS, n_weeks=RECOVERY_WEEKS):` | Generate a dynamic panel with known persistent types and sequentially updated stock. |
| 490 | `rng = np.random.default_rng(seed)` | Compute or initialize `rng` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 491 | `truth = np.array([-0.4, 0.6, -0.2, -0.7, -1.0, -0.8, -0.4] + ([0.0] if USE_CONTROL_FUNCTION else []) + [0.5, 0.4, 0.25, 0.3, 0.2])` | Compute or initialize `truth` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 492 | `households = []` | Compute or initialize `households` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 493 | `for h in range(n_hh):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 494 | `dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D")` | Compute or initialize `dates` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 495–497 | `hh = dict(household_code=str(h), prices=rng.uniform(0.5, 3, (n_weeks, 3)), resids=rng.normal(size=(n_weeks, 3)), choices=np.zeros((n_weeks, 4)), dates=dates, stock=np.zeros((n_weeks, 3)), fallback=np.zeros((n_weeks, 3)))` | Compute or initialize `hh` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 498 | `type_draw = rng.normal(size=(1, 5))` | Compute or initialize `type_draw` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 499 | `for t in range(n_weeks):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 500 | `if t:` | Execute this branch only when the displayed condition holds. |
| 501 | `previous = hh["choices"][t - 1, :3]` | Compute or initialize `previous` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 502 | `shares = previous / previous.sum() if previous.sum() else np.zeros(3)` | Compute or initialize `shares` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 503 | `hh["stock"][t] = DELTA_DISCOUNT * (hh["stock"][t - 1] + shares)` | Compute or initialize `hh['stock'][t]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 504 | `one = {key: hh[key][t:t + 1] for key in ("prices", "resids")}` | Compute or initialize `one` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 505 | `one["z"] = np.log1p(hh["stock"][t:t + 1])` | Compute or initialize `one['z']` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 506 | `u = utility_draws(truth, one, "random_beta_gamma", type_draw)[0, 0]` | Compute or initialize `u` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 507 | `hh["choices"][t] = rng.multinomial(WEEKLY_CAPACITY, np.exp(u - logsumexp(u)))` | Compute or initialize `hh['choices'][t]` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 508 | `households.append(hh)` | Execute this call to print, save, validate, or update the object shown. |
| 509 | `reference = center_states(households)` | Compute or initialize `reference` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 510 | `truth[0] += truth[3] * reference[0]` | Update `truth[0]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 511 | `truth[1:3] += truth[4:6] * reference[1:3] - truth[3] * reference[0]` | Update `truth[1:3]` using the displayed increment; this accumulates rather than replaces its existing value. |
| 512 | `return households, truth` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 515 | `def run_recovery(output_dir, draws):` | Repeat simulation and compare full-model estimates against known parameters. |
| 516 | `rows = []` | Compute or initialize `rows` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 517 | `for replication in range(RECOVERY_REPLICATIONS):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 518 | `households, truth = simulate_panel(SEED + 1000 + replication)` | Compute or initialize `(households, truth)` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 519 | `fit = estimate_model(households, "random_beta_gamma", draws, compute_se=False)` | Compute or initialize `fit` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 520 | `for name, true, estimate in zip(fit["names"], truth, fit["params"]):` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 521–522 | `rows.append(dict(replication=replication, parameter=name, truth=true, estimate=estimate, error=estimate - true, success=fit["success"]))` | Execute this call to print, save, validate, or update the object shown. |
| 523 | `pd.DataFrame(rows).to_csv(output_dir / "recovery.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 526 | `def main():` | Read switches, dispatch enabled stages, and save configuration and outputs. |
| 527 | `parser = argparse.ArgumentParser(description=__doc__)` | Compute or initialize `parser` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 528 | `parser.add_argument("--simulate-only", action="store_true", help="Run recovery without reading Nielsen data")` | Execute this call to print, save, validate, or update the object shown. |
| 529 | `args = parser.parse_args()` | Compute or initialize `args` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 530 | `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` | Execute this call to print, save, validate, or update the object shown. |
| 531 | `draws = make_draws()` | Compute or initialize `draws` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 532 | `settings = {key: value for key, value in globals().items() if key.isupper() and isinstance(value, (str, int, float, bool, type(None)))}` | Compute or initialize `settings` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 533 | `(OUTPUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))` | Execute this call to print, save, validate, or update the object shown. |
| 534 | `if args.simulate_only:` | Execute this branch only when the displayed condition holds. |
| 535 | `run_recovery(OUTPUT_DIR, draws)` | Execute this call to print, save, validate, or update the object shown. |
| 536 | `return` | Return this expression to the caller; its mathematical meaning is given in the enclosing function section above. |
| 537 | `households = load_and_preprocess()` | Compute or initialize `households` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 538 | `if RUN_WITHIN_DIAGNOSTICS:` | Execute this branch only when the displayed condition holds. |
| 539 | `diagnose_within(households, OUTPUT_DIR)` | Execute this call to print, save, validate, or update the object shown. |
| 540 | `pd.DataFrame({"flavor": CATEGORIES[:3], "log_stock_reference": households[0]["reference"]}).to_csv(OUTPUT_DIR / "state_reference.csv", index=False)` | Execute this call to print, save, validate, or update the object shown. |
| 541 | `if RUN_WITHIN_FIXED_EFFECTS:` | Execute this branch only when the displayed condition holds. |
| 542 | `estimate_within_fixed_effects(households, OUTPUT_DIR)` | Execute this call to print, save, validate, or update the object shown. |
| 543 | `last = None` | Compute or initialize `last` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 544 | `for stage, enabled in [("homogeneous", RUN_HOMOGENEOUS), ("random_beta", RUN_RANDOM_BETA), ("random_beta_gamma", RUN_RANDOM_BETA_GAMMA)]:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 545 | `if enabled:` | Execute this branch only when the displayed condition holds. |
| 546 | `last = estimate_model(households, stage, draws, warm_start=last)` | Compute or initialize `last` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 547 | `display_results(last, OUTPUT_DIR)` | Execute this call to print, save, validate, or update the object shown. |
| 548 | `if RUN_PROFILES and last is not None:` | Execute this branch only when the displayed condition holds. |
| 549 | `profile_parameters(last, households, draws, OUTPUT_DIR)` | Execute this call to print, save, validate, or update the object shown. |
| 550 | `if RUN_DRAW_SENSITIVITY and last is not None and last["stage"] != "homogeneous":` | Execute this branch only when the displayed condition holds. |
| 551 | `for n, seed in [(N_DRAWS, SEED + 1), (2 * N_DRAWS, SEED)]:` | Repeat the enclosed calculation over the displayed households, weeks, categories, parameters, or stages. |
| 552 | `fit = estimate_model(households, last["stage"], make_draws(n, seed), warm_start=last)` | Compute or initialize `fit` from the displayed expression; array axes and statistical roles follow the enclosing function description. |
| 553 | `display_results(fit, OUTPUT_DIR, suffix=f"_draws{n}_seed{seed}")` | Execute this call to print, save, validate, or update the object shown. |
| 554 | `if RUN_RECOVERY:` | Execute this branch only when the displayed condition holds. |
| 555 | `run_recovery(OUTPUT_DIR, draws)` | Execute this call to print, save, validate, or update the object shown. |
| 558 | `if __name__ == "__main__":` | Execute this branch only when the displayed condition holds. |
| 559 | `main()` | Execute this call to print, save, validate, or update the object shown. |

## Verification of this delivery

The module compiled and synthetic checks verified across-store aggregation, the eight-inside/zero-outside rule, calendar-gap decay, zero initial stocks, stability under repeated shares, centering equivalence for random utilities, and household integration against the explicit formula. Small simulated fits exercised all three main stages, the curvature calculation, and the fixed-effect benchmark. The production parquet and first-stage regression have not been run in this environment; these checks establish code behavior, not empirical identification or valid real-data estimates.
