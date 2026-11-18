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

## Missing-price handling

An existing store-week-flavor group may contain only missing prices. The resolver now checks values as well as keys: it selects the first finite price above 0.1 from store-week, week-flavor, then overall-flavor averages. Its residual comes from the same selected level; missing or infinite residuals use the existing zero convention. Infinite source prices are converted to missing before the first-stage regression. If every price level fails, an error identifies the flavor and store-week instead of introducing an arbitrary price or silently dropping choices.

Targeted checks cover valid local prices, NaN/infinite price fallbacks, nonfinite residual handling, and failure when every price is missing.

## Line-by-line statement index

Every executable statement is indexed below; multiline expressions form a single row. Blank lines and standalone comments are omitted. See the model sections and adjacent source comments for mathematical roles.

| Lines | Code | Explanation |
| --- | --- | --- |
| 3 | `from pathlib import Path` | Import this dependency. |
| 4 | `import argparse` | Import this dependency. |
| 5 | `import json` | Import this dependency. |
| 6 | `import numpy as np` | Import this dependency. |
| 7 | `import pandas as pd` | Import this dependency. |
| 8 | `import polars as pl` | Import this dependency. |
| 9 | `import scipy as sp` | Import this dependency. |
| 10 | `from scipy.optimize import minimize` | Import this dependency. |
| 11 | `from scipy.special import logsumexp` | Import this dependency. |
| 12 | `from rich.console import Console` | Import this dependency. |
| 13 | `from rich.table import Table` | Import this dependency. |
| 14 | `import statsmodels.formula.api as smf` | Import this dependency. |
| 16 | `console = Console()` | Compute or initialize `console` using the displayed expression. |
| 17 | `MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"` | Compute or initialize `MERGED_PATH` using the displayed expression. |
| 18 | `OUTPUT_DIR = Path("step4_output")` | Compute or initialize `OUTPUT_DIR` using the displayed expression. |
| 19 | `CATEGORIES = ["other", "berry", "plain", "outside"]` | Compute or initialize `CATEGORIES` using the displayed expression. |
| 20 | `cat_map = {name: j for j, name in enumerate(CATEGORIES)}` | Compute or initialize `cat_map` using the displayed expression. |
| 23 | `RUN_WITHIN_DIAGNOSTICS = True` | Compute or initialize `RUN_WITHIN_DIAGNOSTICS` using the displayed expression. |
| 24 | `RUN_WITHIN_FIXED_EFFECTS = True` | Compute or initialize `RUN_WITHIN_FIXED_EFFECTS` using the displayed expression. |
| 25 | `RUN_HOMOGENEOUS = True` | Compute or initialize `RUN_HOMOGENEOUS` using the displayed expression. |
| 26 | `RUN_RANDOM_BETA = True` | Compute or initialize `RUN_RANDOM_BETA` using the displayed expression. |
| 27 | `RUN_RANDOM_BETA_GAMMA = True` | Compute or initialize `RUN_RANDOM_BETA_GAMMA` using the displayed expression. |
| 28 | `RUN_PROFILES = False` | Compute or initialize `RUN_PROFILES` using the displayed expression. |
| 29 | `RUN_RECOVERY = False` | Compute or initialize `RUN_RECOVERY` using the displayed expression. |
| 30 | `RUN_DRAW_SENSITIVITY = False` | Compute or initialize `RUN_DRAW_SENSITIVITY` using the displayed expression. |
| 31 | `COMPUTE_STANDARD_ERRORS = True` | Compute or initialize `COMPUTE_STANDARD_ERRORS` using the displayed expression. |
| 32 | `USE_CONTROL_FUNCTION = True` | Compute or initialize `USE_CONTROL_FUNCTION` using the displayed expression. |
| 35 | `WEEKLY_CAPACITY = 7` | Compute or initialize `WEEKLY_CAPACITY` using the displayed expression. |
| 36 | `OUTSIDE_RULE = "threshold"  # threshold: max(7-Q, 0); strict: reject Q > 7` | Compute or initialize `OUTSIDE_RULE` using the displayed expression. |
| 37 | `DELTA_DISCOUNT = 0.9` | Compute or initialize `DELTA_DISCOUNT` using the displayed expression. |
| 38 | `MIN_WEEKS = 2` | Compute or initialize `MIN_WEEKS` using the displayed expression. |
| 39 | `N_DRAWS = 128` | Compute or initialize `N_DRAWS` using the displayed expression. |
| 40 | `N_STARTS = 3` | Compute or initialize `N_STARTS` using the displayed expression. |
| 41 | `MAX_ITER = 600` | Compute or initialize `MAX_ITER` using the displayed expression. |
| 42 | `SEED = 306` | Compute or initialize `SEED` using the displayed expression. |
| 43 | `MAX_HOUSEHOLDS = None  # Set an integer for a reproducible development sample.` | Compute or initialize `MAX_HOUSEHOLDS` using the displayed expression. |
| 44 | `PROFILE_POINTS = 9` | Compute or initialize `PROFILE_POINTS` using the displayed expression. |
| 45 | `PROFILE_WIDTH = 1.0` | Compute or initialize `PROFILE_WIDTH` using the displayed expression. |
| 46 | `RECOVERY_HOUSEHOLDS = 60` | Compute or initialize `RECOVERY_HOUSEHOLDS` using the displayed expression. |
| 47 | `RECOVERY_WEEKS = 24` | Compute or initialize `RECOVERY_WEEKS` using the displayed expression. |
| 48 | `RECOVERY_REPLICATIONS = 3` | Compute or initialize `RECOVERY_REPLICATIONS` using the displayed expression. |
| 51 | `def make_draws(n_draws=N_DRAWS, seed=SEED):` | Define `make_draws`; its implementation and model role are explained in the sections above and source comments. |
| 53 | `half = np.random.default_rng(seed).standard_normal(((n_draws + 1) // 2, 5))` | Compute or initialize `half` using the displayed expression. |
| 54 | `return np.vstack([half, -half])[:n_draws]` | Return the displayed result to the caller. |
| 60 | `def load_and_preprocess():` | Define `load_and_preprocess`; its implementation and model role are explained in the sections above and source comments. |
| 61 | `schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()` | Compute or initialize `schema_names` using the displayed expression. |
| 63–70 | `exprs = [ pl.col("quantity").cast(pl.Float64), pl.col("head_age").cast(pl.Int64), pl.col("household_income").cast(pl.Int64), pl.col("household_size").cast(pl.Int64), pl.col("serving_per_container_cd").cast(pl.Int64), pl.col("price").cast(pl.Float64), ]` | Compute or initialize `exprs` using the displayed expression. |
| 72 | `if "product_module_code_hms" in schema_names:` | Apply the displayed condition before executing this branch. |
| 73 | `exprs.append(pl.col("product_module_code_hms").cast(pl.Int64))` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 75 | `scan_q = pl.scan_parquet(MERGED_PATH).with_columns(exprs)` | Compute or initialize `scan_q` using the displayed expression. |
| 77 | `if "product_module_code_hms" in schema_names:` | Apply the displayed condition before executing this branch. |
| 78 | `scan_q = scan_q.filter(pl.col("product_module_code_hms").is_in([3612, 3603]))` | Compute or initialize `scan_q` using the displayed expression. |
| 80–85 | `merged_df = ( scan_q.filter(pl.col("household_size") == 1) .filter(pl.col("serving_per_container_cd").is_in([67181961, 65622705])) .collect() .to_pandas() )` | Compute or initialize `merged_df` using the displayed expression. |
| 87 | `merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)` | Compute or initialize `merged_df['flavor_str']` using the displayed expression. |
| 88–90 | `merged_df["flavor_cd"] = pd.to_numeric( merged_df["flavor_cd"], errors="coerce" ).fillna(0)` | Compute or initialize `merged_df['flavor_cd']` using the displayed expression. |
| 92 | `merged_master = merged_df.copy()` | Compute or initialize `merged_master` using the displayed expression. |
| 94–96 | `merged_master["price"] = merged_master["price"].replace( [np.inf, -np.inf], np.nan )` | Compute or initialize `merged_master['price']` using the displayed expression. |
| 98–102 | `descr_cols = [ c for c in merged_df.columns if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"]) ]` | Compute or initialize `descr_cols` using the displayed expression. |
| 104 | `merged_df["full_text"] = ""` | Compute or initialize `merged_df['full_text']` using the displayed expression. |
| 105 | `for c in descr_cols:` | Repeat the enclosed calculation over the displayed sequence. |
| 106 | `merged_df["full_text"] += " " + merged_df[c].fillna("").astype(str)` | Accumulate the displayed contribution in `merged_df['full_text']`. |
| 107 | `merged_df["full_text"] = merged_df["full_text"].str.lower()` | Compute or initialize `merged_df['full_text']` using the displayed expression. |
| 109 | `berry_regex = r"berry\|straw\|blue\|rasp\|black\|cran\|cherry\|wildberry"` | Compute or initialize `berry_regex` using the displayed expression. |
| 110 | `plain_regex = r"plain\|unflavored"` | Compute or initialize `plain_regex` using the displayed expression. |
| 112 | `is_plain = merged_df["full_text"].str.contains(plain_regex, na=False)` | Compute or initialize `is_plain` using the displayed expression. |
| 113 | `is_berry = merged_df["full_text"].str.contains(berry_regex, na=False) & (~is_plain)` | Compute or initialize `is_berry` using the displayed expression. |
| 115 | `merged_master["flavor"] = np.select([is_plain, is_berry], [2, 1], default=0)` | Compute or initialize `merged_master['flavor']` using the displayed expression. |
| 117–121 | `market_price = ( merged_master.groupby(["upc", "week_end", "market_name"])["price"] .mean() .reset_index() )` | Compute or initialize `market_price` using the displayed expression. |
| 122–127 | `totals = ( market_price.groupby(["upc", "week_end"])["price"] .agg(["sum", "count"]) .reset_index() .rename(columns={"sum": "price_sum_all", "count": "n_markets_all"}) )` | Compute or initialize `totals` using the displayed expression. |
| 129 | `market_price = market_price.merge(totals, on=["upc", "week_end"], how="left")` | Compute or initialize `market_price` using the displayed expression. |
| 130–132 | `market_price["price_iv"] = ( market_price["price_sum_all"] - market_price["price"] ) / (market_price["n_markets_all"] - 1)` | Compute or initialize `market_price['price_iv']` using the displayed expression. |
| 133–135 | `market_price["price_iv"] = market_price["price_iv"].replace( [np.inf, -np.inf], np.nan )` | Compute or initialize `market_price['price_iv']` using the displayed expression. |
| 137–141 | `merged_master = merged_master.merge( market_price[["upc", "week_end", "market_name", "price_iv"]], on=["upc", "week_end", "market_name"], how="left", )` | Compute or initialize `merged_master` using the displayed expression. |
| 143–145 | `merged_master = merged_master[ (merged_master["price"] > 0.1) \| (merged_master["price"].isna()) ]` | Compute or initialize `merged_master` using the displayed expression. |
| 146–148 | `iv_res = smf.ols( "price ~ price_iv + brand_cd + C(week_end)", data=merged_master, missing="drop" ).fit()` | Compute or initialize `iv_res` using the displayed expression. |
| 150 | `merged_master["iv_res"] = np.nan` | Compute or initialize `merged_master['iv_res']` using the displayed expression. |
| 151 | `merged_master.loc[iv_res.model.data.row_labels, "iv_res"] = iv_res.resid` | Compute or initialize `merged_master.loc[iv_res.model.data.row_labels, 'iv_res']` using the displayed expression. |
| 153 | `valid_resids = merged_master["iv_res"].dropna()` | Compute or initialize `valid_resids` using the displayed expression. |
| 154 | `if len(valid_resids) > 0:` | Apply the displayed condition before executing this branch. |
| 155 | `res_mean = valid_resids.mean()` | Compute or initialize `res_mean` using the displayed expression. |
| 156 | `res_std = valid_resids.std()` | Compute or initialize `res_std` using the displayed expression. |
| 157–159 | `merged_master["iv_res"] = (merged_master["iv_res"] - res_mean) / ( res_std if res_std > 0 else 1.0 )` | Compute or initialize `merged_master['iv_res']` using the displayed expression. |
| 161–163 | `merged_master["iv_res"] = merged_master["iv_res"].replace( [np.inf, -np.inf], np.nan ).fillna(0.0)` | Compute or initialize `merged_master['iv_res']` using the displayed expression. |
| 165 | `def map_flavor_category(flavor):` | Define `map_flavor_category`; its implementation and model role are explained in the sections above and source comments. |
| 166 | `if pd.isna(flavor):` | Apply the displayed condition before executing this branch. |
| 167 | `return "outside"` | Return the displayed result to the caller. |
| 168 | `elif flavor == 0:` | Apply the displayed condition before executing this branch. |
| 169 | `return "other"` | Return the displayed result to the caller. |
| 170 | `elif flavor == 1:` | Apply the displayed condition before executing this branch. |
| 171 | `return "berry"` | Return the displayed result to the caller. |
| 172 | `elif flavor == 2:` | Apply the displayed condition before executing this branch. |
| 173 | `return "plain"` | Return the displayed result to the caller. |
| 175 | `merged_master["category"] = merged_master["flavor"].map(map_flavor_category)` | Compute or initialize `merged_master['category']` using the displayed expression. |
| 176 | `inside_df = merged_master[merged_master["category"] != "outside"].copy()` | Compute or initialize `inside_df` using the displayed expression. |
| 178 | `quantities = inside_df["quantity"].to_numpy(dtype=float)` | Compute or initialize `quantities` using the displayed expression. |
| 179 | `if (` | Apply the displayed condition before executing this branch. |
| 184–186 | `raise ValueError( "Source quantities must be finite nonnegative integer yogurt units." )` | Stop with an explicit validation error. |
| 187 | `inside_df = inside_df.loc[inside_df["quantity"] > 0].copy()` | Compute or initialize `inside_df` using the displayed expression. |
| 189–193 | `weekly = ( inside_df.groupby(["household_code", "week_end", "category"])["quantity"] .sum() .unstack("category", fill_value=0) )` | Compute or initialize `weekly` using the displayed expression. |
| 194 | `weekly = weekly.reindex(columns=CATEGORIES[:3], fill_value=0).reset_index()` | Compute or initialize `weekly` using the displayed expression. |
| 195 | `weekly["week_end"] = pd.to_datetime(weekly["week_end"])` | Compute or initialize `weekly['week_end']` using the displayed expression. |
| 198–200 | `stores = inside_df.groupby( ["household_code", "week_end", "store_code_uc"], as_index=False )["quantity"].sum()` | Compute or initialize `stores` using the displayed expression. |
| 201 | `stores["week_end"] = pd.to_datetime(stores["week_end"])` | Compute or initialize `stores['week_end']` using the displayed expression. |
| 202–205 | `stores = stores.sort_values( ["household_code", "week_end", "quantity", "store_code_uc"], ascending=[True, True, False, True], )` | Compute or initialize `stores` using the displayed expression. |
| 206 | `stores = stores.drop_duplicates(["household_code", "week_end"])` | Compute or initialize `stores` using the displayed expression. |
| 207–209 | `weekly = weekly.merge( stores[["household_code", "week_end", "store_code_uc"]], validate="one_to_one" )` | Compute or initialize `weekly` using the displayed expression. |
| 212 | `inside_df["week_end"] = pd.to_datetime(inside_df["week_end"])` | Compute or initialize `inside_df['week_end']` using the displayed expression. |
| 213–215 | `store_prices = inside_df.groupby(["store_code_uc", "week_end", "category"])[ ["price", "iv_res"] ].mean()` | Compute or initialize `store_prices` using the displayed expression. |
| 216–218 | `week_prices = inside_df.groupby(["week_end", "category"])[ ["price", "iv_res"] ].mean()` | Compute or initialize `week_prices` using the displayed expression. |
| 219 | `overall = inside_df.groupby("category")[["price", "iv_res"]].mean()` | Compute or initialize `overall` using the displayed expression. |
| 220 | `households = []` | Compute or initialize `households` using the displayed expression. |
| 221 | `for hh_id, group in weekly.groupby("household_code", sort=True):` | Repeat the enclosed calculation over the displayed sequence. |
| 222 | `group = group.sort_values("week_end")` | Compute or initialize `group` using the displayed expression. |
| 223 | `if len(group) < MIN_WEEKS:` | Apply the displayed condition before executing this branch. |
| 224 | `continue` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 225 | `prices, resids, counts, dates, fallback = [], [], [], [], []` | Compute or initialize `(prices, resids, counts, dates, fallback)` using the displayed expression. |
| 226 | `for row in group.itertuples(index=False):` | Repeat the enclosed calculation over the displayed sequence. |
| 227 | `q = np.array([row.other, row.berry, row.plain], dtype=float)` | Compute or initialize `q` using the displayed expression. |
| 228 | `if np.any(q < 0) or not np.allclose(q, np.round(q)):` | Apply the displayed condition before executing this branch. |
| 229 | `raise ValueError("Quantities must be nonnegative integer yogurt units.")` | Stop with an explicit validation error. |
| 230 | `if OUTSIDE_RULE == "strict" and q.sum() > WEEKLY_CAPACITY:` | Apply the displayed condition before executing this branch. |
| 231–233 | `raise ValueError( f"HH {hh_id}: weekly units exceed capacity; inspect units or use threshold rule." )` | Stop with an explicit validation error. |
| 234 | `if OUTSIDE_RULE not in ("strict", "threshold"):` | Apply the displayed condition before executing this branch. |
| 235 | `raise ValueError("Unknown OUTSIDE_RULE")` | Stop with an explicit validation error. |
| 236 | `counts.append(np.append(q, max(WEEKLY_CAPACITY - q.sum(), 0)))` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 237 | `p, r, flags = [], [], []` | Compute or initialize `(p, r, flags)` using the displayed expression. |
| 238 | `for cat in CATEGORIES[:3]:` | Repeat the enclosed calculation over the displayed sequence. |
| 239 | `key = (row.store_code_uc, row.week_end, cat)` | Compute or initialize `key` using the displayed expression. |
| 240 | `wk_key = (row.week_end, cat)` | Compute or initialize `wk_key` using the displayed expression. |
| 241–243 | `price, residual, flag = resolve_price_inputs( store_prices, week_prices, overall, key, wk_key, cat )` | Compute or initialize `(price, residual, flag)` using the displayed expression. |
| 244 | `p.append(price)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 245 | `r.append(residual if USE_CONTROL_FUNCTION else 0.0)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 246 | `flags.append(flag)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 247 | `prices.append(p)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 248 | `resids.append(r)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 249 | `dates.append(row.week_end)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 250 | `fallback.append(flags)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 251 | `choices = np.asarray(counts)` | Compute or initialize `choices` using the displayed expression. |
| 252 | `stock = compute_satiation_state(choices[:, :3], dates)` | Compute or initialize `stock` using the displayed expression. |
| 253–263 | `households.append( dict( household_code=str(hh_id), prices=np.asarray(prices), resids=np.asarray(resids), choices=choices, dates=pd.DatetimeIndex(dates), stock=stock, fallback=np.asarray(fallback), ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 264 | `if MAX_HOUSEHOLDS is not None and len(households) >= MAX_HOUSEHOLDS:` | Apply the displayed condition before executing this branch. |
| 265 | `break` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 266 | `if not households:` | Apply the displayed condition before executing this branch. |
| 267 | `raise ValueError("No eligible households after preprocessing.")` | Stop with an explicit validation error. |
| 268 | `for hh in households:` | Repeat the enclosed calculation over the displayed sequence. |
| 269 | `if not np.isfinite(hh["prices"]).all() or not np.isfinite(hh["resids"]).all():` | Apply the displayed condition before executing this branch. |
| 270 | `raise ValueError("Nonfinite price/control-function inputs.")` | Stop with an explicit validation error. |
| 271 | `center_states(households)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 272 | `return households` | Return the displayed result to the caller. |
| 275 | `def resolve_price_inputs(store_prices, week_prices, overall, key, wk_key, cat):` | Define `resolve_price_inputs`; its implementation and model role are explained in the sections above and source comments. |
| 278 | `for table, lookup, flag in [` | Repeat the enclosed calculation over the displayed sequence. |
| 283 | `if lookup not in table.index:` | Apply the displayed condition before executing this branch. |
| 284 | `continue` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 285 | `values = table.loc[lookup]` | Compute or initialize `values` using the displayed expression. |
| 286 | `price = values["price"]` | Compute or initialize `price` using the displayed expression. |
| 287 | `if pd.isna(price) or not np.isfinite(price) or price <= 0.1:` | Apply the displayed condition before executing this branch. |
| 288 | `continue` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 289 | `residual = values["iv_res"]` | Compute or initialize `residual` using the displayed expression. |
| 291 | `if pd.isna(residual) or not np.isfinite(residual):` | Apply the displayed condition before executing this branch. |
| 292 | `residual = 0.0` | Compute or initialize `residual` using the displayed expression. |
| 293 | `return float(price), float(residual), flag` | Return the displayed result to the caller. |
| 294–298 | `raise ValueError( f"No finite price > 0.1 for flavor {cat!r}: store-week={key!r}. " "Store-week, week-flavor, and overall-flavor fallbacks all failed. " "Inspect source prices and sample restrictions." )` | Stop with an explicit validation error. |
| 301 | `def compute_satiation_state(inside_counts, dates):` | Define `compute_satiation_state`; its implementation and model role are explained in the sections above and source comments. |
| 303 | `stock = np.zeros_like(inside_counts, dtype=float)` | Compute or initialize `stock` using the displayed expression. |
| 304 | `for t in range(1, len(dates)):` | Repeat the enclosed calculation over the displayed sequence. |
| 305 | `elapsed = (pd.Timestamp(dates[t]) - pd.Timestamp(dates[t - 1])).days / 7.0` | Compute or initialize `elapsed` using the displayed expression. |
| 306 | `if elapsed <= 0:` | Apply the displayed condition before executing this branch. |
| 307 | `raise ValueError("Household dates must be unique and increasing.")` | Stop with an explicit validation error. |
| 308 | `previous = inside_counts[t - 1]` | Compute or initialize `previous` using the displayed expression. |
| 309 | `shares = previous / previous.sum() if previous.sum() > 0 else np.zeros(3)` | Compute or initialize `shares` using the displayed expression. |
| 310 | `stock[t] = DELTA_DISCOUNT**elapsed * (stock[t - 1] + shares)` | Compute or initialize `stock[t]` using the displayed expression. |
| 311 | `return stock` | Return the displayed result to the caller. |
| 314 | `def center_states(households, reference=None):` | Define `center_states`; its implementation and model role are explained in the sections above and source comments. |
| 316 | `if reference is None:` | Apply the displayed condition before executing this branch. |
| 317–319 | `reference = np.concatenate([np.log1p(h["stock"]) for h in households]).mean( axis=0 )` | Compute or initialize `reference` using the displayed expression. |
| 320 | `for hh in households:` | Repeat the enclosed calculation over the displayed sequence. |
| 321 | `hh["z"] = np.log1p(hh["stock"]) - reference` | Compute or initialize `hh['z']` using the displayed expression. |
| 322 | `hh["reference"] = np.asarray(reference)` | Compute or initialize `hh['reference']` using the displayed expression. |
| 323 | `return np.asarray(reference)` | Return the displayed result to the caller. |
| 329 | `def diagnose_within(households, output_dir):` | Define `diagnose_within`; its implementation and model role are explained in the sections above and source comments. |
| 330 | `rows = []` | Compute or initialize `rows` using the displayed expression. |
| 331 | `for hh in households:` | Repeat the enclosed calculation over the displayed sequence. |
| 332 | `z = hh["z"]` | Compute or initialize `z` using the displayed expression. |
| 333 | `shares = hh["choices"][:, :3] / hh["choices"][:, :3].sum(axis=1, keepdims=True)` | Compute or initialize `shares` using the displayed expression. |
| 335 | `for j, cat in enumerate(CATEGORIES[:3]):` | Repeat the enclosed calculation over the displayed sequence. |
| 336–345 | `rows.append( dict( household=hh["household_code"], flavor=cat, weeks=len(z), within_sd=float(z[:, j].std()), state_range=float(np.ptp(z[:, j])), share_changes=int(np.count_nonzero(np.diff(shares[:, j]))), ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 346 | `frame = pd.DataFrame(rows)` | Compute or initialize `frame` using the displayed expression. |
| 347 | `frame.to_csv(output_dir / "within_household.csv", index=False)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 348–352 | `console.print( frame.groupby("flavor")[["within_sd", "state_range", "share_changes"]].agg( ["mean", "median"] ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 354 | `dates = np.concatenate([h["dates"].asi8 for h in households])` | Compute or initialize `dates` using the displayed expression. |
| 355 | `_, week_idx = np.unique(dates, return_inverse=True)` | Compute or initialize `(_, week_idx)` using the displayed expression. |
| 356 | `blocks = []` | Compute or initialize `blocks` using the displayed expression. |
| 357 | `for j, cat in enumerate(CATEGORIES[:3]):` | Repeat the enclosed calculation over the displayed sequence. |
| 358 | `target = np.concatenate([h["z"][:, j] for h in households])` | Compute or initialize `target` using the displayed expression. |
| 359 | `price = np.concatenate([h["prices"][:, j] for h in households])` | Compute or initialize `price` using the displayed expression. |
| 360 | `resid = np.concatenate([h["resids"][:, j] for h in households])` | Compute or initialize `resid` using the displayed expression. |
| 361 | `sizes = [len(h["z"]) for h in households]` | Compute or initialize `sizes` using the displayed expression. |
| 362 | `hh_idx = np.repeat(np.arange(len(sizes)), sizes)` | Compute or initialize `hh_idx` using the displayed expression. |
| 364 | `from scipy.sparse import coo_matrix, hstack` | Import this dependency. |
| 365 | `from scipy.sparse.linalg import lsqr` | Import this dependency. |
| 367 | `n = len(target)` | Compute or initialize `n` using the displayed expression. |
| 368–370 | `hd = coo_matrix( (np.ones(n), (np.arange(n), hh_idx)), shape=(n, len(sizes)) ).tocsr()` | Compute or initialize `hd` using the displayed expression. |
| 371 | `wd = coo_matrix((np.ones(n), (np.arange(n), week_idx))).tocsr()` | Compute or initialize `wd` using the displayed expression. |
| 372 | `design = hstack([hd, wd, coo_matrix(np.column_stack([price, resid]))]).tocsr()` | Compute or initialize `design` using the displayed expression. |
| 373 | `residual = target - design @ lsqr(design, target, atol=1e-10, btol=1e-10)[0]` | Compute or initialize `residual` using the displayed expression. |
| 374–376 | `within = np.concatenate( [h["z"][:, j] - h["z"][:, j].mean() for h in households] )` | Compute or initialize `within` using the displayed expression. |
| 377–383 | `blocks.append( dict( flavor=cat, within_sd=float(within.std()), residual_sd=float(residual.std()), ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 384–386 | `pd.DataFrame(blocks).to_csv( output_dir / "residual_state_variation.csv", index=False )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 387 | `console.print(pd.DataFrame(blocks))` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 388 | `all_choices = np.concatenate([h["choices"] for h in households])` | Compute or initialize `all_choices` using the displayed expression. |
| 389–391 | `console.print( f"Households: {len(households)}; purchase weeks: {len(all_choices)}; weeks above seven: {(all_choices[:, :3].sum(axis=1) > 7).sum()}" )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 392–394 | `console.print( f"Price fallback fraction: {(np.concatenate([h['fallback'] for h in households]) > 0).mean():.3f}" )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 400 | `def estimate_within_fixed_effects(households, output_dir):` | Define `estimate_within_fixed_effects`; its implementation and model role are explained in the sections above and source comments. |
| 403 | `n_hh = len(households)` | Compute or initialize `n_hh` using the displayed expression. |
| 404 | `n_common = 5 if USE_CONTROL_FUNCTION else 4` | Compute or initialize `n_common` using the displayed expression. |
| 406 | `def objective_gradient(params):` | Define `objective_gradient`; its implementation and model role are explained in the sections above and source comments. |
| 407 | `gamma, alpha = params[:3], params[3]` | Compute or initialize `(gamma, alpha)` using the displayed expression. |
| 408 | `intercepts = params[n_common:].reshape(n_hh, 3)` | Compute or initialize `intercepts` using the displayed expression. |
| 409 | `gradient = np.zeros_like(params)` | Compute or initialize `gradient` using the displayed expression. |
| 410 | `value = 0.0` | Compute or initialize `value` using the displayed expression. |
| 411 | `for h, hh in enumerate(households):` | Repeat the enclosed calculation over the displayed sequence. |
| 412 | `inside = intercepts[h] + gamma * hh["z"] + alpha * hh["prices"]` | Compute or initialize `inside` using the displayed expression. |
| 413 | `if USE_CONTROL_FUNCTION:` | Apply the displayed condition before executing this branch. |
| 414 | `inside += params[4] * hh["resids"]` | Accumulate the displayed contribution in `inside`. |
| 415 | `u = np.column_stack([inside, np.zeros(len(inside))])` | Compute or initialize `u` using the displayed expression. |
| 416 | `lp = u - logsumexp(u, axis=1, keepdims=True)` | Compute or initialize `lp` using the displayed expression. |
| 417 | `value -= np.sum(hh["choices"] * lp)` | Accumulate the displayed contribution in `value`. |
| 418–420 | `error = ( hh["choices"].sum(axis=1, keepdims=True) * np.exp(lp) - hh["choices"] )[:, :3]` | Compute or initialize `error` using the displayed expression. |
| 421 | `gradient[:3] += np.sum(error * hh["z"], axis=0)` | Accumulate the displayed contribution in `gradient[:3]`. |
| 422 | `gradient[3] += np.sum(error * hh["prices"])` | Accumulate the displayed contribution in `gradient[3]`. |
| 423 | `if USE_CONTROL_FUNCTION:` | Apply the displayed condition before executing this branch. |
| 424 | `gradient[4] += np.sum(error * hh["resids"])` | Accumulate the displayed contribution in `gradient[4]`. |
| 425 | `gradient[n_common + 3 * h : n_common + 3 * h + 3] = error.sum(axis=0)` | Compute or initialize `gradient[n_common + 3 * h:n_common + 3 * h + 3]` using the displayed expression. |
| 426 | `return float(value), gradient` | Return the displayed result to the caller. |
| 428 | `initial = np.zeros(n_common + 3 * n_hh)` | Compute or initialize `initial` using the displayed expression. |
| 429 | `initial[3] = -0.1` | Compute or initialize `initial[3]` using the displayed expression. |
| 430 | `bounds = [(None, None)] * len(initial)` | Compute or initialize `bounds` using the displayed expression. |
| 431 | `bounds[3] = (None, 0.0)` | Compute or initialize `bounds[3]` using the displayed expression. |
| 432–439 | `fit = minimize( objective_gradient, initial, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5), )` | Compute or initialize `fit` using the displayed expression. |
| 440–442 | `names = ["gamma_other", "gamma_berry", "gamma_plain", "price"] + ( ["control_function"] if USE_CONTROL_FUNCTION else [] )` | Compute or initialize `names` using the displayed expression. |
| 443–445 | `pd.DataFrame(dict(parameter=names, estimate=fit.x[:n_common])).to_csv( output_dir / "within_fixed_effects.csv", index=False )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 446–451 | `metadata = dict( success=bool(fit.success), message=str(fit.message), fun=float(fit.fun), warning="Short-panel incidental-parameter bias and separated HH flavor intercepts are possible; no bias correction or SEs.", )` | Compute or initialize `metadata` using the displayed expression. |
| 452–454 | `(output_dir / "within_fixed_effects.json").write_text( json.dumps(metadata, indent=2) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 455–459 | `console.print( "Within-household fixed-effect benchmark", dict(zip(names, fit.x[:n_common])), metadata, )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 460 | `return fit` | Return the displayed result to the caller. |
| 463 | `def parameter_names(stage):` | Define `parameter_names`; its implementation and model role are explained in the sections above and source comments. |
| 464–472 | `names = [ "constant", "beta_berry", "beta_plain", "gamma_other", "gamma_berry", "gamma_plain", "price", ]` | Compute or initialize `names` using the displayed expression. |
| 473 | `if USE_CONTROL_FUNCTION:` | Apply the displayed condition before executing this branch. |
| 474 | `names += ["control_function"]` | Accumulate the displayed contribution in `names`. |
| 475 | `if stage in ("random_beta", "random_beta_gamma"):` | Apply the displayed condition before executing this branch. |
| 476 | `names += ["sd_beta_berry", "sd_beta_plain"]` | Accumulate the displayed contribution in `names`. |
| 477 | `if stage == "random_beta_gamma":` | Apply the displayed condition before executing this branch. |
| 478 | `names += ["sd_gamma_other", "sd_gamma_berry", "sd_gamma_plain"]` | Accumulate the displayed contribution in `names`. |
| 479 | `return names` | Return the displayed result to the caller. |
| 482 | `def utility_draws(params, hh, stage, draws):` | Define `utility_draws`; its implementation and model role are explained in the sections above and source comments. |
| 483 | `n = 1 if stage == "homogeneous" else len(draws)` | Compute or initialize `n` using the displayed expression. |
| 484 | `beta = np.tile([0.0, params[1], params[2]], (n, 1))` | Compute or initialize `beta` using the displayed expression. |
| 485 | `gamma = np.tile(params[3:6], (n, 1))` | Compute or initialize `gamma` using the displayed expression. |
| 486 | `offset = 8 if USE_CONTROL_FUNCTION else 7` | Compute or initialize `offset` using the displayed expression. |
| 487 | `if stage != "homogeneous":` | Apply the displayed condition before executing this branch. |
| 488 | `beta[:, 1:] += draws[:, :2] * params[offset : offset + 2]` | Accumulate the displayed contribution in `beta[:, 1:]`. |
| 489 | `if stage == "random_beta_gamma":` | Apply the displayed condition before executing this branch. |
| 490 | `deviations = draws[:, 2:5] * params[offset + 2 : offset + 5]` | Compute or initialize `deviations` using the displayed expression. |
| 491 | `gamma += deviations` | Accumulate the displayed contribution in `gamma`. |
| 493 | `beta += deviations * hh.get("reference", np.zeros(3))` | Accumulate the displayed contribution in `beta`. |
| 494 | `base = params[0] + params[6] * hh["prices"]` | Compute or initialize `base` using the displayed expression. |
| 495 | `if USE_CONTROL_FUNCTION:` | Apply the displayed condition before executing this branch. |
| 496 | `base += params[7] * hh["resids"]` | Accumulate the displayed contribution in `base`. |
| 497–499 | `inside = ( base[:, None, :] + beta[None, :, :] + gamma[None, :, :] * hh["z"][:, None, :] )` | Compute or initialize `inside` using the displayed expression. |
| 500 | `return np.concatenate([inside, np.zeros((len(inside), n, 1))], axis=2)` | Return the displayed result to the caller. |
| 503 | `def household_loglikes(params, households, stage, draws):` | Define `household_loglikes`; its implementation and model role are explained in the sections above and source comments. |
| 504 | `ll = []` | Compute or initialize `ll` using the displayed expression. |
| 505 | `for hh in households:` | Repeat the enclosed calculation over the displayed sequence. |
| 506 | `u = utility_draws(params, hh, stage, draws)` | Compute or initialize `u` using the displayed expression. |
| 507 | `log_probs = u - logsumexp(u, axis=2, keepdims=True)` | Compute or initialize `log_probs` using the displayed expression. |
| 509 | `conditional = np.sum(hh["choices"][:, None, :] * log_probs, axis=(0, 2))` | Compute or initialize `conditional` using the displayed expression. |
| 510 | `ll.append(logsumexp(conditional) - np.log(len(conditional)))` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 511 | `return np.asarray(ll)` | Return the displayed result to the caller. |
| 514 | `def total_objective(params, households, stage, draws):` | Define `total_objective`; its implementation and model role are explained in the sections above and source comments. |
| 515 | `value = -household_loglikes(params, households, stage, draws).sum()` | Compute or initialize `value` using the displayed expression. |
| 516 | `return float(value) if np.isfinite(value) else 1e100` | Return the displayed result to the caller. |
| 519 | `def numerical_hessian(function, params):` | Define `numerical_hessian`; its implementation and model role are explained in the sections above and source comments. |
| 521 | `from scipy.optimize._numdiff import approx_derivative` | Import this dependency. |
| 523–525 | `gradient = lambda x: approx_derivative( lambda y: np.array([function(y)]), x, method="3-point" ).ravel()` | Compute or initialize `gradient` using the displayed expression. |
| 526 | `matrix = approx_derivative(gradient, params, method="3-point")` | Compute or initialize `matrix` using the displayed expression. |
| 527 | `return (matrix + matrix.T) / 2` | Return the displayed result to the caller. |
| 530 | `def estimate_model(` | Define `estimate_model`; its implementation and model role are explained in the sections above and source comments. |
| 538 | `names = parameter_names(stage)` | Compute or initialize `names` using the displayed expression. |
| 539 | `x0 = np.zeros(len(names))` | Compute or initialize `x0` using the displayed expression. |
| 540 | `x0[3:6] = -0.1` | Compute or initialize `x0[3:6]` using the displayed expression. |
| 541 | `x0[6] = -0.1` | Compute or initialize `x0[6]` using the displayed expression. |
| 542 | `for i, name in enumerate(names):` | Repeat the enclosed calculation over the displayed sequence. |
| 543 | `if name.startswith("sd_"):` | Apply the displayed condition before executing this branch. |
| 544 | `x0[i] = 0.2` | Compute or initialize `x0[i]` using the displayed expression. |
| 545 | `if warm_start is not None and name in warm_start["names"]:` | Apply the displayed condition before executing this branch. |
| 546 | `x0[i] = warm_start["params"][warm_start["names"].index(name)]` | Compute or initialize `x0[i]` using the displayed expression. |
| 548–555 | `bounds = [ (0.0, 5.0) if name.startswith("sd_") else (None, 0.0) if name == "price" else (None, None) for name in names ]` | Compute or initialize `bounds` using the displayed expression. |
| 556 | `objective = lambda p: total_objective(p, households, stage, draws)` | Compute or initialize `objective` using the displayed expression. |
| 557 | `results = []` | Compute or initialize `results` using the displayed expression. |
| 558 | `rng = np.random.default_rng(SEED)` | Compute or initialize `rng` using the displayed expression. |
| 559 | `for k in range(starts):` | Repeat the enclosed calculation over the displayed sequence. |
| 560 | `initial = x0.copy() if k == 0 else x0 + rng.normal(0, 0.15, len(x0))` | Compute or initialize `initial` using the displayed expression. |
| 561 | `for i, (lo, hi) in enumerate(bounds):` | Repeat the enclosed calculation over the displayed sequence. |
| 562–564 | `initial[i] = np.clip( initial[i], -np.inf if lo is None else lo, np.inf if hi is None else hi )` | Compute or initialize `initial[i]` using the displayed expression. |
| 565–573 | `results.append( minimize( objective, initial, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5), ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 574 | `converged = [r for r in results if r.success]` | Compute or initialize `converged` using the displayed expression. |
| 575 | `best = min(converged or results, key=lambda r: r.fun)` | Compute or initialize `best` using the displayed expression. |
| 576 | `se = np.full(len(names), np.nan)` | Compute or initialize `se` using the displayed expression. |
| 577 | `eigenvalues = np.array([])` | Compute or initialize `eigenvalues` using the displayed expression. |
| 578 | `condition = None` | Compute or initialize `condition` using the displayed expression. |
| 579–584 | `boundary = [ name for name, p, (lo, hi) in zip(names, best.x, bounds) if (lo is not None and abs(p - lo) < 1e-5) or (hi is not None and abs(p - hi) < 1e-5) ]` | Compute or initialize `boundary` using the displayed expression. |
| 585 | `if compute_se:` | Apply the displayed condition before executing this branch. |
| 586 | `from scipy.optimize._numdiff import approx_derivative` | Import this dependency. |
| 588 | `hessian = numerical_hessian(objective, best.x)` | Compute or initialize `hessian` using the displayed expression. |
| 589 | `eigenvalues = np.linalg.eigvalsh(hessian)` | Compute or initialize `eigenvalues` using the displayed expression. |
| 590 | `condition = float(np.linalg.cond(hessian))` | Compute or initialize `condition` using the displayed expression. |
| 591 | `if eigenvalues.min() > 0 and not boundary:` | Apply the displayed condition before executing this branch. |
| 593–597 | `scores = approx_derivative( lambda p: household_loglikes(p, households, stage, draws), best.x, method="3-point", )` | Compute or initialize `scores` using the displayed expression. |
| 598 | `bread = np.linalg.inv(hessian)` | Compute or initialize `bread` using the displayed expression. |
| 599 | `covariance = bread @ (scores.T @ scores) @ bread` | Compute or initialize `covariance` using the displayed expression. |
| 600 | `se = np.sqrt(np.maximum(0, np.diag(covariance)))` | Compute or initialize `se` using the displayed expression. |
| 601–613 | `return dict( stage=stage, names=names, params=best.x, se=se, fun=float(best.fun), success=bool(best.success), message=str(best.message), boundary=boundary, hessian_eigenvalues=eigenvalues, hessian_condition=condition, starts=[dict(fun=float(r.fun), success=bool(r.success)) for r in results], )` | Return the displayed result to the caller. |
| 616 | `def display_results(result, output_dir, suffix=""):` | Define `display_results`; its implementation and model role are explained in the sections above and source comments. |
| 617 | `table = Table(title=result["stage"] + suffix)` | Compute or initialize `table` using the displayed expression. |
| 618 | `for name in ["Parameter", "Estimate", "HH sandwich SE"]:` | Repeat the enclosed calculation over the displayed sequence. |
| 619 | `table.add_column(name)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 620 | `for name, value, se in zip(result["names"], result["params"], result["se"]):` | Repeat the enclosed calculation over the displayed sequence. |
| 621 | `table.add_row(name, f"{value:.5f}", f"{se:.5f}")` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 622 | `console.print(table)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 623–625 | `console.print( f"Converged: {result['success']}; negative LL: {result['fun']:.4f}; {result['message']}" )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 626–628 | `console.print( f"Boundary parameters: {result['boundary']}; Hessian condition: {result['hessian_condition']}" )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 629–631 | `pd.DataFrame( dict(parameter=result["names"], estimate=result["params"], se=result["se"]) ).to_csv(output_dir / f"{result['stage']}{suffix}.csv", index=False)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 632–635 | `serial = { key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items() }` | Compute or initialize `serial` using the displayed expression. |
| 636–638 | `(output_dir / f"{result['stage']}{suffix}.json").write_text( json.dumps(serial, indent=2) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 644 | `def profile_parameters(result, households, draws, output_dir):` | Define `profile_parameters`; its implementation and model role are explained in the sections above and source comments. |
| 645 | `names, fitted = result["names"], result["params"]` | Compute or initialize `(names, fitted)` using the displayed expression. |
| 646 | `rows = []` | Compute or initialize `rows` using the displayed expression. |
| 647 | `for index, name in enumerate(names):` | Repeat the enclosed calculation over the displayed sequence. |
| 648 | `if not name.startswith("gamma_"):` | Apply the displayed condition before executing this branch. |
| 649 | `continue` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 651 | `keep = np.arange(len(fitted)) != index` | Compute or initialize `keep` using the displayed expression. |
| 653 | `def expand(free, fixed):` | Define `expand`; its implementation and model role are explained in the sections above and source comments. |
| 654 | `p = fitted.copy()` | Compute or initialize `p` using the displayed expression. |
| 655 | `p[keep], p[index] = free, fixed` | Compute or initialize `(p[keep], p[index])` using the displayed expression. |
| 656 | `return p` | Return the displayed result to the caller. |
| 658–666 | `bounds = [ (0, 5) if n.startswith("sd_") else (None, 0) if n == "price" else (None, None) for i, n in enumerate(names) if i != index ]` | Compute or initialize `bounds` using the displayed expression. |
| 667 | `for fixed in np.linspace(` | Repeat the enclosed calculation over the displayed sequence. |
| 670–678 | `fit = minimize( lambda free: total_objective( expand(free, fixed), households, result["stage"], draws ), fitted[keep], method="L-BFGS-B", bounds=bounds, options=dict(maxiter=MAX_ITER, ftol=1e-10), )` | Compute or initialize `fit` using the displayed expression. |
| 679 | `p = expand(fit.x, fixed)` | Compute or initialize `p` using the displayed expression. |
| 680–690 | `rows.append( dict( parameter=name, fixed=fixed, nll=float(fit.fun), delta_nll=float(fit.fun - result["fun"]), beta_berry=p[1], beta_plain=p[2], success=bool(fit.success), ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 691 | `pd.DataFrame(rows).to_csv(output_dir / "gamma_profiles.csv", index=False)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 694 | `def simulate_panel(seed, n_hh=RECOVERY_HOUSEHOLDS, n_weeks=RECOVERY_WEEKS):` | Define `simulate_panel`; its implementation and model role are explained in the sections above and source comments. |
| 696 | `rng = np.random.default_rng(seed)` | Compute or initialize `rng` using the displayed expression. |
| 697–701 | `truth = np.array( [-0.4, 0.6, -0.2, -0.7, -1.0, -0.8, -0.4] + ([0.0] if USE_CONTROL_FUNCTION else []) + [0.5, 0.4, 0.25, 0.3, 0.2] )` | Compute or initialize `truth` using the displayed expression. |
| 702 | `households = []` | Compute or initialize `households` using the displayed expression. |
| 703 | `for h in range(n_hh):` | Repeat the enclosed calculation over the displayed sequence. |
| 704 | `dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D")` | Compute or initialize `dates` using the displayed expression. |
| 705–713 | `hh = dict( household_code=str(h), prices=rng.uniform(0.5, 3, (n_weeks, 3)), resids=rng.normal(size=(n_weeks, 3)), choices=np.zeros((n_weeks, 4)), dates=dates, stock=np.zeros((n_weeks, 3)), fallback=np.zeros((n_weeks, 3)), )` | Compute or initialize `hh` using the displayed expression. |
| 714 | `type_draw = rng.normal(size=(1, 5))` | Compute or initialize `type_draw` using the displayed expression. |
| 715 | `for t in range(n_weeks):` | Repeat the enclosed calculation over the displayed sequence. |
| 716 | `if t:` | Apply the displayed condition before executing this branch. |
| 717 | `previous = hh["choices"][t - 1, :3]` | Compute or initialize `previous` using the displayed expression. |
| 718 | `shares = previous / previous.sum() if previous.sum() else np.zeros(3)` | Compute or initialize `shares` using the displayed expression. |
| 719 | `hh["stock"][t] = DELTA_DISCOUNT * (hh["stock"][t - 1] + shares)` | Compute or initialize `hh['stock'][t]` using the displayed expression. |
| 720 | `one = {key: hh[key][t : t + 1] for key in ("prices", "resids")}` | Compute or initialize `one` using the displayed expression. |
| 721 | `one["z"] = np.log1p(hh["stock"][t : t + 1])` | Compute or initialize `one['z']` using the displayed expression. |
| 722 | `u = utility_draws(truth, one, "random_beta_gamma", type_draw)[0, 0]` | Compute or initialize `u` using the displayed expression. |
| 723–725 | `hh["choices"][t] = rng.multinomial( WEEKLY_CAPACITY, np.exp(u - logsumexp(u)) )` | Compute or initialize `hh['choices'][t]` using the displayed expression. |
| 726 | `households.append(hh)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 727 | `reference = center_states(households)` | Compute or initialize `reference` using the displayed expression. |
| 728 | `truth[0] += truth[3] * reference[0]` | Accumulate the displayed contribution in `truth[0]`. |
| 729 | `truth[1:3] += truth[4:6] * reference[1:3] - truth[3] * reference[0]` | Accumulate the displayed contribution in `truth[1:3]`. |
| 730 | `return households, truth` | Return the displayed result to the caller. |
| 733 | `def run_recovery(output_dir, draws):` | Define `run_recovery`; its implementation and model role are explained in the sections above and source comments. |
| 734 | `rows = []` | Compute or initialize `rows` using the displayed expression. |
| 735 | `for replication in range(RECOVERY_REPLICATIONS):` | Repeat the enclosed calculation over the displayed sequence. |
| 736 | `households, truth = simulate_panel(SEED + 1000 + replication)` | Compute or initialize `(households, truth)` using the displayed expression. |
| 737 | `fit = estimate_model(households, "random_beta_gamma", draws, compute_se=False)` | Compute or initialize `fit` using the displayed expression. |
| 738 | `for name, true, estimate in zip(fit["names"], truth, fit["params"]):` | Repeat the enclosed calculation over the displayed sequence. |
| 739–748 | `rows.append( dict( replication=replication, parameter=name, truth=true, estimate=estimate, error=estimate - true, success=fit["success"], ) )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 749 | `pd.DataFrame(rows).to_csv(output_dir / "recovery.csv", index=False)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 752 | `def main():` | Define `main`; its implementation and model role are explained in the sections above and source comments. |
| 753 | `parser = argparse.ArgumentParser(description=__doc__)` | Compute or initialize `parser` using the displayed expression. |
| 754–758 | `parser.add_argument( "--simulate-only", action="store_true", help="Run recovery without reading Nielsen data", )` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 759 | `args = parser.parse_args()` | Compute or initialize `args` using the displayed expression. |
| 760 | `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 761 | `draws = make_draws()` | Compute or initialize `draws` using the displayed expression. |
| 762–766 | `settings = { key: value for key, value in globals().items() if key.isupper() and isinstance(value, (str, int, float, bool, type(None))) }` | Compute or initialize `settings` using the displayed expression. |
| 767 | `(OUTPUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 768 | `if args.simulate_only:` | Apply the displayed condition before executing this branch. |
| 769 | `run_recovery(OUTPUT_DIR, draws)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 770 | `return` | Return the displayed result to the caller. |
| 771 | `households = load_and_preprocess()` | Compute or initialize `households` using the displayed expression. |
| 772 | `if RUN_WITHIN_DIAGNOSTICS:` | Apply the displayed condition before executing this branch. |
| 773 | `diagnose_within(households, OUTPUT_DIR)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 774–776 | `pd.DataFrame( {"flavor": CATEGORIES[:3], "log_stock_reference": households[0]["reference"]} ).to_csv(OUTPUT_DIR / "state_reference.csv", index=False)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 777 | `if RUN_WITHIN_FIXED_EFFECTS:` | Apply the displayed condition before executing this branch. |
| 778 | `estimate_within_fixed_effects(households, OUTPUT_DIR)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 779 | `last = None` | Compute or initialize `last` using the displayed expression. |
| 780 | `for stage, enabled in [` | Repeat the enclosed calculation over the displayed sequence. |
| 785 | `if enabled:` | Apply the displayed condition before executing this branch. |
| 786 | `last = estimate_model(households, stage, draws, warm_start=last)` | Compute or initialize `last` using the displayed expression. |
| 787 | `display_results(last, OUTPUT_DIR)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 788 | `if RUN_PROFILES and last is not None:` | Apply the displayed condition before executing this branch. |
| 789 | `profile_parameters(last, households, draws, OUTPUT_DIR)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 790 | `if RUN_DRAW_SENSITIVITY and last is not None and last["stage"] != "homogeneous":` | Apply the displayed condition before executing this branch. |
| 791 | `for n, seed in [(N_DRAWS, SEED + 1), (2 * N_DRAWS, SEED)]:` | Repeat the enclosed calculation over the displayed sequence. |
| 792–794 | `fit = estimate_model( households, last["stage"], make_draws(n, seed), warm_start=last )` | Compute or initialize `fit` using the displayed expression. |
| 795 | `display_results(fit, OUTPUT_DIR, suffix=f"_draws{n}_seed{seed}")` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 796 | `if RUN_RECOVERY:` | Apply the displayed condition before executing this branch. |
| 797 | `run_recovery(OUTPUT_DIR, draws)` | Execute the displayed call or control-flow statement within this part of the calculation. |
| 800 | `if __name__ == "__main__":` | Apply the displayed condition before executing this branch. |
| 801 | `main()` | Execute the displayed call or control-flow statement within this part of the calculation. |

## Verification of this delivery

The module compiled and synthetic checks verified across-store aggregation, the eight-inside/zero-outside rule, calendar-gap decay, zero initial stocks, stability under repeated shares, centering equivalence for random utilities, and household integration against the explicit formula. Small simulated fits exercised all three main stages, the curvature calculation, and the fixed-effect benchmark. The production parquet and first-stage regression have not been run in this environment; these checks establish code behavior, not empirical identification or valid real-data estimates.
