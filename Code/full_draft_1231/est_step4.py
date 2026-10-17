"""Household-week own-flavor estimation. See est_step4.md before interpreting results."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp
from rich.console import Console
from rich.table import Table
import statsmodels.formula.api as smf

console = Console()
MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUTPUT_DIR = Path("step4_output")
CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {name: j for j, name in enumerate(CATEGORIES)}

# Flip these switches independently. Expensive diagnostics are off by default.
RUN_WITHIN_DIAGNOSTICS = True
RUN_WITHIN_FIXED_EFFECTS = True
RUN_HOMOGENEOUS = True
RUN_RANDOM_BETA = True
RUN_RANDOM_BETA_GAMMA = True
RUN_PROFILES = False
RUN_RECOVERY = False
RUN_DRAW_SENSITIVITY = False
COMPUTE_STANDARD_ERRORS = True
USE_CONTROL_FUNCTION = True

# Seven is the outside-choice threshold, NOT a cap on recorded inside purchases.
WEEKLY_CAPACITY = 7
OUTSIDE_RULE = "threshold"  # threshold: max(7-Q, 0); strict: reject Q > 7
DELTA_DISCOUNT = 0.9
MIN_WEEKS = 2
N_DRAWS = 128
N_STARTS = 3
MAX_ITER = 600
SEED = 306
MAX_HOUSEHOLDS = None  # Set an integer for a reproducible development sample.
PROFILE_POINTS = 9
PROFILE_WIDTH = 1.0
RECOVERY_HOUSEHOLDS = 60
RECOVERY_WEEKS = 24
RECOVERY_REPLICATIONS = 3


def make_draws(n_draws=N_DRAWS, seed=SEED):
    # Antithetic draws reduce simulation noise; the same draws persist within HH.
    half = np.random.default_rng(seed).standard_normal(((n_draws + 1) // 2, 5))
    return np.vstack([half, -half])[:n_draws]


# =========================================================
# PART 1: DATA AND STABLE OWN-FLAVOR STOCK
# =========================================================
def load_and_preprocess():
    schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

    exprs = [
        pl.col("quantity").cast(pl.Float64),
        pl.col("head_age").cast(pl.Int64),
        pl.col("household_income").cast(pl.Int64),
        pl.col("household_size").cast(pl.Int64),
        pl.col("serving_per_container_cd").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
    ]

    if "product_module_code_hms" in schema_names:
        exprs.append(pl.col("product_module_code_hms").cast(pl.Int64))

    scan_q = pl.scan_parquet(MERGED_PATH).with_columns(exprs)

    if "product_module_code_hms" in schema_names:
        scan_q = scan_q.filter(pl.col("product_module_code_hms").is_in([3612, 3603]))

    merged_df = (
        scan_q.filter(pl.col("household_size") == 1)
        .filter(pl.col("serving_per_container_cd").is_in([67181961, 65622705]))
        .collect()
        .to_pandas()
    )

    merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)
    merged_df["flavor_cd"] = pd.to_numeric(
        merged_df["flavor_cd"], errors="coerce"
    ).fillna(0)

    merged_master = merged_df.copy()
    # Infinite prices are missing observations, not usable first-stage outcomes.
    merged_master["price"] = merged_master["price"].replace(
        [np.inf, -np.inf], np.nan
    )

    descr_cols = [
        c
        for c in merged_df.columns
        if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"])
    ]

    merged_df["full_text"] = ""
    for c in descr_cols:
        merged_df["full_text"] += " " + merged_df[c].fillna("").astype(str)
    merged_df["full_text"] = merged_df["full_text"].str.lower()

    berry_regex = r"berry|straw|blue|rasp|black|cran|cherry|wildberry"
    plain_regex = r"plain|unflavored"

    is_plain = merged_df["full_text"].str.contains(plain_regex, na=False)
    is_berry = merged_df["full_text"].str.contains(berry_regex, na=False) & (~is_plain)

    merged_master["flavor"] = np.select([is_plain, is_berry], [2, 1], default=0)

    market_price = (
        merged_master.groupby(["upc", "week_end", "market_name"])["price"]
        .mean()
        .reset_index()
    )
    totals = (
        market_price.groupby(["upc", "week_end"])["price"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "price_sum_all", "count": "n_markets_all"})
    )

    market_price = market_price.merge(totals, on=["upc", "week_end"], how="left")
    market_price["price_iv"] = (
        market_price["price_sum_all"] - market_price["price"]
    ) / (market_price["n_markets_all"] - 1)
    market_price["price_iv"] = market_price["price_iv"].replace(
        [np.inf, -np.inf], np.nan
    )

    merged_master = merged_master.merge(
        market_price[["upc", "week_end", "market_name", "price_iv"]],
        on=["upc", "week_end", "market_name"],
        how="left",
    )

    merged_master = merged_master[
        (merged_master["price"] > 0.1) | (merged_master["price"].isna())
    ]
    iv_res = smf.ols(
        "price ~ price_iv + brand_cd + C(week_end)", data=merged_master, missing="drop"
    ).fit()

    merged_master["iv_res"] = np.nan
    merged_master.loc[iv_res.model.data.row_labels, "iv_res"] = iv_res.resid

    valid_resids = merged_master["iv_res"].dropna()
    if len(valid_resids) > 0:
        res_mean = valid_resids.mean()
        res_std = valid_resids.std()
        merged_master["iv_res"] = (merged_master["iv_res"] - res_mean) / (
            res_std if res_std > 0 else 1.0
        )

    merged_master["iv_res"] = merged_master["iv_res"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    def map_flavor_category(flavor):
        if pd.isna(flavor):
            return "outside"
        elif flavor == 0:
            return "other"
        elif flavor == 1:
            return "berry"
        elif flavor == 2:
            return "plain"

    merged_master["category"] = merged_master["flavor"].map(map_flavor_category)
    inside_df = merged_master[merged_master["category"] != "outside"].copy()

    quantities = inside_df["quantity"].to_numpy(dtype=float)
    if (
        not np.isfinite(quantities).all()
        or np.any(quantities < 0)
        or not np.allclose(quantities, np.round(quantities))
    ):
        raise ValueError(
            "Source quantities must be finite nonnegative integer yogurt units."
        )
    inside_df = inside_df.loc[inside_df["quantity"] > 0].copy()
    # Aggregate all stores ONCE into household-week flavor counts.
    weekly = (
        inside_df.groupby(["household_code", "week_end", "category"])["quantity"]
        .sum()
        .unstack("category", fill_value=0)
    )
    weekly = weekly.reindex(columns=CATEGORIES[:3], fill_value=0).reset_index()
    weekly["week_end"] = pd.to_datetime(weekly["week_end"])

    # A single reference store per HH-week: most purchased units, deterministic ties.
    stores = inside_df.groupby(
        ["household_code", "week_end", "store_code_uc"], as_index=False
    )["quantity"].sum()
    stores["week_end"] = pd.to_datetime(stores["week_end"])
    stores = stores.sort_values(
        ["household_code", "week_end", "quantity", "store_code_uc"],
        ascending=[True, True, False, True],
    )
    stores = stores.drop_duplicates(["household_code", "week_end"])
    weekly = weekly.merge(
        stores[["household_code", "week_end", "store_code_uc"]], validate="one_to_one"
    )

    # Keep step2's transaction-based price proxies, with explicit fallback flags.
    inside_df["week_end"] = pd.to_datetime(inside_df["week_end"])
    store_prices = inside_df.groupby(["store_code_uc", "week_end", "category"])[
        ["price", "iv_res"]
    ].mean()
    week_prices = inside_df.groupby(["week_end", "category"])[
        ["price", "iv_res"]
    ].mean()
    overall = inside_df.groupby("category")[["price", "iv_res"]].mean()
    households = []
    for hh_id, group in weekly.groupby("household_code", sort=True):
        group = group.sort_values("week_end")
        if len(group) < MIN_WEEKS:
            continue
        prices, resids, counts, dates, fallback = [], [], [], [], []
        for row in group.itertuples(index=False):
            q = np.array([row.other, row.berry, row.plain], dtype=float)
            if np.any(q < 0) or not np.allclose(q, np.round(q)):
                raise ValueError("Quantities must be nonnegative integer yogurt units.")
            if OUTSIDE_RULE == "strict" and q.sum() > WEEKLY_CAPACITY:
                raise ValueError(
                    f"HH {hh_id}: weekly units exceed capacity; inspect units or use threshold rule."
                )
            if OUTSIDE_RULE not in ("strict", "threshold"):
                raise ValueError("Unknown OUTSIDE_RULE")
            counts.append(np.append(q, max(WEEKLY_CAPACITY - q.sum(), 0)))
            p, r, flags = [], [], []
            for cat in CATEGORIES[:3]:
                key = (row.store_code_uc, row.week_end, cat)
                wk_key = (row.week_end, cat)
                price, residual, flag = resolve_price_inputs(
                    store_prices, week_prices, overall, key, wk_key, cat
                )
                p.append(price)
                r.append(residual if USE_CONTROL_FUNCTION else 0.0)
                flags.append(flag)
            prices.append(p)
            resids.append(r)
            dates.append(row.week_end)
            fallback.append(flags)
        choices = np.asarray(counts)
        stock = compute_satiation_state(choices[:, :3], dates)
        households.append(
            dict(
                household_code=str(hh_id),
                prices=np.asarray(prices),
                resids=np.asarray(resids),
                choices=choices,
                dates=pd.DatetimeIndex(dates),
                stock=stock,
                fallback=np.asarray(fallback),
            )
        )
        if MAX_HOUSEHOLDS is not None and len(households) >= MAX_HOUSEHOLDS:
            break
    if not households:
        raise ValueError("No eligible households after preprocessing.")
    for hh in households:
        if not np.isfinite(hh["prices"]).all() or not np.isfinite(hh["resids"]).all():
            raise ValueError("Nonfinite price/control-function inputs.")
    center_states(households)
    return households


def resolve_price_inputs(store_prices, week_prices, overall, key, wk_key, cat):
    # An existing group can still have an entirely missing price column.
    # Fall back on VALUE validity, not merely presence of the group key.
    for table, lookup, flag in [
        (store_prices, key, 0),
        (week_prices, wk_key, 1),
        (overall, cat, 2),
    ]:
        if lookup not in table.index:
            continue
        values = table.loc[lookup]
        price = values["price"]
        if pd.isna(price) or not np.isfinite(price) or price <= 0.1:
            continue
        residual = values["iv_res"]
        # Preserve step2's explicit zero convention for missing control residuals.
        if pd.isna(residual) or not np.isfinite(residual):
            residual = 0.0
        return float(price), float(residual), flag
    raise ValueError(
        f"No finite price > 0.1 for flavor {cat!r}: store-week={key!r}. "
        "Store-week, week-flavor, and overall-flavor fallbacks all failed. "
        "Inspect source prices and sample restrictions."
    )


def compute_satiation_state(inside_counts, dates):
    # Stock BEFORE this week's choices. First observed stock is exactly zero.
    stock = np.zeros_like(inside_counts, dtype=float)
    for t in range(1, len(dates)):
        elapsed = (pd.Timestamp(dates[t]) - pd.Timestamp(dates[t - 1])).days / 7.0
        if elapsed <= 0:
            raise ValueError("Household dates must be unique and increasing.")
        previous = inside_counts[t - 1]
        shares = previous / previous.sum() if previous.sum() > 0 else np.zeros(3)
        stock[t] = DELTA_DISCOUNT**elapsed * (stock[t - 1] + shares)
    return stock


def center_states(households, reference=None):
    # Global centering preserves the model and zero-stock interpretation via conversion.
    if reference is None:
        reference = np.concatenate([np.log1p(h["stock"]) for h in households]).mean(
            axis=0
        )
    for hh in households:
        hh["z"] = np.log1p(hh["stock"]) - reference
        hh["reference"] = np.asarray(reference)
    return np.asarray(reference)


# =========================================================
# PART 2: WITHIN-HOUSEHOLD IDENTIFICATION DIAGNOSTICS
# =========================================================
def diagnose_within(households, output_dir):
    rows = []
    for hh in households:
        z = hh["z"]
        shares = hh["choices"][:, :3] / hh["choices"][:, :3].sum(axis=1, keepdims=True)
        # All estimation observations are purchase weeks, so the denominator is positive.
        for j, cat in enumerate(CATEGORIES[:3]):
            rows.append(
                dict(
                    household=hh["household_code"],
                    flavor=cat,
                    weeks=len(z),
                    within_sd=float(z[:, j].std()),
                    state_range=float(np.ptp(z[:, j])),
                    share_changes=int(np.count_nonzero(np.diff(shares[:, j]))),
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "within_household.csv", index=False)
    console.print(
        frame.groupby("flavor")[["within_sd", "state_range", "share_changes"]].agg(
            ["mean", "median"]
        )
    )
    # Residualize state against HH intercepts, then price, control, and calendar week.
    dates = np.concatenate([h["dates"].asi8 for h in households])
    _, week_idx = np.unique(dates, return_inverse=True)
    blocks = []
    for j, cat in enumerate(CATEGORIES[:3]):
        target = np.concatenate([h["z"][:, j] for h in households])
        price = np.concatenate([h["prices"][:, j] for h in households])
        resid = np.concatenate([h["resids"][:, j] for h in households])
        sizes = [len(h["z"]) for h in households]
        hh_idx = np.repeat(np.arange(len(sizes)), sizes)
        # Sparse projection avoids a dense household-dummy matrix.
        from scipy.sparse import coo_matrix, hstack
        from scipy.sparse.linalg import lsqr

        n = len(target)
        hd = coo_matrix(
            (np.ones(n), (np.arange(n), hh_idx)), shape=(n, len(sizes))
        ).tocsr()
        wd = coo_matrix((np.ones(n), (np.arange(n), week_idx))).tocsr()
        design = hstack([hd, wd, coo_matrix(np.column_stack([price, resid]))]).tocsr()
        residual = target - design @ lsqr(design, target, atol=1e-10, btol=1e-10)[0]
        within = np.concatenate(
            [h["z"][:, j] - h["z"][:, j].mean() for h in households]
        )
        blocks.append(
            dict(
                flavor=cat,
                within_sd=float(within.std()),
                residual_sd=float(residual.std()),
            )
        )
    pd.DataFrame(blocks).to_csv(
        output_dir / "residual_state_variation.csv", index=False
    )
    console.print(pd.DataFrame(blocks))
    all_choices = np.concatenate([h["choices"] for h in households])
    console.print(
        f"Households: {len(households)}; purchase weeks: {len(all_choices)}; weeks above seven: {(all_choices[:, :3].sum(axis=1) > 7).sum()}"
    )
    console.print(
        f"Price fallback fraction: {(np.concatenate([h['fallback'] for h in households]) > 0).mean():.3f}"
    )


# =========================================================
# PART 3: HOMOGENEOUS AND HOUSEHOLD RANDOM-COEFFICIENT LOGITS
# =========================================================
def estimate_within_fixed_effects(households, output_dir):
    # Unrestricted HH-by-flavor intercepts isolate time variation for shared slopes.
    # This is a short-panel sensitivity benchmark, not a bias-corrected estimator.
    n_hh = len(households)
    n_common = 5 if USE_CONTROL_FUNCTION else 4

    def objective_gradient(params):
        gamma, alpha = params[:3], params[3]
        intercepts = params[n_common:].reshape(n_hh, 3)
        gradient = np.zeros_like(params)
        value = 0.0
        for h, hh in enumerate(households):
            inside = intercepts[h] + gamma * hh["z"] + alpha * hh["prices"]
            if USE_CONTROL_FUNCTION:
                inside += params[4] * hh["resids"]
            u = np.column_stack([inside, np.zeros(len(inside))])
            lp = u - logsumexp(u, axis=1, keepdims=True)
            value -= np.sum(hh["choices"] * lp)
            error = (
                hh["choices"].sum(axis=1, keepdims=True) * np.exp(lp) - hh["choices"]
            )[:, :3]
            gradient[:3] += np.sum(error * hh["z"], axis=0)
            gradient[3] += np.sum(error * hh["prices"])
            if USE_CONTROL_FUNCTION:
                gradient[4] += np.sum(error * hh["resids"])
            gradient[n_common + 3 * h : n_common + 3 * h + 3] = error.sum(axis=0)
        return float(value), gradient

    initial = np.zeros(n_common + 3 * n_hh)
    initial[3] = -0.1
    bounds = [(None, None)] * len(initial)
    bounds[3] = (None, 0.0)
    fit = minimize(
        objective_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5),
    )
    names = ["gamma_other", "gamma_berry", "gamma_plain", "price"] + (
        ["control_function"] if USE_CONTROL_FUNCTION else []
    )
    pd.DataFrame(dict(parameter=names, estimate=fit.x[:n_common])).to_csv(
        output_dir / "within_fixed_effects.csv", index=False
    )
    metadata = dict(
        success=bool(fit.success),
        message=str(fit.message),
        fun=float(fit.fun),
        warning="Short-panel incidental-parameter bias and separated HH flavor intercepts are possible; no bias correction or SEs.",
    )
    (output_dir / "within_fixed_effects.json").write_text(
        json.dumps(metadata, indent=2)
    )
    console.print(
        "Within-household fixed-effect benchmark",
        dict(zip(names, fit.x[:n_common])),
        metadata,
    )
    return fit


def parameter_names(stage):
    names = [
        "constant",
        "beta_berry",
        "beta_plain",
        "gamma_other",
        "gamma_berry",
        "gamma_plain",
        "price",
    ]
    if USE_CONTROL_FUNCTION:
        names += ["control_function"]
    if stage in ("random_beta", "random_beta_gamma"):
        names += ["sd_beta_berry", "sd_beta_plain"]
    if stage == "random_beta_gamma":
        names += ["sd_gamma_other", "sd_gamma_berry", "sd_gamma_plain"]
    return names


def utility_draws(params, hh, stage, draws):
    n = 1 if stage == "homogeneous" else len(draws)
    beta = np.tile([0.0, params[1], params[2]], (n, 1))
    gamma = np.tile(params[3:6], (n, 1))
    offset = 8 if USE_CONTROL_FUNCTION else 7
    if stage != "homogeneous":
        beta[:, 1:] += draws[:, :2] * params[offset : offset + 2]
    if stage == "random_beta_gamma":
        deviations = draws[:, 2:5] * params[offset + 2 : offset + 5]
        gamma += deviations
        # Anchor independent random intercepts at ZERO stock despite global centering.
        beta += deviations * hh.get("reference", np.zeros(3))
    base = params[0] + params[6] * hh["prices"]
    if USE_CONTROL_FUNCTION:
        base += params[7] * hh["resids"]
    inside = (
        base[:, None, :] + beta[None, :, :] + gamma[None, :, :] * hh["z"][:, None, :]
    )
    return np.concatenate([inside, np.zeros((len(inside), n, 1))], axis=2)


def household_loglikes(params, households, stage, draws):
    ll = []
    for hh in households:
        u = utility_draws(params, hh, stage, draws)
        log_probs = u - logsumexp(u, axis=2, keepdims=True)
        # SUM weeks and counts FIRST; integrate household types SECOND.
        conditional = np.sum(hh["choices"][:, None, :] * log_probs, axis=(0, 2))
        ll.append(logsumexp(conditional) - np.log(len(conditional)))
    return np.asarray(ll)


def total_objective(params, households, stage, draws):
    value = -household_loglikes(params, households, stage, draws).sum()
    return float(value) if np.isfinite(value) else 1e100


def numerical_hessian(function, params):
    # Differentiate numerical gradients; symmetrize before curvature diagnostics.
    from scipy.optimize._numdiff import approx_derivative

    gradient = lambda x: approx_derivative(
        lambda y: np.array([function(y)]), x, method="3-point"
    ).ravel()
    matrix = approx_derivative(gradient, params, method="3-point")
    return (matrix + matrix.T) / 2


def estimate_model(
    households,
    stage,
    draws,
    warm_start=None,
    starts=N_STARTS,
    compute_se=COMPUTE_STANDARD_ERRORS,
):
    names = parameter_names(stage)
    x0 = np.zeros(len(names))
    x0[3:6] = -0.1
    x0[6] = -0.1
    for i, name in enumerate(names):
        if name.startswith("sd_"):
            x0[i] = 0.2
        if warm_start is not None and name in warm_start["names"]:
            x0[i] = warm_start["params"][warm_start["names"].index(name)]
    # Direct SDs, including zero: reporting and objective use identical units.
    bounds = [
        (0.0, 5.0)
        if name.startswith("sd_")
        else (None, 0.0)
        if name == "price"
        else (None, None)
        for name in names
    ]
    objective = lambda p: total_objective(p, households, stage, draws)
    results = []
    rng = np.random.default_rng(SEED)
    for k in range(starts):
        initial = x0.copy() if k == 0 else x0 + rng.normal(0, 0.15, len(x0))
        for i, (lo, hi) in enumerate(bounds):
            initial[i] = np.clip(
                initial[i], -np.inf if lo is None else lo, np.inf if hi is None else hi
            )
        results.append(
            minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=MAX_ITER, ftol=1e-10, gtol=1e-5),
            )
        )
    converged = [r for r in results if r.success]
    best = min(converged or results, key=lambda r: r.fun)
    se = np.full(len(names), np.nan)
    eigenvalues = np.array([])
    condition = None
    boundary = [
        name
        for name, p, (lo, hi) in zip(names, best.x, bounds)
        if (lo is not None and abs(p - lo) < 1e-5)
        or (hi is not None and abs(p - hi) < 1e-5)
    ]
    if compute_se:
        from scipy.optimize._numdiff import approx_derivative

        hessian = numerical_hessian(objective, best.x)
        eigenvalues = np.linalg.eigvalsh(hessian)
        condition = float(np.linalg.cond(hessian))
        if eigenvalues.min() > 0 and not boundary:
            # Household score sandwich, conditional on the fitted first-stage residuals.
            scores = approx_derivative(
                lambda p: household_loglikes(p, households, stage, draws),
                best.x,
                method="3-point",
            )
            bread = np.linalg.inv(hessian)
            covariance = bread @ (scores.T @ scores) @ bread
            se = np.sqrt(np.maximum(0, np.diag(covariance)))
    return dict(
        stage=stage,
        names=names,
        params=best.x,
        se=se,
        fun=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        boundary=boundary,
        hessian_eigenvalues=eigenvalues,
        hessian_condition=condition,
        starts=[dict(fun=float(r.fun), success=bool(r.success)) for r in results],
    )


def display_results(result, output_dir, suffix=""):
    table = Table(title=result["stage"] + suffix)
    for name in ["Parameter", "Estimate", "HH sandwich SE"]:
        table.add_column(name)
    for name, value, se in zip(result["names"], result["params"], result["se"]):
        table.add_row(name, f"{value:.5f}", f"{se:.5f}")
    console.print(table)
    console.print(
        f"Converged: {result['success']}; negative LL: {result['fun']:.4f}; {result['message']}"
    )
    console.print(
        f"Boundary parameters: {result['boundary']}; Hessian condition: {result['hessian_condition']}"
    )
    pd.DataFrame(
        dict(parameter=result["names"], estimate=result["params"], se=result["se"])
    ).to_csv(output_dir / f"{result['stage']}{suffix}.csv", index=False)
    serial = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }
    (output_dir / f"{result['stage']}{suffix}.json").write_text(
        json.dumps(serial, indent=2)
    )


# =========================================================
# PART 4: OPTIONAL PROFILE, DRAW, AND DYNAMIC RECOVERY CHECKS
# =========================================================
def profile_parameters(result, households, draws, output_dir):
    names, fitted = result["names"], result["params"]
    rows = []
    for index, name in enumerate(names):
        if not name.startswith("gamma_"):
            continue
        # Reoptimize ALL nuisance coefficients, including beta, at each fixed gamma.
        keep = np.arange(len(fitted)) != index

        def expand(free, fixed):
            p = fitted.copy()
            p[keep], p[index] = free, fixed
            return p

        bounds = [
            (0, 5)
            if n.startswith("sd_")
            else (None, 0)
            if n == "price"
            else (None, None)
            for i, n in enumerate(names)
            if i != index
        ]
        for fixed in np.linspace(
            fitted[index] - PROFILE_WIDTH, fitted[index] + PROFILE_WIDTH, PROFILE_POINTS
        ):
            fit = minimize(
                lambda free: total_objective(
                    expand(free, fixed), households, result["stage"], draws
                ),
                fitted[keep],
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=MAX_ITER, ftol=1e-10),
            )
            p = expand(fit.x, fixed)
            rows.append(
                dict(
                    parameter=name,
                    fixed=fixed,
                    nll=float(fit.fun),
                    delta_nll=float(fit.fun - result["fun"]),
                    beta_berry=p[1],
                    beta_plain=p[2],
                    success=bool(fit.success),
                )
            )
    pd.DataFrame(rows).to_csv(output_dir / "gamma_profiles.csv", index=False)


def simulate_panel(seed, n_hh=RECOVERY_HOUSEHOLDS, n_weeks=RECOVERY_WEEKS):
    # Simulate choices and then update histories: never reuse observed endogenous stocks.
    rng = np.random.default_rng(seed)
    truth = np.array(
        [-0.4, 0.6, -0.2, -0.7, -1.0, -0.8, -0.4]
        + ([0.0] if USE_CONTROL_FUNCTION else [])
        + [0.5, 0.4, 0.25, 0.3, 0.2]
    )
    households = []
    for h in range(n_hh):
        dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D")
        hh = dict(
            household_code=str(h),
            prices=rng.uniform(0.5, 3, (n_weeks, 3)),
            resids=rng.normal(size=(n_weeks, 3)),
            choices=np.zeros((n_weeks, 4)),
            dates=dates,
            stock=np.zeros((n_weeks, 3)),
            fallback=np.zeros((n_weeks, 3)),
        )
        type_draw = rng.normal(size=(1, 5))
        for t in range(n_weeks):
            if t:
                previous = hh["choices"][t - 1, :3]
                shares = previous / previous.sum() if previous.sum() else np.zeros(3)
                hh["stock"][t] = DELTA_DISCOUNT * (hh["stock"][t - 1] + shares)
            one = {key: hh[key][t : t + 1] for key in ("prices", "resids")}
            one["z"] = np.log1p(hh["stock"][t : t + 1])
            u = utility_draws(truth, one, "random_beta_gamma", type_draw)[0, 0]
            hh["choices"][t] = rng.multinomial(
                WEEKLY_CAPACITY, np.exp(u - logsumexp(u))
            )
        households.append(hh)
    reference = center_states(households)
    truth[0] += truth[3] * reference[0]
    truth[1:3] += truth[4:6] * reference[1:3] - truth[3] * reference[0]
    return households, truth


def run_recovery(output_dir, draws):
    rows = []
    for replication in range(RECOVERY_REPLICATIONS):
        households, truth = simulate_panel(SEED + 1000 + replication)
        fit = estimate_model(households, "random_beta_gamma", draws, compute_se=False)
        for name, true, estimate in zip(fit["names"], truth, fit["params"]):
            rows.append(
                dict(
                    replication=replication,
                    parameter=name,
                    truth=true,
                    estimate=estimate,
                    error=estimate - true,
                    success=fit["success"],
                )
            )
    pd.DataFrame(rows).to_csv(output_dir / "recovery.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Run recovery without reading Nielsen data",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draws = make_draws()
    settings = {
        key: value
        for key, value in globals().items()
        if key.isupper() and isinstance(value, (str, int, float, bool, type(None)))
    }
    (OUTPUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))
    if args.simulate_only:
        run_recovery(OUTPUT_DIR, draws)
        return
    households = load_and_preprocess()
    if RUN_WITHIN_DIAGNOSTICS:
        diagnose_within(households, OUTPUT_DIR)
    pd.DataFrame(
        {"flavor": CATEGORIES[:3], "log_stock_reference": households[0]["reference"]}
    ).to_csv(OUTPUT_DIR / "state_reference.csv", index=False)
    if RUN_WITHIN_FIXED_EFFECTS:
        estimate_within_fixed_effects(households, OUTPUT_DIR)
    last = None
    for stage, enabled in [
        ("homogeneous", RUN_HOMOGENEOUS),
        ("random_beta", RUN_RANDOM_BETA),
        ("random_beta_gamma", RUN_RANDOM_BETA_GAMMA),
    ]:
        if enabled:
            last = estimate_model(households, stage, draws, warm_start=last)
            display_results(last, OUTPUT_DIR)
    if RUN_PROFILES and last is not None:
        profile_parameters(last, households, draws, OUTPUT_DIR)
    if RUN_DRAW_SENSITIVITY and last is not None and last["stage"] != "homogeneous":
        for n, seed in [(N_DRAWS, SEED + 1), (2 * N_DRAWS, SEED)]:
            fit = estimate_model(
                households, last["stage"], make_draws(n, seed), warm_start=last
            )
            display_results(fit, OUTPUT_DIR, suffix=f"_draws{n}_seed{seed}")
    if RUN_RECOVERY:
        run_recovery(OUTPUT_DIR, draws)


if __name__ == "__main__":
    main()
