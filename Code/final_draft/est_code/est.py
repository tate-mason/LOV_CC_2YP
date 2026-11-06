"""Weekly yogurt demand: fixed or mixed logit, with or without gamma.

Read README.md first. Run `python3 est.py --help` for server paths and options.
The raw data are not required for importing or testing the likelihood.
"""

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp

from price_control import check_dma, add_price_controls
from bootstrap_se import bootstrap_standard_errors
from hessian_se import hessian_standard_errors
from wtp import implied_wtp

FLAVORS = ["other", "berry", "plain"]
DATA_DIR = Path("/scratch/dtm63837/Kilts_Panel/Data")
# Set this to the flavor-reference CSV on the Slurm server.
FLAVOR_FILE = DATA_DIR / "summary_outputs/estimation_flavor_reference.csv"


def normalize_upcs(values):
    """Canonical numeric UPC keys without floating-point conversion."""
    text = values.astype("string").str.strip()
    numeric = text.str.fullmatch(r"[0-9]+(?:\.0+)?", na=False)
    canonical = text.str.replace(r"\.0+$", "", regex=True).str.lstrip("0")
    return text.where(~numeric, canonical.replace("", "0"))


def normalized_upc_expr():
    text = pl.col("upc").cast(pl.String).str.strip_chars()
    numeric = text.str.contains(r"^[0-9]+(?:\.0+)?$")
    canonical = text.str.replace(r"\.0+$", "").str.strip_chars_start("0")
    return (
        pl.when(numeric)
        .then(pl.when(canonical == "").then(pl.lit("0")).otherwise(canonical))
        .otherwise(text)
        .alias("upc")
    )


@server_only
def read_flavors(path, products_extra=None):
    """Use the same plain-first classification as data_code/summ_stats.py."""
    frame = pd.read_csv(path, dtype={"upc": str})
    if {"upc", "flavor"}.issubset(frame.columns):
        # Consume the exact mapping saved by summ_stats; no recovery or relabeling.
        frame["upc"] = normalize_upcs(frame["upc"])
        if (
            frame["upc"].isna().any()
            or frame["upc"].eq("").any()
            or frame["upc"].duplicated().any()
        ):
            raise ValueError(f"{path}: saved mapping needs unique, nonmissing UPCs.")
        if not frame["flavor"].isin(FLAVORS).all():
            raise ValueError(f"{path}: saved mapping must use other, berry, plain.")
        print(
            f"Loaded {len(frame):,} resolved UPC flavor labels from {path}.", flush=True
        )
        return frame[["upc", "flavor"]]
    missing = {"upc", "flavor_descr"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["upc"] = normalize_upcs(frame["upc"])
    if frame["upc"].isna().any() or frame["upc"].eq("").any():
        raise ValueError(f"{path}: flavor reference contains missing UPCs.")
    if products_extra is not None:
        products_extra = Path(products_extra)
        if products_extra.exists():
            products = pl.scan_parquet(products_extra)
            names = products.collect_schema().names()
            if not {"upc", "flavor_descr"}.issubset(names):
                raise ValueError(
                    f"{products_extra}: needs upc and flavor_descr columns."
                )
            columns = [
                c
                for c in ["upc", "flavor_descr", "product_descr", "panel_year"]
                if c in names
            ]
            products = (
                products.select(columns)
                .with_columns(normalized_upc_expr())
                .join(
                    pl.DataFrame(
                        {"upc": frame["upc"].dropna().unique().tolist()}
                    ).lazy(),
                    on="upc",
                    how="anti",
                )
            )
            products = products.filter(
                pl.col("upc").is_not_null()
                & pl.col("flavor_descr").is_not_null()
                & ~pl.col("flavor_descr")
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["", "na", "n/a", "nan", "null", "none"])
            )
            if "panel_year" in names:
                products = products.group_by("upc").agg(
                    [
                        pl.col(c)
                        .sort_by("panel_year", nulls_last=True)
                        .first()
                        .alias(c)
                        for c in columns
                        if c != "upc"
                    ]
                )
            else:
                products = products.unique(
                    subset="upc", keep="first", maintain_order=True
                )
            recovered = products.collect(engine="streaming").to_pandas()
            frame = pd.concat([frame, recovered], ignore_index=True)
            print(
                f"Recovered flavor labels for {len(recovered):,} additional UPCs from {products_extra}.",
                flush=True,
            )
        else:
            print(
                f"Flavor recovery source not found: {products_extra}; using CSV reference only.",
                flush=True,
            )
    text = frame["flavor_descr"].fillna("")
    if "product_descr" in frame.columns:
        text = text + " " + frame["product_descr"].fillna("")
    text = text.str.lower()
    frame["flavor"] = np.select(
        [
            text.str.contains("plain|unflavored"),
            text.str.contains("berry|straw|blue|rasp|black|cran|cherry|wildberry"),
        ],
        ["plain", "berry"],
        default="other",
    )
    # Match summ_stats: retain missing labels and use the first CSV row per UPC.
    conflicts = (frame.groupby("upc")["flavor"].nunique() > 1).sum()
    if conflicts:
        print(
            f"{conflicts:,} UPCs have conflicting flavor classifications; "
            "keeping the first flavor in CSV row order.",
            flush=True,
        )
    return frame[["upc", "flavor"]].drop_duplicates(subset="upc", keep="first")


def week_dates(values):
    """Parse YYYYMMDD safely (integer dates are NOT nanoseconds)."""
    return pd.to_datetime(
        values.astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="raise",
    )


@server_only
def prepare_data(args):
    """Use only the merged purchase table, including its existing prices.

    The IV pools all flavors: the same-week average observed price outside the
    consumer's DMA. Repeated purchases do not duplicate store/UPC price quotes.
    """
    base = pl.scan_parquet(args.sample)
    names = base.collect_schema().names()
    if "household_size" not in names:
        raise ValueError("Estimation requires household_size to restrict purchases to single-person households.")
    base = base.filter(pl.col("panel_year").is_between(args.first_year, args.last_year))
    if "flavor" not in names:
        products_extra = getattr(args, "products_extra", None) or DATA_DIR / "RMS/combined/products_extra.parquet"
        flavor = read_flavors(args.flavors, products_extra)
        base = base.with_columns(normalized_upc_expr()).join(pl.from_pandas(flavor).lazy(), on="upc", how="left")
    else:
        base = base.with_columns(normalized_upc_expr())
    price_column = next((c for c in ["price", "price_rms"] if c in names), None)
    if price_column is None:
        raise ValueError("Merged purchase table needs a price or price_rms column.")
    consumer_dma = getattr(args, "household_dma_column", "dma_cd")
    columns = ["household_code", "household_size", "panel_year", "purchase_date", "trip_code_uc", "upc",
               "store_code_uc", "quantity", "flavor", pl.col(price_column).cast(pl.Float64).alias("price")]
    if not args.no_price_control:
        columns.append(pl.col(consumer_dma).alias("dma_cd"))
    base = base.select(columns).with_columns(
        pl.col("purchase_date").cast(pl.String).str.slice(0, 10).str.to_date())
    base = base.filter(pl.col("price").is_finite() & (pl.col("price") > 0))
    base = base.with_columns((pl.col("purchase_date") + pl.duration(
        days=(6 - pl.col("purchase_date").dt.weekday()) % 7)).alias("week"))
    # Keep the full merged table for prices/IVs, even in a household-limited run.
    # Restrict demand histories before weekly aggregation; retain all quotes for menus/IVs.
    selected = base.filter(pl.col("household_size").cast(pl.Float64, strict=False) == 1)
    if args.max_households:
        ids = (selected.select("household_code").unique().sort("household_code")
               .limit(args.max_households).collect(engine="streaming")["household_code"])
        selected = selected.filter(pl.col("household_code").is_in(ids.implode()))
    print(f"Loading merged purchase table: {args.sample}", flush=True)
    lines = selected.collect(engine="streaming").to_pandas()
    if lines.empty:
        raise ValueError("No purchase lines in selected sample.")
    if lines["flavor"].isna().any() or not lines["flavor"].isin(FLAVORS).all():
        raise ValueError("Merged purchase table has missing or invalid flavor labels.")
    q = lines["quantity"].to_numpy(float)
    if not np.isfinite(q).all() or (q <= 0).any() or not np.allclose(q, np.round(q)):
        raise ValueError("Quantity must be positive integer package counts.")
    lines["week"] = pd.to_datetime(lines["week"])
    keys = ["household_code", "week"]
    weekly = (lines.groupby(keys + ["flavor"])["quantity"].sum().unstack("flavor", fill_value=0)
              .reindex(columns=FLAVORS, fill_value=0).reset_index())
    stores = lines.groupby(keys + ["store_code_uc"], as_index=False)["quantity"].sum()
    stores = stores.sort_values(keys + ["quantity", "store_code_uc"], ascending=[True, True, False, True])
    stores = stores.drop_duplicates(keys)[keys + ["store_code_uc"]]
    weekly = weekly.merge(stores, on=keys, validate="one_to_one")
    if not args.no_price_control:
        lines["dma_cd"] = check_dma(lines["dma_cd"], "consumer")
        if (lines.groupby(keys)["dma_cd"].nunique() > 1).any():
            raise ValueError("Multiple consumer DMAs within a household-week.")
        weekly = weekly.merge(lines[keys + ["dma_cd"]].drop_duplicates(), on=keys, validate="one_to_one")

    quote_keys = ["store_code_uc", "week", "upc"]
    price_columns = quote_keys + ["flavor", "price"]
    if not args.no_price_control:
        price_columns.append("dma_cd")
    quotes = (base.filter(pl.col("price").is_finite() & (pl.col("price") > 0))
              .select(price_columns).unique())
    # A store/week/UPC should have one price and one flavor. Consumer DMA may
    # differ when consumers from different markets visit the same store.
    local = quotes.select(quote_keys + ["flavor", "price"]).unique()
    conflicting = (local.group_by(quote_keys).len().filter(pl.col("len") > 1)
                   .limit(1).collect(engine="streaming"))
    if len(conflicting):
        raise ValueError("Conflicting merged prices or flavors for the same store/week/UPC.")
    menus = (local.group_by(["store_code_uc", "week", "flavor"])
             .agg(pl.col("price").mean()).collect(engine="streaming").to_pandas())
    menus["week"] = pd.to_datetime(menus["week"])
    menus = (menus.pivot(index=["store_code_uc", "week"], columns="flavor", values="price")
             .reindex(columns=FLAVORS).rename(columns={f: "price_" + f for f in FLAVORS}).reset_index())
    weekly = weekly.merge(menus, on=["store_code_uc", "week"], how="left", validate="many_to_one")
    print("Using merged prices directly; no separate RMS data are loaded.", flush=True)
    if not args.no_price_control:
        invalid_dma = quotes.select(
            ((pl.col("dma_cd").cast(pl.Float64, strict=False).is_null())
             | (pl.col("dma_cd").cast(pl.Float64, strict=False) <= 0)
             | (pl.col("dma_cd").cast(pl.Float64, strict=False) != pl.col("dma_cd").cast(pl.Float64, strict=False).floor()))
            .fill_null(True).any()).collect(engine="streaming").item()
        if invalid_dma:
            raise ValueError("Merged price records contain invalid dma_cd values for the other-DMA IV.")
        # Pool across ALL flavors. Deduplicate quote repeats within each DMA.
        by_dma = (quotes.with_columns(pl.col("dma_cd").cast(pl.Float64).cast(pl.Int64))
                  .group_by(["week", "dma_cd"]).agg(
                      pl.col("price").sum().alias("dma_sum"), pl.len().alias("dma_count"))
                  .collect(engine="streaming").to_pandas())
        by_dma["week"] = pd.to_datetime(by_dma["week"])
        totals = (by_dma.groupby("week", as_index=False)[["dma_sum", "dma_count"]].sum()
                  .rename(columns={"dma_sum": "total_sum", "dma_count": "total_count"}))
        weekly = weekly.merge(totals, on="week", how="left", validate="many_to_one")
        weekly = weekly.merge(by_dma, on=["week", "dma_cd"], how="left", validate="many_to_one")
        denominator = weekly["total_count"] - weekly["dma_count"].fillna(0)
        weekly["iv_overall"] = ((weekly["total_sum"] - weekly["dma_sum"].fillna(0))
                                / denominator.where(denominator > 0))
        for f in FLAVORS:
            weekly["iv_" + f] = weekly["iv_overall"]
        weekly = weekly.drop(columns=["total_sum", "total_count", "dma_sum", "dma_count"])
        print("IV: overall same-week mean price outside consumer dma_cd, pooled across flavors.", flush=True)
    weekly = add_stocks(weekly.sort_values(keys).reset_index(drop=True), args.habit_weight)
    price_cols = ["price_" + f for f in FLAVORS]
    available = np.isfinite(weekly[price_cols].to_numpy(float)) & (weekly[price_cols].to_numpy(float) > 0)
    counts = weekly[FLAVORS].to_numpy(float)
    complete = available.any(axis=1) & ((counts == 0) | available).all(axis=1)
    for j, f in enumerate(FLAVORS):
        weekly["available_" + f] = available[:, j]
    print(f"Weeks excluded for purchased flavors without a covered price: {(~complete).sum():,}", flush=True)
    if not args.no_price_control:
        iv_cols = ["iv_" + f for f in FLAVORS]
        iv_complete = np.isfinite(weekly[iv_cols]).all(axis=1)
        print(f"Additional weeks excluded for missing other-DMA IVs: {(complete & ~iv_complete).sum():,}", flush=True)
        complete &= iv_complete
    weekly = weekly.loc[complete].copy()
    repeated = weekly.groupby("household_code")["week"].transform("nunique") >= 2
    print(f"Households excluded for fewer than two usable weeks: {weekly.loc[~repeated, 'household_code'].nunique():,}", flush=True)
    weekly = weekly.loc[repeated].reset_index(drop=True)
    weekly["outside"] = np.maximum(args.capacity - weekly[FLAVORS].sum(axis=1), 0)
    if weekly.empty:
        raise ValueError("No households with at least two usable purchase weeks.")
    return weekly


def add_stocks(weekly, habit_weight=0.7):
    """All past observed occasions, delta=1; no rolling window or decay.

    S_jt = sum_{s<t} share_js
    N_jt = sum_{k != j} share_k,t-1  (immediately preceding calendar week)
    Xi_jt = (1-habit_weight)*N_jt - habit_weight*S_jt

    Positive gamma rewards recent other-flavor consumption and penalizes own
    fatigue. Negative gamma represents habit. A purchase occasion is an observed
    household purchase week. Current purchases enter history only AFTER Xi is
    recorded. The first observed occasion starts with zero history.
    """
    result = (
        weekly.sort_values(["household_code", "week"]).reset_index(drop=True).copy()
    )
    stocks = np.zeros((len(result), 3))
    novelty = np.zeros_like(stocks)
    for _, group in result.groupby("household_code", sort=False):
        stock = np.zeros(3)
        previous_share = np.zeros(3)
        previous_week = None
        for index, row in group.iterrows():
            stocks[index] = stock
            if previous_week is not None and (row["week"] - previous_week).days == 7:
                novelty[index] = previous_share.sum() - previous_share
            counts = row[FLAVORS].to_numpy(float)
            previous_share = counts / counts.sum()
            stock = stock + previous_share
            previous_week = row["week"]
    for j, name in enumerate(FLAVORS):
        result["stock_" + name] = stocks[:, j]
        result["novelty_" + name] = novelty[:, j]
        result["xi_" + name] = (1 - habit_weight) * novelty[
            :, j
        ] - habit_weight * stocks[:, j]
    return result


def make_arrays(frame, capacity):
    """Use household codes to sum log likelihoods before integrating types."""
    hh, labels = pd.factorize(frame["household_code"], sort=True)
    counts = frame[FLAVORS + (["outside"] if capacity else [])].to_numpy(float)
    prices = frame[["price_" + f for f in FLAVORS]].to_numpy(float)
    xi = frame[["xi_" + f for f in FLAVORS]].to_numpy(float)
    history = {key: frame[[prefix + f for f in FLAVORS]].to_numpy(float)
               for key, prefix in [("novelty", "novelty_"), ("stock", "stock_")]
               if all(prefix + f in frame for f in FLAVORS)}
    if any(not np.isfinite(value).all() for value in history.values()):
        raise ValueError("Nonfinite separate history inputs.")
    available = np.isfinite(prices) & (prices > 0)
    if not all(np.isfinite(x).all() for x in [counts, xi]):
        raise ValueError("Nonfinite estimation inputs.")
    if ((counts[:, :3] > 0) & ~available).any() or not available.any(axis=1).all():
        raise ValueError("Purchased flavors must have covered prices and a nonempty choice set.")
    # Zero placeholders are masked out of utility, never treated as free goods.
    prices = np.where(available, prices, 0.)
    cf_cols = ["cf_" + f for f in FLAVORS]
    control = all(c in frame for c in cf_cols)
    cf = frame[cf_cols].to_numpy(float) if control else np.zeros_like(prices)
    if not np.isfinite(cf).all():
        raise ValueError("Nonfinite price-control residuals.")
    return dict(
        counts=counts,
        prices=prices,
        available=available,
        xi=xi,
        **history,
        cf=cf,
        control=control,
        hh=hh,
        labels=labels,
        n_households=len(labels),
        outside=bool(capacity),
    )


def make_draws(number, seed, model=None):
    """Fixed antithetic normal draws, reused throughout optimization."""
    if number < 2 or number % 2:
        raise ValueError("Use an even number of draws, at least two.")
    z = np.random.default_rng(seed).standard_normal((number // 2, 3 if model == "heterogeneous_history" else 2))
    return np.vstack([z, -z])


def parameter_names(model, outside, control=False):
    names = (["inside_constant"] if outside else []) + ["beta_berry", "beta_plain"]
    names += ["mu_log_minus_alpha"]
    if model in ("homogeneous", "heterogeneous"):
        names += ["mean_gamma"]
    if model in ("mixed_naive", "heterogeneous"):
        names += ["sd_log_minus_alpha"]
    if model == "heterogeneous":
        names += ["sd_gamma"]
    if model == "mixture":
        names += ["mean_gamma", "sd_log_minus_alpha", "sd_gamma",
                  "gamma_mean_gap", "sd_gamma_2", "mixture_weight"]
    if model in ("separate_history", "heterogeneous_history"):
        names += ["mean_novelty", "mean_own_stock", "sd_log_minus_alpha"]
    if model == "heterogeneous_history":
        names += ["sd_novelty", "sd_own_stock"]
    if control:
        names += ["price_control"]
    return names


def mixture_components(theta, data):
    """Ordered component means prevent label switching; alpha is shared."""
    names = parameter_names("mixture", data["outside"], data["control"])
    p = dict(zip(names, theta))
    normal_names = parameter_names("heterogeneous", data["outside"], data["control"])
    first = np.array([p[n] for n in normal_names])
    second = first.copy()
    second[normal_names.index("mean_gamma")] += p["gamma_mean_gap"]
    second[normal_names.index("sd_gamma")] = p["sd_gamma_2"]
    return names, normal_names, first, second, p["mixture_weight"]


def mixture_loglike_score(theta, data, draws, score=False):
    """Mix whole household histories, never individual weeks."""
    names, normal_names, first, second, weight = mixture_components(theta, data)
    if score:
        ll1, g1 = demand_objective_gradient(first, data, draws, "heterogeneous", household=True)
        ll2, g2 = demand_objective_gradient(second, data, draws, "heterogeneous", household=True)
    else:
        ll1 = household_loglike(first, data, draws, "heterogeneous")
        ll2 = household_loglike(second, data, draws, "heterogeneous")
    terms = np.array([np.log(weight) + ll1, np.log1p(-weight) + ll2])
    ll = logsumexp(terms, axis=0)
    if not score:
        return ll
    responsibility = np.exp(terms - ll)
    gradient = np.zeros((data["n_households"], len(names)))
    for k, name in enumerate(normal_names):
        if name == "sd_gamma":
            gradient[:, names.index(name)] = responsibility[0] * g1[:, k]
            gradient[:, names.index("sd_gamma_2")] = responsibility[1] * g2[:, k]
        else:
            gradient[:, names.index(name)] = responsibility[0] * g1[:, k] + responsibility[1] * g2[:, k]
    gradient[:, names.index("gamma_mean_gap")] = responsibility[1] * g2[:, normal_names.index("mean_gamma")]
    gradient[:, names.index("mixture_weight")] = responsibility[0] / weight - responsibility[1] / (1 - weight)
    return -ll.mean(), -gradient.mean(axis=0)


def history_utility(p, data, z, model):
    if model in ("separate_history", "heterogeneous_history"):
        novelty = p["mean_novelty"] + p.get("sd_novelty", 0.) * z[1]
        own_stock = p["mean_own_stock"]
        if model == "heterogeneous_history":
            if len(z) < 3:
                raise ValueError("heterogeneous_history requires three-dimensional draws.")
            own_stock += p["sd_own_stock"] * z[2]
        return novelty * data["novelty"] + own_stock * data["stock"]
    gamma = p.get("mean_gamma", 0.) + p.get("sd_gamma", 0.) * z[1]
    return gamma * data["xi"]


def household_loglike(theta, data, draws, model):
    """L_i = mean_r exp(sum_t sum_j count_itj * log P_itj,r).

    Gamma is normal and common across flavors; alpha = -exp(mu + sd*z). Beta never varies by draw.
    Process one draw at a time to avoid a weeks x draws x products array.
    The multinomial combinatorial constant is omitted (independent of theta).
    """
    if model == "mixture":
        return mixture_loglike_score(theta, data, draws)
    p = dict(zip(parameter_names(model, data["outside"], data["control"]), theta))
    beta = np.array([0.0, p["beta_berry"], p["beta_plain"]])
    beta += p.get("inside_constant", 0.0)
    loglikes = []
    active_draws = (
        draws if model in ("mixed_naive", "heterogeneous", "separate_history", "heterogeneous_history") else np.zeros((1, 2))
    )
    for z in active_draws:
        alpha = -np.exp(
            p["mu_log_minus_alpha"] + p.get("sd_log_minus_alpha", 0.0) * z[0]
        )
        gamma = p.get("mean_gamma", 0.) + p.get("sd_gamma", 0.) * z[1]
        utility = (
            beta
            + alpha * data["prices"]
            + history_utility(p, data, z, model)
            + p.get("price_control", 0.0) * data["cf"]
        )
        utility = np.where(data["available"], utility, -np.inf)
        if data["outside"]:
            utility = np.column_stack([utility, np.zeros(len(utility))])
        logprob = utility - logsumexp(utility, axis=1, keepdims=True)
        row_ll = (data["counts"] * np.where(data["counts"] > 0, logprob, 0.)).sum(axis=1)
        loglikes.append(
            np.bincount(data["hh"], weights=row_ll, minlength=data["n_households"])
        )
    return logsumexp(loglikes, axis=0) - np.log(len(active_draws))


def demand_objective_gradient(theta, data, draws, model, household=False):
    """Exact simulated likelihood score, weighted by household type posteriors."""
    if model == "mixture":
        return mixture_loglike_score(theta, data, draws, score=True)
    names = parameter_names(model, data["outside"], data["control"])
    p = dict(zip(names, theta))
    active = draws if model in ("mixed_naive", "heterogeneous", "separate_history", "heterogeneous_history") else np.zeros((1, 2))
    likelihoods, scores = [], []
    total = data["counts"].sum(axis=1)
    for z in active:
        alpha = -np.exp(p["mu_log_minus_alpha"] + p.get("sd_log_minus_alpha", 0.) * z[0])
        gamma = p.get("mean_gamma", 0.) + p.get("sd_gamma", 0.) * z[1]
        utility = (np.array([0., p["beta_berry"], p["beta_plain"]]) + p.get("inside_constant", 0.)
                   + alpha * data["prices"] + history_utility(p, data, z, model) + p.get("price_control", 0.) * data["cf"])
        utility = np.where(data["available"], utility, -np.inf)
        if data["outside"]:
            utility = np.column_stack([utility, np.zeros(len(utility))])
        logprob = utility - logsumexp(utility, axis=1, keepdims=True)
        row_ll = (data["counts"] * np.where(data["counts"] > 0, logprob, 0.)).sum(axis=1)
        likelihoods.append(np.bincount(data["hh"], weights=row_ll, minlength=data["n_households"]))
        residual = data["counts"][:, :3] - total[:, None] * np.exp(logprob[:, :3])
        price_score = (residual * data["prices"]).sum(axis=1) * alpha
        columns = {"inside_constant": residual.sum(axis=1), "beta_berry": residual[:, 1],
                   "beta_plain": residual[:, 2], "mu_log_minus_alpha": price_score,
                   "sd_log_minus_alpha": price_score * z[0],
                   "price_control": (residual * data["cf"]).sum(axis=1)}
        columns["mean_gamma"] = (residual * data["xi"]).sum(axis=1)
        columns["sd_gamma"] = columns["mean_gamma"] * z[1]
        if model in ("separate_history", "heterogeneous_history"):
            columns["mean_novelty"] = (residual * data["novelty"]).sum(axis=1)
            columns["mean_own_stock"] = (residual * data["stock"]).sum(axis=1)
            columns["sd_novelty"] = columns["mean_novelty"] * z[1]
            if model == "heterogeneous_history":
                columns["sd_own_stock"] = columns["mean_own_stock"] * z[2]
        scores.append(np.column_stack([np.bincount(data["hh"], weights=columns[n],
                                                   minlength=data["n_households"]) for n in names]))
    likelihoods = np.asarray(likelihoods)
    normalizer = logsumexp(likelihoods, axis=0)
    weights = np.exp(likelihoods - normalizer)
    household_scores = np.einsum("rh,rhp->hp", weights, np.asarray(scores))
    if household:
        return normalizer - np.log(len(active)), household_scores
    gradient = -household_scores.mean(axis=0)
    return -(normalizer - np.log(len(active))).mean(), gradient


def optimize_demand(data, args, initial=None, number_starts=None, quiet=False):
    draws = make_draws(args.draws, args.seed, args.model)
    names = parameter_names(args.model, data["outside"], data["control"])
    start = np.zeros(len(names))
    bounds = []
    for k, name in enumerate(names):
        if name.startswith("sd_"):
            start[k] = 0.3  # Zero SD is a stationary point: start away from zero.
            bounds.append((0.0, 3.0))
        elif name == "gamma_mean_gap":
            start[k] = 0.4
            bounds.append((0.0, 40.0))
        elif name == "mixture_weight":
            start[k] = 0.5
            bounds.append((0.001, 0.999))
        elif name == "mu_log_minus_alpha":
            bounds.append((-6.0, 3.0))
        else:
            bounds.append((-20.0, 20.0))
    if args.model == "mixture":
        start[names.index("mean_gamma")] = -0.2
    if initial is not None:
        start = np.asarray(initial).copy()
    rng = np.random.default_rng(args.seed + 1)
    results = []

    def objective(theta):
        return demand_objective_gradient(theta, data, draws, args.model)

    for s in range(args.starts if number_starts is None else number_starts):
        initial = start.copy()
        if s:
            initial += rng.normal(0, 0.5, len(start))
            initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        began = time.monotonic()
        iterations = [0]

        def progress(theta):
            iterations[0] += 1
            if not quiet and iterations[0] % 10 == 0:
                print(f"Start {s + 1}: iteration {iterations[0]}, elapsed {time.monotonic() - began:.0f}s", flush=True)

        fit = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            callback=progress,
            bounds=bounds,
            options={"maxiter": args.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        results.append(fit)
        if not quiet:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "starts_checkpoint.json").write_text(json.dumps([
                dict(theta=r.x.tolist(), nll=float(r.fun * data["n_households"]),
                     converged=bool(r.success), iterations=int(r.nit), evaluations=int(r.nfev))
                for r in results], indent=2))
        if not quiet:
            print(
                f"Start {s + 1}: NLL={fit.fun * data['n_households']:.4f}; "
                f"converged={fit.success}",
                flush=True,
            )
    # Keep the lowest objective, even if it has not converged yet.
    best = min(results, key=lambda r: r.fun)
    return best, results, bounds


def plot_gamma_distribution(table, output, converged=True):
    """Plot the fitted population distribution, not uncertainty in its mean."""
    estimates = table.set_index("parameter")["estimate"]
    if "mean_gamma" not in estimates:
        return  # Naive models have no gamma parameter.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    mean = float(estimates["mean_gamma"])
    sd = float(estimates.get("sd_gamma", 0.))
    fig, ax = plt.subplots(figsize=(7, 5))
    if "mixture_weight" in estimates:
        weight = float(estimates["mixture_weight"])
        mean2 = mean + float(estimates["gamma_mean_gap"])
        sd2 = float(estimates["sd_gamma_2"])
        x = np.linspace(min(mean - 4 * sd, mean2 - 4 * sd2) - .01,
                        max(mean + 4 * sd, mean2 + 4 * sd2) + .01, 1001)
        density = np.zeros_like(x)
        negative = 0.
        for mu, sigma, share in [(mean, sd, weight), (mean2, sd2, 1 - weight)]:
            if sigma > 0:
                component = share * norm.pdf(x, loc=mu, scale=sigma)
                density += component
                ax.plot(x, component, linestyle=":", label=f"Component (weight {share:.1%})")
                negative += share * norm.cdf(0, loc=mu, scale=sigma)
            else:
                ax.axvline(mu, linestyle=":", label=f"Point mass (weight {share:.1%})")
                negative += share * (mu < 0)
        ax.plot(x, density, color="steelblue", label="Mixture continuous density")
        ax.set_ylabel("Population density")
        ax.legend()
        description = f"Component means = {mean:.4g}, {mean2:.4g}; P(gamma < 0) = {negative:.1%}"
    elif sd > 0:
        x = np.linspace(mean - 4 * sd, mean + 4 * sd, 801)
        density = norm.pdf(x, loc=mean, scale=sd)
        ax.plot(x, density, color="steelblue")
        ax.fill_between(x, density, where=x < 0, color="steelblue", alpha=.25)
        ax.set_ylabel("Population density")
        negative = norm.cdf(0, loc=mean, scale=sd)
        description = f"Mean = {mean:.4g}; SD = {sd:.4g}; P(gamma < 0) = {negative:.1%}"
    else:
        ax.axvline(mean, color="steelblue", linewidth=2, label="Point mass (SD = 0)")
        ax.set_xlim(mean - 1, mean + 1)
        ax.set_yticks([])
        ax.set_ylabel("Point mass")
        ax.legend()
        description = f"All households: gamma = {mean:.4g} (SD = 0)"
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    title = "Fitted population gamma distribution"
    if not converged or table["at_bound"].any():
        title += " (provisional fit)"
    ax.set(xlabel="Gamma (common across flavors)", title=title)
    fig.text(.5, .02, description, ha="center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "gamma_distribution.pdf")
    plt.close(fig)


def plot_history_distributions(table, output, converged=True):
    """Export fitted marginal population distributions after every split fit."""
    from history_distributions import export_history_distributions
    export_history_distributions(table, output, converged)


@server_only
def fit_model(data, args, frame=None):
    """Estimate demand, compute requested inference, and save readable outputs."""
    best, results, bounds = optimize_demand(data, args)
    names = parameter_names(args.model, data["outside"], data["control"])
    table = pd.DataFrame({"parameter": names, "estimate": best.x})
    table["at_bound"] = [
        min(abs(v - lo), abs(v - hi)) < 1e-4 for v, (lo, hi) in zip(best.x, bounds)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "point_estimates.csv", index=False)
    plot_gamma_distribution(table, args.output, converged=best.success)
    plot_history_distributions(table, args.output, converged=best.success)
    wtp = implied_wtp(table, getattr(args, "habit_weight", .7))
    wtp["provisional_fit"] = not best.success or bool(table["at_bound"].any())
    wtp.to_csv(args.output / "implied_wtp.csv", index=False)
    inference = {"method": args.se_method, "status": "disabled"}
    if args.se_method == "bootstrap":
        if not best.success:
            inference["status"] = "main fit not converged; bootstrap not run"
        elif frame is None:
            raise ValueError("Bootstrap requires the weekly input frame.")
        else:

            def estimate(sample, initial):
                bootstrap_data = make_arrays(sample, args.capacity)
                fitted, _, _ = optimize_demand(
                    bootstrap_data, args, initial, args.bootstrap_starts, quiet=True
                )
                return fitted

            standard_errors, inference = bootstrap_standard_errors(
                frame, args, estimate, best.x, names
            )
            if standard_errors is not None:
                table = table.merge(
                    standard_errors, on="parameter", validate="one_to_one"
                )
    elif args.se_method == "hessian":
        if not best.success:
            inference["status"] = "main fit not converged; Hessian inference not run"
        else:
            draws = make_draws(args.draws, args.seed, args.model)
            gradient = lambda theta: demand_objective_gradient(theta, data, draws, args.model)[1]
            standard_errors, inference = hessian_standard_errors(
                best.x, gradient, data["n_households"], bounds, names, args.output)
            if standard_errors is not None:
                table = table.merge(standard_errors, on="parameter", validate="one_to_one")
    # An explicit missing column distinguishes skipped inference from zero SEs.
    for column in ["std_error", "ci_lower", "ci_upper"]:
        if column not in table:
            table[column] = np.nan
    table.to_csv(args.output / "estimates.csv", index=False)
    summary = dict(
        model=args.model,
        price_control=data["control"],
        history_specification="separate" if args.model in ("separate_history", "heterogeneous_history") else "combined_or_none",
        sample_restriction="household_size == 1 on each purchase record",
        households=data["n_households"],
        weeks=len(data["hh"]),
        capacity=args.capacity,
        inference=inference,
        converged=bool(best.success),
        message=str(best.message),
        nll=float(best.fun * data["n_households"]),
        parameters=len(names),
        aic=float(2 * best.fun * data["n_households"] + 2 * len(names)),
        bic=float(2 * best.fun * data["n_households"] + np.log(data["n_households"]) * len(names)),
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        starts=[
            dict(
                nll=float(r.fun * data["n_households"]),
                converged=bool(r.success),
                theta=r.x.tolist(),
            )
            for r in results
        ],
    )
    (args.output / "fit.json").write_text(json.dumps(summary, indent=2))
    print("Inference: " + inference["status"], flush=True)
    print(table.to_string(index=False))
    if not best.success or table["at_bound"].any():
        print("Provisional fit: inspect convergence/bounds before interpretation.")
    return best


@server_only
def main(default_model="heterogeneous", default_no_price_control=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=DATA_DIR / "summary_outputs/yogurt_estimation_sample.parquet",
    )
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument(
        "--products-extra",
        type=Path,
        help="Annual flavor metadata; defaults to RMS/combined/products_extra.parquet beside the RMS data directory.",
    )
    parser.add_argument("--output", type=Path, default=Path("/scratch/dtm63837/Kilts_Panel/results/estimation_split"))
    parser.add_argument(
        "--model",
        choices=["naive", "homogeneous", "mixed_naive", "heterogeneous", "mixture", "separate_history", "heterogeneous_history"],
        default=default_model,
    )
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--last-year", type=int, default=2015)
    parser.add_argument("--max-households", type=int)
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="0: conditional flavor demand; 7: outside=max(7-Q,0)",
    )
    parser.add_argument(
        "--no-price-control",
        action="store_true",
        default=default_no_price_control,
        help="Disable the price control using the overall other-DMA mean-price IV",
    )
    parser.add_argument("--household-dma-column", default="dma_cd")
    parser.add_argument(
        "--se-method", choices=["hessian", "bootstrap", "none"], default="hessian"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=49)
    parser.add_argument("--bootstrap-starts", type=int, default=2)
    parser.add_argument("--habit-weight", type=float, default=0.7)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--seed", type=int, default=306)
    args = parser.parse_args()
    if not 0 <= args.habit_weight <= 1 or args.capacity < 0 or args.starts < 1:
        parser.error("Check habit weight, capacity, and starts.")
    if args.first_year > args.last_year or (
        args.max_households is not None and args.max_households < 1
    ):
        parser.error("Check years and household limit.")
    if args.draws < 2 or args.draws % 2 or args.maxiter < 1:
        parser.error("Use an even draw count >= 2 and maxiter >= 1.")
    if args.se_method == "bootstrap" and (
        args.bootstrap_reps < 20 or args.bootstrap_starts < 1
    ):
        parser.error("Bootstrap requires at least 20 replications and one start.")
    frame = prepare_data(args)
    if not args.no_price_control:
        frame, first_stage = add_price_controls(frame)
        args.output.mkdir(parents=True, exist_ok=True)
        first_stage.to_csv(args.output / "first_stage.csv", index=False)
        print(first_stage.to_string(index=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "weekly_inputs.csv", index=False)
    fit_model(make_arrays(frame, args.capacity), args, frame)


if __name__ == "__main__":
    main()
