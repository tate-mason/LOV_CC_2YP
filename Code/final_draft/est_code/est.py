"""Weekly yogurt demand: fixed or mixed logit, with or without gamma.

Read README.md first. Run `python3 est.py --help` for server paths and options.
The raw data are not required for importing or testing the likelihood.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp

from price_control import check_dma, other_dma_instruments, add_price_controls
from bootstrap_se import bootstrap_standard_errors

FLAVORS = ["other", "berry", "plain"]
DATA_DIR = Path("/scratch/dtm63837/Kilts_Panel/Data")
# Set this to the flavor-reference CSV on the Slurm server.
FLAVOR_FILE = DATA_DIR / "recovered_flavor_reference.csv"


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


def read_flavors(path):
    """Use the same plain-first classification as data_code/summ_stats.py."""
    frame = pd.read_csv(path, dtype={"upc": str})
    missing = {"upc", "flavor_descr"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["upc"] = normalize_upcs(frame["upc"])
    if frame["upc"].isna().any() or frame["upc"].eq("").any():
        raise ValueError(f"{path}: flavor reference contains missing UPCs.")
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


def prepare_data(args):
    """Read purchase lines, aggregate baskets, and attach RMS menu prices.

    Polars filters/projects on the server before converting smaller tables to
    pandas. RMS price is used directly, without a multiplier adjustment.
    Prices are equal-weight means over UPCs in a store/week/flavor.
    No prices are filled from the household's chosen product or future weeks.
    """
    flavor = read_flavors(args.flavors)
    scan = pl.scan_parquet(args.sample)
    required = [
        "household_code",
        "panel_year",
        "purchase_date",
        "trip_code_uc",
        "upc",
        "store_code_uc",
        "quantity",
    ]
    if not args.no_price_control:
        required.append(pl.col(args.household_dma_column).alias("dma_cd"))
    scan = scan.select(required).filter(
        pl.col("panel_year").is_between(args.first_year, args.last_year)
    )
    if args.max_households:
        ids = (
            scan.select("household_code")
            .unique()
            .sort("household_code")
            .limit(args.max_households)
            .collect()["household_code"]
        )
        scan = scan.filter(pl.col("household_code").is_in(ids.implode()))
    lines = scan.collect().to_pandas()
    if lines.empty:
        raise ValueError("No purchase lines in selected sample.")
    lines["upc"] = normalize_upcs(lines["upc"])
    lines = lines.merge(flavor, on="upc", how="left", validate="many_to_one")
    if lines["flavor"].isna().any():
        args.output.mkdir(parents=True, exist_ok=True)
        unmatched = (
            lines.loc[lines["flavor"].isna()]
            .groupby("upc", dropna=False)
            .agg(purchase_lines=("quantity", "size"), packages=("quantity", "sum"))
            .reset_index()
            .sort_values("purchase_lines", ascending=False)
        )
        unmatched.to_csv(args.output / "unmatched_purchase_upcs.csv", index=False)
        print(unmatched.head(20).to_string(index=False), flush=True)
        raise ValueError(
            "Purchased UPCs still lack flavor labels after formatting normalization; "
            f"see {args.output / 'unmatched_purchase_upcs.csv'}."
        )
    q = lines["quantity"].to_numpy(float)
    if not np.isfinite(q).all() or (q <= 0).any() or not np.allclose(q, np.round(q)):
        raise ValueError("Quantity must be positive integer package counts.")
    dates = pd.to_datetime(lines["purchase_date"], errors="raise")
    # Recompute Saturday weeks: upstream panel_week.py maps Sundays backwards.
    lines["week"] = (
        dates + pd.to_timedelta((5 - dates.dt.dayofweek) % 7, unit="D")
    ).dt.normalize()
    keys = ["household_code", "week"]
    weekly = (
        lines.groupby(keys + ["flavor"])["quantity"]
        .sum()
        .unstack("flavor", fill_value=0)
        .reindex(columns=FLAVORS, fill_value=0)
        .reset_index()
    )
    # Select one exposure store, but count purchases at ALL stores exactly once.
    stores = lines.groupby(keys + ["store_code_uc"], as_index=False)["quantity"].sum()
    stores = stores.sort_values(
        keys + ["quantity", "store_code_uc"], ascending=[True, True, False, True]
    )
    stores = stores.drop_duplicates(keys)[keys + ["store_code_uc"]]
    weekly = weekly.merge(stores, on=keys, validate="one_to_one")
    if not args.no_price_control:
        lines["dma_cd"] = check_dma(lines["dma_cd"], "consumer")
        if (lines.groupby(keys)["dma_cd"].nunique() > 1).any():
            raise ValueError("Multiple consumer DMAs within a household-week.")
        dmas = lines[keys + ["dma_cd"]].drop_duplicates()
        weekly = weekly.merge(dmas, on=keys, validate="one_to_one")

    rms = pl.scan_parquet(args.rms)
    cols = ["upc", "store_code_uc", "week_end", "price"]
    if not args.no_price_control:
        cols.append(pl.col(args.retail_dma_column).alias("dma_code"))
    # Keep the national RMS scan available: IVs must not be limited to stores
    # visited by the selected households.
    national = (
        rms.filter(pl.col("product_module_code").is_in([3603, 3612]))
        .filter(
            pl.col("week_end").is_between(
                args.first_year * 10000 + 101, (args.last_year + 1) * 10000 + 107
            )
        )
        .select(cols)
        .with_columns(pl.col("price").cast(pl.Float64).alias("unit_price"))
    )
    retail = (
        national.filter(
            pl.col("store_code_uc").is_in(weekly["store_code_uc"].unique().tolist())
        )
        .collect()
        .to_pandas()
    )
    retail["upc"] = normalize_upcs(retail["upc"])
    retail = retail.merge(flavor, on="upc", how="left", validate="many_to_one")
    if retail["flavor"].isna().any():
        args.output.mkdir(parents=True, exist_ok=True)
        unmatched = retail.loc[retail["flavor"].isna(), ["upc"]].drop_duplicates()
        unmatched.to_csv(args.output / "unmatched_retail_upcs.csv", index=False)
        raise ValueError(
            "RMS UPCs still lack flavor labels after formatting normalization; "
            f"see {args.output / 'unmatched_retail_upcs.csv'}."
        )
    retail["week"] = week_dates(retail["week_end"])
    rk = ["store_code_uc", "week", "upc"]
    if retail.duplicated(rk).any():
        raise ValueError("Duplicate RMS store/week/UPC keys; audit retail merge first.")
    good = np.isfinite(retail["unit_price"]) & (retail["unit_price"] > 0)
    print(f"Invalid RMS prices excluded: {(~good).sum():,}", flush=True)
    menus = (
        retail.loc[good]
        .groupby(["store_code_uc", "week", "flavor"])["unit_price"]
        .mean()
        .unstack("flavor")
        .reindex(columns=FLAVORS)
        .rename(columns={f: "price_" + f for f in FLAVORS})
        .reset_index()
    )
    weekly = weekly.merge(
        menus, on=["store_code_uc", "week"], how="left", validate="many_to_one"
    )
    if not args.no_price_control:
        # Only these UPCs are needed, but their prices come from ALL retail DMAs.
        upcs = retail["upc"].unique().tolist()
        national = national.with_columns(normalized_upc_expr()).filter(
            pl.col("upc").is_in(upcs)
        )
        duplicate = (
            national.group_by(["store_code_uc", "week_end", "upc"])
            .len()
            .filter(pl.col("len") > 1)
            .limit(1)
            .collect()
        )
        if len(duplicate):
            raise ValueError("Duplicate national RMS store/week/UPC keys; audit merge.")
        valid_national = national.filter(
            pl.col("unit_price").is_finite() & (pl.col("unit_price") > 0)
        )
        dma_prices = (
            valid_national.group_by(["upc", "week_end", "dma_code"])
            .agg(pl.col("unit_price").mean().alias("dma_price"))
            .collect()
            .to_pandas()
        )
        dma_prices["week"] = week_dates(dma_prices.pop("week_end"))
        exposure_upcs = retail.loc[good, ["store_code_uc", "week", "upc", "flavor"]]
        weekly = other_dma_instruments(weekly, exposure_upcs, dma_prices)
    weekly = weekly.sort_values(keys).reset_index(drop=True)
    weekly = add_stocks(weekly, args.habit_weight)
    # Missing menus are excluded from the likelihood AFTER updating history.
    price_cols = ["price_" + f for f in FLAVORS]
    complete = np.isfinite(weekly[price_cols]).all(axis=1)
    print(f"Weeks excluded for incomplete menus: {(~complete).sum():,}", flush=True)
    if not args.no_price_control:
        iv_complete = np.isfinite(weekly[["iv_" + f for f in FLAVORS]]).all(axis=1)
        print(
            f"Additional weeks excluded for missing other-DMA IVs: "
            f"{(complete & ~iv_complete).sum():,}",
            flush=True,
        )
        complete &= iv_complete
    weekly = weekly.loc[complete].copy()
    # Count usable household-weeks after all menu/IV exclusions.
    repeated = weekly.groupby("household_code")["week"].transform("nunique") >= 2
    print(
        f"Households excluded for fewer than two usable weeks: "
        f"{weekly.loc[~repeated, 'household_code'].nunique():,}",
        flush=True,
    )
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
    if not all(np.isfinite(x).all() for x in [counts, prices, xi]):
        raise ValueError("Nonfinite estimation inputs.")
    cf_cols = ["cf_" + f for f in FLAVORS]
    control = all(c in frame for c in cf_cols)
    cf = frame[cf_cols].to_numpy(float) if control else np.zeros_like(prices)
    if not np.isfinite(cf).all():
        raise ValueError("Nonfinite price-control residuals.")
    return dict(
        counts=counts,
        prices=prices,
        xi=xi,
        cf=cf,
        control=control,
        hh=hh,
        labels=labels,
        n_households=len(labels),
        outside=bool(capacity),
    )


def make_draws(number, seed):
    """Fixed antithetic normal draws, reused throughout optimization."""
    if number < 2 or number % 2:
        raise ValueError("Use an even number of draws, at least two.")
    z = np.random.default_rng(seed).standard_normal((number // 2, 4))
    return np.vstack([z, -z])


def parameter_names(model, outside, control=False):
    names = (["inside_constant"] if outside else []) + ["beta_berry", "beta_plain"]
    names += ["mu_log_minus_alpha"]
    if model in ("homogeneous", "heterogeneous"):
        names += ["mean_gamma_" + f for f in FLAVORS]
    if model in ("mixed_naive", "heterogeneous"):
        names += ["sd_log_minus_alpha"]
    if model == "heterogeneous":
        names += ["sd_gamma_" + f for f in FLAVORS]
    if control:
        names += ["price_control"]
    return names


def household_loglike(theta, data, draws, model):
    """L_i = mean_r exp(sum_t sum_j count_itj * log P_itj,r).

    Gamma is normal; alpha = -exp(mu + sd*z). Beta never varies by draw.
    Process one draw at a time to avoid a weeks x draws x products array.
    The multinomial combinatorial constant is omitted (independent of theta).
    """
    p = dict(zip(parameter_names(model, data["outside"], data["control"]), theta))
    beta = np.array([0.0, p["beta_berry"], p["beta_plain"]])
    beta += p.get("inside_constant", 0.0)
    loglikes = []
    active_draws = (
        draws if model in ("mixed_naive", "heterogeneous") else np.zeros((1, 4))
    )
    for z in active_draws:
        alpha = -np.exp(
            p["mu_log_minus_alpha"] + p.get("sd_log_minus_alpha", 0.0) * z[0]
        )
        gamma = np.array(
            [
                p.get("mean_gamma_" + f, 0.0) + p.get("sd_gamma_" + f, 0.0) * z[j + 1]
                for j, f in enumerate(FLAVORS)
            ]
        )
        utility = (
            beta
            + alpha * data["prices"]
            + gamma * data["xi"]
            + p.get("price_control", 0.0) * data["cf"]
        )
        if data["outside"]:
            utility = np.column_stack([utility, np.zeros(len(utility))])
        logprob = utility - logsumexp(utility, axis=1, keepdims=True)
        row_ll = (data["counts"] * logprob).sum(axis=1)
        loglikes.append(
            np.bincount(data["hh"], weights=row_ll, minlength=data["n_households"])
        )
    return logsumexp(loglikes, axis=0) - np.log(len(active_draws))


def optimize_demand(data, args, initial=None, number_starts=None, quiet=False):
    draws = make_draws(args.draws, args.seed)
    names = parameter_names(args.model, data["outside"], data["control"])
    start = np.zeros(len(names))
    bounds = []
    for k, name in enumerate(names):
        if name.startswith("sd_"):
            start[k] = 0.3  # Zero SD is a stationary point: start away from zero.
            bounds.append((0.0, 3.0))
        elif name == "mu_log_minus_alpha":
            bounds.append((-6.0, 3.0))
        else:
            bounds.append((-20.0, 20.0))
    if initial is not None:
        start = np.asarray(initial).copy()
    rng = np.random.default_rng(args.seed + 1)
    results = []

    def objective(theta):
        return -household_loglike(theta, data, draws, args.model).mean()

    for s in range(args.starts if number_starts is None else number_starts):
        initial = start.copy()
        if s:
            initial += rng.normal(0, 0.5, len(start))
            initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        fit = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": args.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        results.append(fit)
        if not quiet:
            print(
                f"Start {s + 1}: NLL={fit.fun * data['n_households']:.4f}; "
                f"converged={fit.success}",
                flush=True,
            )
    # Keep the lowest objective, even if it has not converged yet.
    best = min(results, key=lambda r: r.fun)
    return best, results, bounds


def fit_model(data, args, frame=None):
    """Estimate demand, run the two-stage bootstrap, and save readable outputs."""
    best, results, bounds = optimize_demand(data, args)
    names = parameter_names(args.model, data["outside"], data["control"])
    table = pd.DataFrame({"parameter": names, "estimate": best.x})
    table["at_bound"] = [
        min(abs(v - lo), abs(v - hi)) < 1e-4 for v, (lo, hi) in zip(best.x, bounds)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "point_estimates.csv", index=False)
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
    # An explicit missing column distinguishes skipped inference from zero SEs.
    for column in ["std_error", "ci_lower", "ci_upper"]:
        if column not in table:
            table[column] = np.nan
    table.to_csv(args.output / "estimates.csv", index=False)
    summary = dict(
        model=args.model,
        households=data["n_households"],
        weeks=len(data["hh"]),
        capacity=args.capacity,
        inference=inference,
        converged=bool(best.success),
        message=str(best.message),
        nll=float(best.fun * data["n_households"]),
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


def main(default_model="heterogeneous"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=DATA_DIR / "yogurt_estimation_sample.parquet"
    )
    parser.add_argument("--rms", type=Path, default=DATA_DIR / "merged_rms.parquet")
    parser.add_argument("--flavors", type=Path, default=FLAVOR_FILE)
    parser.add_argument("--output", type=Path, default=Path("estimation_output"))
    parser.add_argument(
        "--model",
        choices=["naive", "homogeneous", "mixed_naive", "heterogeneous"],
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
        help="Explicitly disable the other-DMA price control function",
    )
    parser.add_argument("--household-dma-column", default="dma_cd")
    parser.add_argument("--retail-dma-column", default="dma_code")
    parser.add_argument(
        "--se-method", choices=["bootstrap", "none"], default="bootstrap"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=199)
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
