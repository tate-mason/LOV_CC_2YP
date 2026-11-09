"""Demand estimation. Edit Parameters, then run python3 demand/estimation.py.

Read from top to bottom: inputs -> weekly data -> histories -> probabilities ->
likelihood -> optimization -> standard errors -> saved results.
No calculations are imported from final_draft. All seven model choices remain.
"""

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
import argparse
import json
import os
import sys
import tempfile
from pprint import pformat

import numpy as np
import pandas as pd

# Use a writable plotting cache, including on a shared research server.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "final_simple_matplotlib"))

import time
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t
from scipy.stats import norm



# === Parameters: edit these values === #

# Paths are ordinary Python values. No JSON knowledge is required.
DATA_DIR = Path("/scratch/dtm63837/Kilts_Panel/Data")
sample = DATA_DIR / "summary_outputs/yogurt_estimation_sample.parquet"
flavors = DATA_DIR / "summary_outputs/estimation_flavor_reference.csv"
products_extra = None                 # Optional flavor metadata for older inputs.
output = Path("/scratch/dtm63837/Kilts_Panel/results") / "estimation_split"

# Main model: common beta, household-specific alpha, novelty, and own-stock habit.
# Other choices: naive, homogeneous, mixed_naive, mixture,
#               separate_history, heterogeneous_history.
model = "heterogeneous_history"
first_year = 2006
last_year = 2015
max_households = None                # Use 100 for a development sample.
capacity = 7                         # Pricing setup; use 0 for conditional flavor choice.
no_price_control = False             # False means include the IV residual control.
household_dma_column = "dma_cd"
habit_weight = 0.7                   # Xi = (1-lambda)*novelty - lambda*own stock.
draws = 128                          # Even number of fixed integration draws.
starts = 3                           # Number of optimizer starting points.
maxiter = 600
seed = 306
se_method = "hessian"                # hessian, bootstrap, or none.
bootstrap_reps = 49
bootstrap_starts = 2


# Shared descriptor rules; the selected categories are recorded with each fit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flavor_specs import categories, classification_expression

FLAVORS = ["other", "berry", "plain"] # This column order is used in all arrays.
FLAVOR_FILE = flavors




# === 1. Check market codes and form price controls === #



def check_dma(values, description):
    """Normalize Nielsen's numeric DMA identifiers across panel and RMS files."""
    codes = pd.to_numeric(values, errors="coerce")
    valid = codes.notna() & (codes > 0) & (codes == np.floor(codes))
    if not valid.all():
        raise ValueError(f"Missing/invalid {description} DMA codes: {(~valid).sum():,}")
    return codes.astype("int64")


def other_dma_instruments(weekly, exposure_upcs, dma_prices):
    """Match each exposure-store UPC to its same-week price in OTHER DMAs.

    dma_prices has ONE mean package price per UPC/week/retail DMA. First average
    these means over all DMAs except the CONSUMER's DMA, then average over the
    exposure store's UPCs in a flavor. This matches the local price's UPC basket.
    """
    dma_prices = dma_prices.copy()
    dma_prices["dma_code"] = check_dma(dma_prices["dma_code"], "retail")
    pk = ["upc", "week"]
    totals = (dma_prices.groupby(pk)["dma_price"].agg(["sum", "count"])
              .rename(columns={"sum": "sum_dma", "count": "n_dma"}).reset_index())
    exposure = weekly[["store_code_uc", "week", "dma_cd"]].drop_duplicates()
    exposure = exposure.merge(exposure_upcs, on=["store_code_uc", "week"],
                              how="left", validate="many_to_many")
    exposure = exposure.merge(totals, on=pk, how="left", validate="many_to_one")
    own = dma_prices.rename(columns={"dma_code": "dma_cd", "dma_price": "own_dma_price"})
    exposure = exposure.merge(own, on=pk + ["dma_cd"], how="left", validate="many_to_one")
    present = exposure["own_dma_price"].notna()
    denominator = exposure["n_dma"] - present.astype(int)
    exposure["iv"] = ((exposure["sum_dma"] - exposure["own_dma_price"].fillna(0))
                      / denominator.where(denominator > 0))
    # Do not silently change the UPC basket when an instrument is missing.
    keys = ["store_code_uc", "week", "dma_cd", "flavor"]
    grouped = exposure.groupby(keys)["iv"].agg(["mean", "count", "size"])
    grouped.loc[grouped["count"] != grouped["size"], "mean"] = np.nan
    result = (grouped["mean"].unstack("flavor").reindex(columns=FLAVORS)
              .rename(columns={f: "iv_" + f for f in FLAVORS}).reset_index())
    return weekly.merge(result, on=["store_code_uc", "week", "dma_cd"], how="left",
                        validate="many_to_one")


# Repeatedly subtract group means until the week and DMA fixed effects have both been
# removed.
def absorb_effects(values, groups, tolerance=1e-10, maxiter=2000):
    """Remove week and DMA means without a huge matrix of dummy variables.

    Repeated demeaning is needed because DMA/week cells are unbalanced.
    Each household-week exposure has equal weight in the first stage.
    """
    residual = np.asarray(values, dtype=float).copy()
    codes = [pd.factorize(group, sort=True)[0] for group in groups]
    for _ in range(maxiter):
        previous = residual.copy()
        for code in codes:
            counts = np.bincount(code)
            means = np.zeros((len(counts), residual.shape[1]))
            np.add.at(means, code, residual)
            means /= counts[:, None]
            residual -= means[code]
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise ValueError("First-stage fixed-effect demeaning did not converge.")


# Run one price equation per flavor, removing week and DMA means and controlling for all
# three history indices.
def add_price_controls(frame):
    """OLS separately by flavor: price ~ IV + all three Xi + week FE + DMA FE.

    Refit this function in every household bootstrap sample. Return residuals
    in DOLLARS (no arbitrary rescaling) for the choice-model control function.
    """
    result = frame.copy()
    reports = []
    for flavor in FLAVORS:
        columns = ["price_" + flavor, "iv_" + flavor] + ["xi_" + f for f in FLAVORS]
        covered = np.isfinite(result["price_" + flavor]) & (result["price_" + flavor] > 0)
        sample = result.loc[covered]
        if sample.empty:
            raise ValueError(f"No covered prices for the {flavor} first stage.")
        hh, households = pd.factorize(sample["household_code"], sort=True)
        values = sample[columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite first-stage inputs for {flavor}.")
        within = absorb_effects(values, [sample["week"], sample["dma_cd"]])
        y, z, controls = within[:, 0], within[:, 1], within[:, 2:]
        # Partial out Xi before estimating the excluded instrument's coefficient.
        # This tolerates redundant Xi columns, but NOT a redundant instrument.
        z_partial = z - controls @ np.linalg.lstsq(controls, z, rcond=None)[0]
        y_partial = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        z_ss = z_partial @ z_partial
        if z_ss <= 1e-12 * max(1., values[:, 1] @ values[:, 1]):
            raise ValueError(f"No excluded-instrument variation after controls for {flavor}.")
        pi = (z_partial @ y_partial) / z_ss
        control_beta = np.linalg.lstsq(controls, y - pi*z, rcond=None)[0]
        residual = y - pi*z - controls @ control_beta
        if np.std(residual) < 1e-10:
            raise ValueError(f"First-stage residual for {flavor} is essentially zero.")
        result["cf_" + flavor] = 0.
        result.loc[covered, "cf_" + flavor] = residual
        # Household-clustered partial-IV t/F diagnostic (not a weak-IV-robust test).
        scores = np.bincount(hh, weights=z_partial * residual, minlength=len(households))
        pi_se = np.sqrt((scores @ scores) / z_ss**2)
        if len(households) > 1:
            pi_se *= np.sqrt(len(households) / (len(households) - 1))
        f_stat = float((pi / pi_se)**2) if pi_se > 0 else np.nan
        restricted_ss = y_partial @ y_partial
        partial_r2 = 1 - (residual @ residual) / restricted_ss if restricted_ss > 0 else np.nan
        reports.append(dict(flavor=flavor, iv_coefficient=float(pi),
                            iv_cluster_se=float(pi_se), iv_cluster_F=f_stat,
                            iv_pvalue=float(2*student_t.sf(np.sqrt(f_stat), len(households)-1))
                            if len(households) > 1 else np.nan,
                            partial_R2=float(partial_r2), residual_sd=float(np.std(residual)),
                            observations=len(sample), households=len(households)))
    return result, pd.DataFrame(reports)



# === 2. Read inputs and build household-week data === #



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
            raise ValueError(f"{path}: saved mapping must use {FLAVORS}.")
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
    from flavor_specs import classify_descriptor
    specification = "five" if len(FLAVORS) == 5 else "three"
    frame["flavor"] = text.map(lambda value: classify_descriptor(value, specification))
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

# Build one row per household-week. Counts come from all stores; prices use the exposure
# store. History is built before unusable weeks are removed.
@server_only
def prepare_data(settings):
    """Use only the merged purchase table, including its existing prices.

    The IV pools all flavors: the same-week average observed price outside the
    consumer's DMA. Repeated purchases do not duplicate store/UPC price quotes.
    """
    FLAVORS[:] = categories(getattr(settings, "flavor_spec", "three"))
    base = pl.scan_parquet(settings.sample)
    names = base.collect_schema().names()
    if "household_size" not in names:
        raise ValueError("Estimation requires household_size to restrict purchases to single-person households.")
    base = base.filter(pl.col("panel_year").is_between(settings.first_year, settings.last_year))
    if "flavor" not in names:
        products_extra = getattr(settings, "products_extra", None) or DATA_DIR / "RMS/combined/products_extra.parquet"
        flavor = read_flavors(settings.flavors, products_extra)
        base = base.with_columns(normalized_upc_expr()).join(pl.from_pandas(flavor).lazy(), on="upc", how="left")
    else:
        base = base.with_columns(normalized_upc_expr())
    price_column = next((c for c in ["price", "price_rms"] if c in names), None)
    if price_column is None:
        raise ValueError("Merged purchase table needs a price or price_rms column.")
    consumer_dma = getattr(settings, "household_dma_column", "dma_cd")
    columns = ["household_code", "household_size", "panel_year", "purchase_date", "trip_code_uc", "upc",
               "store_code_uc", "quantity", "flavor", pl.col(price_column).cast(pl.Float64).alias("price")]
    if not settings.no_price_control:
        columns.append(pl.col(consumer_dma).alias("dma_cd"))
    base = base.select(columns).with_columns(
        pl.col("purchase_date").cast(pl.String).str.slice(0, 10).str.to_date())
    base = base.filter(pl.col("price").is_finite() & (pl.col("price") > 0))
    base = base.with_columns((pl.col("purchase_date") + pl.duration(
        days=(6 - pl.col("purchase_date").dt.weekday()) % 7)).alias("week"))
    # Keep the full merged table for prices/IVs, even in a household-limited run.
    # Restrict demand histories before weekly aggregation; retain all quotes for menus/IVs.
    selected = base.filter(pl.col("household_size").cast(pl.Float64, strict=False) == 1)
    if settings.max_households:
        ids = (selected.select("household_code").unique().sort("household_code")
               .limit(settings.max_households).collect(engine="streaming")["household_code"])
        selected = selected.filter(pl.col("household_code").is_in(ids.implode()))
    print(f"Loading merged purchase table: {settings.sample}", flush=True)
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
    if not settings.no_price_control:
        lines["dma_cd"] = check_dma(lines["dma_cd"], "consumer")
        if (lines.groupby(keys)["dma_cd"].nunique() > 1).any():
            raise ValueError("Multiple consumer DMAs within a household-week.")
        weekly = weekly.merge(lines[keys + ["dma_cd"]].drop_duplicates(), on=keys, validate="one_to_one")

    quote_keys = ["store_code_uc", "week", "upc"]
    price_columns = quote_keys + ["flavor", "price"]
    if not settings.no_price_control:
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
    if not settings.no_price_control:
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
    weekly = add_stocks(weekly.sort_values(keys).reset_index(drop=True), settings.habit_weight,
                        getattr(settings, "novelty_measure", "previous_share"))
    price_cols = ["price_" + f for f in FLAVORS]
    available = np.isfinite(weekly[price_cols].to_numpy(float)) & (weekly[price_cols].to_numpy(float) > 0)
    counts = weekly[FLAVORS].to_numpy(float)
    complete = available.any(axis=1) & ((counts == 0) | available).all(axis=1)
    for j, f in enumerate(FLAVORS):
        weekly["available_" + f] = available[:, j]
    print(f"Weeks excluded for purchased flavors without a covered price: {(~complete).sum():,}", flush=True)
    if not settings.no_price_control:
        iv_cols = ["iv_" + f for f in FLAVORS]
        iv_complete = np.isfinite(weekly[iv_cols]).all(axis=1)
        print(f"Additional weeks excluded for missing other-DMA IVs: {(complete & ~iv_complete).sum():,}", flush=True)
        complete &= iv_complete
    weekly = weekly.loc[complete].copy()
    repeated = weekly.groupby("household_code")["week"].transform("nunique") >= 2
    print(f"Households excluded for fewer than two usable weeks: {weekly.loc[~repeated, 'household_code'].nunique():,}", flush=True)
    weekly = weekly.loc[repeated].reset_index(drop=True)
    weekly["outside"] = np.maximum(settings.capacity - weekly[FLAVORS].sum(axis=1), 0)
    if weekly.empty:
        raise ValueError("No households with at least two usable purchase weeks.")
    return weekly

# Record history BEFORE this week's purchases. Then add the current basket to the stock
# for next week.
def switch_away_event(before, after):
    """Binary recovery event for each flavor; inputs are consecutive basket shares.

    A strict majority is dominance. A new entry only rewards flavors whose
    shares fall. Multiple triggers still produce a single unit boost.
    """
    entered = (before == 0) & (after > 0)
    takeover = ((before > .5)[:, None] & (after > .5)[None, :]
                & ~np.eye(len(before), dtype=bool)).any(axis=1)
    return (takeover | (entered.any() & (after < before))).astype(float)


def add_stocks(weekly, habit_weight=0.7, novelty_measure="previous_share"):
    """Record cumulative stock and novelty before current purchases.

    previous_share: legacy N = 1 - own share in the preceding calendar week.
    lagged_switch_away: N in t is the binary switch-away event from t-2 to t-1.
    Both transitions must be adjacent calendar weeks; missing history gives 0.
    Stock remains the sum of all preceding observed own-flavor shares.
    """
    if novelty_measure not in {"previous_share", "lagged_switch_away"}:
        raise ValueError("Unknown novelty measure: " + str(novelty_measure))
    result = (
        weekly.sort_values(["household_code", "week"]).reset_index(drop=True).copy()
    )
    stocks = np.zeros((len(result), len(FLAVORS)))
    novelty = np.zeros_like(stocks)
    for _, group in result.groupby("household_code", sort=False):
        stock = np.zeros(len(FLAVORS))
        previous_share = np.zeros(len(FLAVORS))
        previous_week = None
        previous_event = np.zeros(len(FLAVORS))
        for index, row in group.iterrows():
            stocks[index] = stock
            adjacent = previous_week is not None and (row["week"] - previous_week).days == 7
            if adjacent:
                novelty[index] = (previous_share.sum() - previous_share
                                  if novelty_measure == "previous_share" else previous_event)
            counts = row[FLAVORS].to_numpy(float)
            current_share = counts / counts.sum()
            previous_event = (switch_away_event(previous_share, current_share)
                              if adjacent else np.zeros(len(FLAVORS)))
            previous_share = current_share
            stock = stock + previous_share
            previous_week = row["week"]
    for j, name in enumerate(FLAVORS):
        result["stock_" + name] = stocks[:, j]
        result["novelty_" + name] = novelty[:, j]
        result["xi_" + name] = (1 - habit_weight) * novelty[
            :, j
        ] - habit_weight * stocks[:, j]
    return result

# Turn the named table columns into numeric arrays. Rows are household-weeks; inside
# columns are other, berry, plain.
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
    if ((counts[:, :len(FLAVORS)] > 0) & ~available).any() or not available.any(axis=1).all():
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



# === 3. Persistent types, utilities, likelihood, and analytic gradient === #



# Each row is one possible persistent household type. Use the same rows at every optimizer
# step.
def make_draws(number, seed, model=None):
    """Fixed antithetic normal draws, reused throughout optimization."""
    if number < 2 or number % 2:
        raise ValueError("Use an even number of draws, at least two.")
    z = np.random.default_rng(seed).standard_normal((number // 2, 3 if model == "heterogeneous_history" else 2))
    return np.vstack([z, -z])

# This list fixes the order of coefficients in theta. Never change theta's order
# independently of this function.
def parameter_names(model, outside, control=False):
    names = (["inside_constant"] if outside else []) + ["beta_" + f for f in FLAVORS[1:]]
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

# Two ordered gamma distributions share the same price-coefficient distribution. Ordering
# means prevents component-label swapping.
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

# Combine the likelihoods of entire household histories from the two gamma populations.
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

# The main model multiplies Xi by gamma. Optional separate-history models estimate novelty
# and stock effects separately.
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

# For one possible type: sum log probabilities over the whole household history. Only then
# average across possible types.
def household_loglike(theta, data, draws, model):
    """L_i = mean_r exp(sum_t sum_j count_itj * log P_itj,r).

    Gamma is normal and common across flavors; alpha = -exp(mu + sd*z). Beta never varies by draw.
    Process one draw at a time to avoid a weeks x draws x products array.
    The multinomial combinatorial constant is omitted (independent of theta).
    """
    if model == "mixture":
        return mixture_loglike_score(theta, data, draws)
    p = dict(zip(parameter_names(model, data["outside"], data["control"]), theta))
    beta = np.array([0.] + [p["beta_" + f] for f in FLAVORS[1:]])
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

# Compute the same likelihood AND its slope with respect to every parameter. The optimizer
# uses these slopes to choose its next step.
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
        utility = (np.array([0.] + [p["beta_" + f] for f in FLAVORS[1:]]) + p.get("inside_constant", 0.)
                   + alpha * data["prices"] + history_utility(p, data, z, model) + p.get("price_control", 0.) * data["cf"])
        utility = np.where(data["available"], utility, -np.inf)
        if data["outside"]:
            utility = np.column_stack([utility, np.zeros(len(utility))])
        logprob = utility - logsumexp(utility, axis=1, keepdims=True)
        row_ll = (data["counts"] * np.where(data["counts"] > 0, logprob, 0.)).sum(axis=1)
        likelihoods.append(np.bincount(data["hh"], weights=row_ll, minlength=data["n_households"]))
        residual = data["counts"][:, :len(FLAVORS)] - total[:, None] * np.exp(logprob[:, :len(FLAVORS)])
        price_score = (residual * data["prices"]).sum(axis=1) * alpha
        columns = {"inside_constant": residual.sum(axis=1), "beta_berry": residual[:, 1],
                   "beta_plain": residual[:, 2], "mu_log_minus_alpha": price_score,
                   "sd_log_minus_alpha": price_score * z[0],
                   "price_control": (residual * data["cf"]).sum(axis=1)}
        columns.update({"beta_" + f: residual[:, j] for j, f in enumerate(FLAVORS) if j})
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
    # weights is draws-by-households; scores is draws-by-households-by-parameters.
    # Weight each draw's score, then sum over the draw axis.
    weighted_scores = weights[:, :, None] * np.asarray(scores)
    household_scores = weighted_scores.sum(axis=0)
    if household:
        return normalizer - np.log(len(active)), household_scores
    gradient = -household_scores.mean(axis=0)
    return -(normalizer - np.log(len(active))).mean(), gradient



# === 4. Find the parameter estimates === #



# Try each starting vector, then retain the smallest negative log likelihood. Check
# convergence before interpreting the result.
def optimize_demand(data, settings, initial=None, number_starts=None, quiet=False):
    draws = make_draws(settings.draws, settings.seed, settings.model)
    names = parameter_names(settings.model, data["outside"], data["control"])
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
    if settings.model == "mixture":
        start[names.index("mean_gamma")] = -0.2
    if initial is not None:
        start = np.asarray(initial).copy()
    rng = np.random.default_rng(settings.seed + 1)
    results = []

    def objective(theta):
        return demand_objective_gradient(theta, data, draws, settings.model)

    for s in range(settings.starts if number_starts is None else number_starts):
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
            options={"maxiter": settings.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        results.append(fit)
        if not quiet:
            settings.output.mkdir(parents=True, exist_ok=True)
            (settings.output / "starts_checkpoint.json").write_text(json.dumps([
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



# === 5. Standard errors: bootstrap or Hessian === #



# A bootstrap draw selects whole households, not isolated weeks. Selecting the same
# household twice creates two independent copied IDs.
def resample_households(frame, rng):
    """Keep whole histories; repeated households get distinct bootstrap IDs.

    Stocks/Xi stay attached to the original histories. Rebuilding stocks after
    dropping incomplete price weeks would incorrectly erase observed purchases.
    """
    groups = [group for _, group in frame.groupby("household_code", sort=True)]
    selected = rng.integers(0, len(groups), size=len(groups))
    samples = []
    for new_id, old_id in enumerate(selected):
        group = groups[old_id].copy()
        group["household_code"] = new_id
        samples.append(group)
    return pd.concat(samples, ignore_index=True)


# Resample households, refit price controls, and refit demand. The spread across
# successful fits gives the standard errors.
@server_only
def bootstrap_standard_errors(frame, settings, estimate, initial, names):
    """Refit price controls AND demand in every household-cluster replication.

    estimate(frame, initial) returns a scipy optimizer result. All replications
    use the same integration draws as the main fit. Retail instruments stay
    fixed: this is inference conditional on the observed retail price universe.
    """
    rng = np.random.default_rng(settings.seed + 10000)
    records = []
    values = []
    for replication in range(settings.bootstrap_reps):
        record = {"replication": replication + 1, "converged": False}
        try:
            sample = resample_households(frame, rng)
            if not settings.no_price_control:
                sample, _ = add_price_controls(sample)
            fit = estimate(sample, initial)
            record.update(converged=bool(fit.success), objective=float(fit.fun),
                          message=str(fit.message))
            record.update(dict(zip(names, fit.x)))
            if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun):
                values.append(fit.x)
        except (ValueError, np.linalg.LinAlgError) as error:
            # An unidentifiable resample is recorded, never silently accepted.
            record["message"] = str(error)
        records.append(record)
        # Checkpoint after each replication in case a long cluster job ends.
        pd.DataFrame(records).to_csv(settings.output / "bootstrap_draws.csv", index=False)
        print(f"Bootstrap {replication+1}/{settings.bootstrap_reps}: "
              f"{len(values)} usable fits", flush=True)
    valid = len(values)
    report = dict(method=("household bootstrap" if settings.no_price_control
                          else "household bootstrap, refitting both stages"),
                  requested=settings.bootstrap_reps, usable=valid,
                  conditional_on="retail instruments and fixed integration draws")
    # A large failure fraction can select the bootstrap distribution.
    if valid < max(20, int(np.ceil(.8 * settings.bootstrap_reps))):
        report["status"] = "insufficient converged replications; no SEs reported"
        return None, report
    estimates = np.asarray(values)
    table = pd.DataFrame({"parameter": names,
                          "std_error": estimates.std(axis=0, ddof=1),
                          "ci_lower": np.quantile(estimates, .025, axis=0),
                          "ci_upper": np.quantile(estimates, .975, axis=0)})
    covariance = pd.DataFrame(np.cov(estimates, rowvar=False), index=names, columns=names)
    covariance.to_csv(settings.output / "covariance.csv")
    report["status"] = "computed; percentile 95% intervals"
    return table, report

# Differentiate the likelihood gradient numerically, check the Hessian, then invert the
# information matrix.
def observed_covariance(theta, gradient, households, bounds):
    """Differentiate the mean negative-log-likelihood gradient, then scale by I."""
    theta = np.asarray(theta, dtype=float)
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(theta))
    lower, upper = np.asarray(bounds, dtype=float).T
    if np.any(np.minimum(theta - lower, upper - theta) <= np.maximum(steps, 1e-4)):
        raise ValueError("parameter at or near bounds; normal inference withheld")
    hessian = np.empty((len(theta), len(theta)))
    for j, step in enumerate(steps):
        delta = np.zeros(len(theta))
        delta[j] = step
        hessian[:, j] = (gradient(theta + delta) - gradient(theta - delta)) / (2 * step)
    hessian = (hessian + hessian.T) / 2
    if not np.isfinite(hessian).all():
        raise ValueError("nonfinite observed information; SEs withheld")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= max(0., eigenvalues[-1] * 1e-10):
        raise ValueError("observed information not positive definite or ill-conditioned; SEs withheld")
    return np.linalg.solve(hessian, np.eye(len(theta))) / households


# Hessian uncertainty holds the fitted price controls fixed. It does not include
# uncertainty from estimating those controls.
def hessian_standard_errors(theta, gradient, households, bounds, names, output):
    report = dict(method="observed-information Hessian",
                  conditional_on="estimated price controls, retail instruments, and fixed integration draws",
                  first_stage_uncertainty="not included", robust=False)
    try:
        covariance = observed_covariance(theta, gradient, households, bounds)
    except ValueError as error:
        report["status"] = str(error)
        return None, report
    pd.DataFrame(covariance, index=names, columns=names).to_csv(output / "covariance.csv")
    se = np.sqrt(np.diag(covariance))
    critical = norm.ppf(.975)
    table = pd.DataFrame(dict(parameter=names, std_error=se,
                              ci_lower=theta - critical * se, ci_upper=theta + critical * se))
    report["status"] = "computed; normal 95% intervals; first-stage uncertainty omitted"
    return table, report



# === 6. Economic interpretation and distribution plots === #



# Convert utility coefficients to population willingness-to-pay moments in the units of
# price. These SDs are heterogeneity, not sampling SEs.
def implied_wtp(estimates, habit_weight=.7):
    """Return population moments, not sampling standard errors, in price units."""
    p = dict(zip(estimates.parameter, estimates.estimate))
    mu, sigma = p['mu_log_minus_alpha'], p.get('sd_log_minus_alpha', 0.)
    inverse_mean = np.exp(-mu + sigma**2 / 2)
    inverse_second = np.exp(-2 * mu + 2 * sigma**2)
    rows = []

    def add(term, mean, second, interpretation):
        wtp_mean = mean * inverse_mean
        variance = max(0., second * inverse_second - wtp_mean**2)
        rows.append(dict(term=term, mean_wtp=wtp_mean, population_sd=np.sqrt(variance),
                         interpretation=interpretation))

    for name in ['inside_constant'] + ['beta_' + f for f in FLAVORS[1:]]:
        if name in p:
            interpretation = ('Other flavor vs outside, at zero history and control residual'
                              if name == 'inside_constant' else 'Flavor premium relative to other')
            add(name, p[name], p[name]**2, interpretation)
    if 'mean_gamma' in p:
        mean, sd = p['mean_gamma'], p.get('sd_gamma', 0.)
        second = mean**2 + sd**2
        if 'mixture_weight' in p:
            weight = p['mixture_weight']
            mean2 = mean + p['gamma_mean_gap']
            second2 = mean2**2 + p['sd_gamma_2']**2
            add('gamma_component_1', mean, second, 'Per unit Xi, conditional on component 1')
            add('gamma_component_2', mean2, second2, 'Per unit Xi, conditional on component 2')
            mean, second = weight * mean + (1 - weight) * mean2, weight * second + (1 - weight) * second2
        add('gamma', mean, second, 'Per unit increase in history index Xi')
        add('gamma_novelty', (1 - habit_weight) * mean, (1 - habit_weight)**2 * second,
            'Per unit novelty N, holding own-flavor stock fixed')
        add('gamma_own_stock', -habit_weight * mean, habit_weight**2 * second,
            'Per unit own-flavor stock S, holding novelty fixed')
    for term in ["novelty", "own_stock"]:
        if "mean_" + term in p:
            mean, sd = p["mean_" + term], p.get("sd_" + term, 0.)
            add(term, mean, mean**2 + sd**2,
                "Per unit " + term + ", holding the other history measure fixed")
    return pd.DataFrame(rows)

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

def export_history_distributions(table, output, converged=True):
    p = table.set_index('parameter').estimate
    if 'mean_novelty' not in p:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provisional = not converged or table.get('at_bound', pd.Series(False)).astype(str).str.lower().eq('true').any()
    summaries, curves = [], []
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, term, label in zip(axes, ['novelty', 'own_stock'], ['Switching / novelty', 'Own-stock habit']):
        mean, sd = float(p['mean_'+term]), float(p.get('sd_'+term, 0.))
        if not np.isfinite([mean, sd]).all() or sd < 0:
            raise ValueError('Invalid distribution parameters')
        normal = NormalDist(mean, sd) if sd > 0 else None
        row = dict(coefficient=term, mean=mean, sd=sd, point_mass=sd == 0,
                   share_below_zero=normal.cdf(0) if normal else float(mean < 0), provisional=bool(provisional))
        for q in [.01, .05, .25, .5, .75, .95, .99]:
            row[f'q{int(100*q):02d}'] = normal.inv_cdf(q) if normal else mean
        summaries.append(row)
        if normal:
            x = np.linspace(mean-4*sd, mean+4*sd, 801)
            density = np.exp(-.5*((x-mean)/sd)**2)/(sd*np.sqrt(2*np.pi))
            ax.plot(x, density)
            curves.extend(dict(coefficient=term, value=float(v), density=float(d), cdf=normal.cdf(float(v)), point_mass=False)
                          for v, d in zip(x, density))
        else:
            ax.axvline(mean, label='Point mass')
            curves.append(dict(coefficient=term, value=mean, density=np.nan, cdf=1., point_mass=True))
        ax.axvline(0, color='gray', ls='--')
        ax.set(title=f'{label}\nMean={mean:.4g}; SD={sd:.4g}', xlabel='Utility coefficient', ylabel='Population density')
    pd.DataFrame(summaries).to_csv(output/'history_distribution_summary.csv', index=False)
    pd.DataFrame(curves).to_csv(output/'history_distribution_grid.csv', index=False)
    fig.suptitle('Fitted population history distributions'+(' (provisional)' if provisional else ''))
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(output/f'history_distributions.{extension}')
    plt.close(fig)


def plot_history_distributions(table, output, converged=True):
    export_history_distributions(table, output, converged)


# === 7. Save estimates, uncertainty, and a plain-text run record === #



def write_run_notes(output, information):
    """Write the same run information in plain text; you never edit a JSON file."""
    output.mkdir(parents=True, exist_ok=True)
    text = "RUN INFORMATION\n\n"
    text += "This is an output record, not a settings file.\n"
    text += "Change the Python parameters at the top of the script to change the run.\n\n"
    text += pformat(information, sort_dicts=False, width=90)
    (output / "run_notes.txt").write_text(text + "\n")


# Save point estimates first, then run the requested inference and write final tables,
# plots, and run notes.
@server_only
def fit_model(data, settings, frame=None):
    """Estimate demand, compute requested inference, and save readable outputs."""
    best, results, bounds = optimize_demand(data, settings)
    names = parameter_names(settings.model, data["outside"], data["control"])
    table = pd.DataFrame({"parameter": names, "estimate": best.x})
    table["at_bound"] = [
        min(abs(v - lo), abs(v - hi)) < 1e-4 for v, (lo, hi) in zip(best.x, bounds)
    ]
    settings.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.output / "point_estimates.csv", index=False)
    plot_gamma_distribution(table, settings.output, converged=best.success)
    plot_history_distributions(table, settings.output, converged=best.success)
    wtp = implied_wtp(table, getattr(settings, "habit_weight", .7))
    wtp["provisional_fit"] = not best.success or bool(table["at_bound"].any())
    wtp.to_csv(settings.output / "implied_wtp.csv", index=False)
    inference = {"method": settings.se_method, "status": "disabled"}
    if settings.se_method == "bootstrap":
        if not best.success:
            inference["status"] = "main fit not converged; bootstrap not run"
        elif frame is None:
            raise ValueError("Bootstrap requires the weekly input frame.")
        else:

            def estimate(sample, initial):
                bootstrap_data = make_arrays(sample, settings.capacity)
                fitted, _, _ = optimize_demand(
                    bootstrap_data, settings, initial, settings.bootstrap_starts, quiet=True
                )
                return fitted

            standard_errors, inference = bootstrap_standard_errors(
                frame, settings, estimate, best.x, names
            )
            if standard_errors is not None:
                table = table.merge(
                    standard_errors, on="parameter", validate="one_to_one"
                )
    elif settings.se_method == "hessian":
        if not best.success:
            inference["status"] = "main fit not converged; Hessian inference not run"
        else:
            draws = make_draws(settings.draws, settings.seed, settings.model)
            gradient = lambda theta: demand_objective_gradient(theta, data, draws, settings.model)[1]
            standard_errors, inference = hessian_standard_errors(
                best.x, gradient, data["n_households"], bounds, names, settings.output)
            if standard_errors is not None:
                table = table.merge(standard_errors, on="parameter", validate="one_to_one")
    # An explicit missing column distinguishes skipped inference from zero SEs.
    for column in ["std_error", "ci_lower", "ci_upper"]:
        if column not in table:
            table[column] = np.nan
    table.to_csv(settings.output / "estimates.csv", index=False)
    summary = dict(
        model=settings.model,
        price_control=data["control"],
        flavor_spec=getattr(settings, "flavor_spec", "three"),
        flavor_categories=list(FLAVORS),
        novelty_measure=getattr(settings, "novelty_measure", "previous_share"),
        history_specification="separate" if settings.model in ("separate_history", "heterogeneous_history") else "combined_or_none",
        sample_restriction="household_size == 1 on each purchase record",
        households=data["n_households"],
        weeks=len(data["hh"]),
        capacity=settings.capacity,
        inference=inference,
        converged=bool(best.success),
        message=str(best.message),
        nll=float(best.fun * data["n_households"]),
        parameters=len(names),
        aic=float(2 * best.fun * data["n_households"] + 2 * len(names)),
        bic=float(2 * best.fun * data["n_households"] + np.log(data["n_households"]) * len(names)),
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(settings).items()
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
    (settings.output / "fit.json").write_text(json.dumps(summary, indent=2))
    write_run_notes(settings.output, summary)
    print("Inference: " + inference["status"], flush=True)
    print(table.to_string(index=False))
    if not best.success or table["at_bound"].any():
        print("Provisional fit: inspect convergence/bounds before interpretation.")
    return best



# === 8. Optional command-line overrides === #



def read_command_line(settings):
    """Optional server flags. Normal use: edit the parameters at the top."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor-spec", choices=["three", "five"], default="three")
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
        default=model,
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
        default=no_price_control,
        help="Disable the price control using the overall other-DMA mean-price IV",
    )
    parser.add_argument("--household-dma-column", default="dma_cd")
    parser.add_argument(
        "--se-method", choices=["hessian", "bootstrap", "none"], default="hessian"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=49)
    parser.add_argument("--bootstrap-starts", type=int, default=2)
    parser.add_argument("--novelty-measure", choices=["previous_share", "lagged_switch_away"],
                        default="previous_share", help="Legacy share or lagged binary recovery event")
    parser.add_argument("--habit-weight", type=float, default=0.7)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--seed", type=int, default=306)
    parser.set_defaults(**vars(settings))
    return parser.parse_args()




# === 9. Run the estimation, step by step === #



@server_only
def run_estimation(settings):
    """The actual research steps, in the order they run."""
    if not 0 <= settings.habit_weight <= 1 or settings.capacity < 0 or settings.starts < 1:
        raise ValueError("Check habit weight, capacity, and starts.")
    if settings.first_year > settings.last_year or (
        settings.max_households is not None and settings.max_households < 1
    ):
        raise ValueError("Check years and household limit.")
    if settings.draws < 2 or settings.draws % 2 or settings.maxiter < 1:
        raise ValueError("Use an even draw count >= 2 and maxiter >= 1.")
    if settings.se_method == "bootstrap" and (
        settings.bootstrap_reps < 20 or settings.bootstrap_starts < 1
    ):
        raise ValueError("Bootstrap requires at least 20 replications and one start.")
    # Step 1: read purchases and build weekly counts, prices, and histories.
    frame = prepare_data(settings)

    # Step 2: estimate price equations and attach their residuals.
    if not settings.no_price_control:
        frame, first_stage = add_price_controls(frame)
        settings.output.mkdir(parents=True, exist_ok=True)
        first_stage.to_csv(settings.output / "first_stage.csv", index=False)
        print(first_stage.to_string(index=False), flush=True)
    settings.output.mkdir(parents=True, exist_ok=True)
    # Step 3: save the exact table that will enter the likelihood.
    frame.to_csv(settings.output / "weekly_inputs.csv", index=False)

    # Step 4: form numeric arrays, estimate demand, compute inference, save results.
    data = make_arrays(frame, settings.capacity)
    fit_model(data, settings, frame)

if __name__ == "__main__":
    # Collect the ordinary Python variables above into one settings record.
    # settings.draws means exactly the variable draws from the Parameters section.
    settings = SimpleNamespace(
        sample=sample,
        flavors=flavors,
        products_extra=products_extra,
        output=output,
        model=model,
        first_year=first_year,
        last_year=last_year,
        max_households=max_households,
        capacity=capacity,
        no_price_control=no_price_control,
        household_dma_column=household_dma_column,
        se_method=se_method,
        bootstrap_reps=bootstrap_reps,
        bootstrap_starts=bootstrap_starts,
        habit_weight=habit_weight,
        novelty_measure="previous_share",
        draws=draws,
        starts=starts,
        maxiter=maxiter,
        seed=seed,
    )
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_estimation(settings)
