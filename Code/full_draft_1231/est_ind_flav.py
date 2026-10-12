import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp
import seaborn as sns
import statsmodels.formula.api as smf

install()
console = Console()

MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Output/"
PLOT_OUTPUT_DIR = "../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

TOP_K_FLAVORS = 8  # Top K individual non-plain flavors
CALIBRATED_LAMBDA = 0.7  # Calibrated memory decay parameter
N_DRAWS = 50


def compute_satiation_state_unbinned(
    weekly_purchases,
    n_inside,
    lambda_mem=CALIBRATED_LAMBDA,
    delta_discount=0.9,
):
    sorted_df = weekly_purchases.sort_values(["household_code", "week_end"])
    inventory_records = []

    for hh_id, hh_group in sorted_df.groupby("household_code"):
        weeks = hh_group["week_end"].unique()
        C_history = {}
        prev_X = np.zeros(n_inside)

        for t_idx, week in enumerate(weeks):
            C_current = np.zeros(n_inside)
            memory_term = np.zeros(n_inside)

            if t_idx > 0:
                for s in range(t_idx):
                    discount = delta_discount ** (t_idx - s)
                    memory_term += discount * C_history[s]

            for j in range(n_inside):
                cross_term = np.sum(prev_X) - prev_X[j]
                C_current[j] = (lambda_mem * memory_term[j]) + (
                    (1.0 - lambda_mem) * cross_term
                )

            record = {"household_code": hh_id, "week_end": week}
            for j in range(n_inside):
                record[f"C_sat_{j}"] = C_current[j]
            inventory_records.append(record)

            C_history[t_idx] = C_current.copy()

            sub = hh_group[hh_group["week_end"] == week]
            curr_X = np.zeros(n_inside)
            for row in sub.itertuples():
                if row.choice_idx < n_inside:
                    curr_X[row.choice_idx] = row.units_bought
            total_inside_units = curr_X.sum()
            if total_inside_units > 0:
                curr_X = curr_X / total_inside_units
            prev_X = curr_X

    return pd.DataFrame(inventory_records)


def load_and_preprocess_unbinned(weekly_capacity=14, top_k=TOP_K_FLAVORS):
    schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

    exprs = [
        pl.col("quantity").cast(pl.Int64),
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

    merged_df["flavor_str"] = (
        merged_df["flavor"].fillna("").astype(str).str.lower().str.strip()
    )

    plain_regex = r"plain|unflavored"
    is_plain = merged_df["flavor_str"].str.contains(plain_regex, na=False)

    non_plain_df = merged_df[~is_plain]
    top_flavors = non_plain_df["flavor_str"].value_counts().head(top_k).index.tolist()

    # Categories: Plain (Index 0 - EXCLUDED BASE), top_flavors, and outside option
    categories = ["plain"] + top_flavors + ["outside"]
    cat_map = {c: i for i, c in enumerate(categories)}
    n_inside = len(categories) - 1

    console.print(
        f"\n[bold yellow]UNBINNED FLAVOR CATEGORIES (BASE = Plain):[/bold yellow] {categories}\n"
    )

    def assign_category(row):
        f_str = row["flavor_str"]
        if pd.isna(f_str) or f_str == "":
            return "outside"
        if "plain" in f_str or "unflavored" in f_str:
            return "plain"
        for flv in top_flavors:
            if flv in f_str:
                return flv
        return "outside"

    merged_df["category"] = merged_df.apply(assign_category, axis=1)

    # Price IV Construction
    market_price = (
        merged_df.groupby(["upc", "week_end", "market_name"])["price"]
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

    merged_df = merged_df.merge(
        market_price[["upc", "week_end", "market_name", "price_iv"]],
        on=["upc", "week_end", "market_name"],
        how="left",
    )

    merged_df = merged_df[(merged_df["price"] > 0.1) | (merged_df["price"].isna())]
    iv_res = smf.ols(
        "price ~ price_iv + brand_cd + C(week_end)", data=merged_df, missing="drop"
    ).fit()

    merged_df["iv_res"] = np.nan
    merged_df.loc[iv_res.model.data.row_labels, "iv_res"] = iv_res.resid
    valid_resids = merged_df["iv_res"].dropna()
    if len(valid_resids) > 0:
        res_mean = valid_resids.mean()
        res_std = valid_resids.std()
        merged_df["iv_res"] = (merged_df["iv_res"] - res_mean) / (
            res_std if res_std > 0 else 1.0
        )
    merged_df["iv_res"] = merged_df["iv_res"].fillna(0.0)

    inside_df = merged_df[merged_df["category"] != "outside"].copy()

    weekly_purchases = (
        inside_df.groupby(["household_code", "store_code_uc", "week_end", "category"])
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["category"].map(cat_map)

    cat_choice_sets = (
        inside_df.groupby(["store_code_uc", "week_end", "category"])
        .agg(price=("price", "mean"), iv_res=("iv_res", "mean"))
        .reset_index()
    )

    cat_choice_sets["price"] = cat_choice_sets.groupby(["store_code_uc", "category"])[
        "price"
    ].transform(lambda x: x.fillna(x.mean()))
    cat_choice_sets["price"] = cat_choice_sets.groupby("category")["price"].transform(
        lambda x: x.fillna(x.mean())
    )
    cat_choice_sets["iv_res"] = cat_choice_sets["iv_res"].fillna(0.0)

    overall_cat_prices = inside_df.groupby("category")["price"].mean().to_dict()

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((n_inside, 2), dtype=np.float64)

        for cat, idx in cat_map.items():
            if cat != "outside" and cat in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat]

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    console.print(
        f"[bold green]Computing unbinned satiation states with lambda = {CALIBRATED_LAMBDA}...[/bold green]"
    )
    df_sat = compute_satiation_state_unbinned(
        weekly_purchases,
        n_inside=n_inside,
        lambda_mem=CALIBRATED_LAMBDA,
        delta_discount=0.9,
    )
    for j in range(n_inside):
        c_col = f"C_sat_{j}"
        c_sd = df_sat[c_col].std()
        if c_sd > 0:
            df_sat[c_col] = df_sat[c_col] / c_sd

    hh_weeks = weekly_purchases[
        ["household_code", "store_code_uc", "week_end"]
    ].drop_duplicates()
    hh_weeks = hh_weeks.merge(
        df_sat, on=["household_code", "week_end"], how="left"
    ).fillna(0.0)

    hh_income_map = (
        merged_df.drop_duplicates(subset=["household_code"])
        .set_index("household_code")["household_income"]
        .to_dict()
    )

    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(n_inside, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < n_inside:
                c_arr[row.choice_idx] = row.units_bought
        counts_dict[(hh_id, week)] = c_arr

    all_prices, all_resids, all_choices, all_c_states, all_log_inc = (
        [],
        [],
        [],
        [],
        [],
    )

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = row_dict["week_end"]
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        raw_inc = hh_income_map.get(hh_id, 1)
        log_inc = np.log(max(float(raw_inc) if pd.notna(raw_inc) else 1.0, 1.0))

        inside_counts = counts_dict.get(
            (hh_id, week), np.zeros(n_inside, dtype=np.int64)
        )
        inside_units = np.sum(inside_counts)

        effective_capacity = max(weekly_capacity, inside_units + 1)
        outside_count = effective_capacity - inside_units

        full_counts = np.append(inside_counts, outside_count)
        c_vec = np.array(
            [row_dict[f"C_sat_{j}"] for j in range(n_inside)], dtype=np.float64
        )

        all_prices.append(choice_set_matrix[(store, week)][:n_inside, 0])
        all_resids.append(choice_set_matrix[(store, week)][:n_inside, 1])
        all_choices.append(full_counts)
        all_c_states.append(c_vec)
        all_log_inc.append(log_inc)

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "log_inc": np.array(all_log_inc, dtype=np.float64),
        "top_flavors": top_flavors,
        "n_inside": n_inside,
    }

    return vec_data


# =========================================================
# PART 1: UNBINNED NAIVE SPECIFICATIONS (NO GAMMAS)
# =========================================================
def total_objective_unbinned_naive(params, vec_data):
    n_inside = vec_data["n_inside"]
    const = params[0]
    beta_non_plain = params[1:n_inside]
    beta_vec = np.append(0.0, beta_non_plain)  # Plain beta is fixed to 0.0 (Base)

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]

    u_inside = const + beta_vec + alpha * prices + sigma * resids
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def total_objective_rc_unbinned_naive(params, vec_data, draws):
    n_inside = vec_data["n_inside"]
    n_non_plain = n_inside - 1
    n_draws = draws.shape[0]

    const = params[0]
    mu_beta_non_plain = params[1:n_inside]
    sd_beta_non_plain = params[n_inside : 2 * n_inside - 1]
    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]

    b_draws = (
        mu_beta_non_plain[None, :] + sd_beta_non_plain[None, :] * draws[:, :n_non_plain]
    )
    beta_draws_matrix = np.hstack([np.zeros((n_draws, 1)), b_draws])

    u_base = const + alpha * prices + sigma * resids
    u_inside = u_base[:, None, :] + beta_draws_matrix[None, :, :]
    u_out = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_out], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll = logsumexp(log_probs, axis=1) - np.log(n_draws)
    total_ll = np.sum(choices * obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_unbinned_naive_model(vec_data):
    n_inside = vec_data["n_inside"]
    # const (1) + beta_non_plain (n_inside - 1) + alpha (1) + sigma (1) = n_inside + 2
    x0 = np.zeros(n_inside + 2)
    x0[-2] = -0.1
    bounds = [(None, None)] * n_inside + [(None, 0.0), (None, None)]

    res = minimize(
        total_objective_unbinned_naive,
        x0=x0,
        args=(vec_data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8, "gtol": 1e-5},
    )

    eps = 1e-5
    n = len(res.x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x1, x2, x3, x4 = res.x.copy(), res.x.copy(), res.x.copy(), res.x.copy()
            x1[i] += eps
            x1[j] += eps
            x2[i] += eps
            x2[j] -= eps
            x3[i] -= eps
            x3[j] += eps
            x4[i] -= eps
            x4[j] -= eps

            f1 = total_objective_unbinned_naive(x1, vec_data)
            f2 = total_objective_unbinned_naive(x2, vec_data)
            f3 = total_objective_unbinned_naive(x3, vec_data)
            f4 = total_objective_unbinned_naive(x4, vec_data)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    try:
        se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)

    z = res.x / se
    p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def estimate_rc_unbinned_naive_model(vec_data, draws):
    n_inside = vec_data["n_inside"]
    n_non_plain = n_inside - 1

    total_params = 2 * n_non_plain + 3
    x0 = np.zeros(total_params)
    x0[n_inside : 2 * n_inside - 1] = 0.5
    x0[-2] = -0.1

    bounds = (
        [(-10.0, 10.0)] * n_inside
        + [(1e-4, 5.0)] * n_non_plain
        + [(-10.0, -0.01), (-10.0, 10.0)]
    )

    res = minimize(
        total_objective_rc_unbinned_naive,
        x0=x0,
        args=(vec_data, draws),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-7, "gtol": 1e-5, "maxiter": 1000},
    )

    eps = 1e-4
    n = len(res.x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x1, x2, x3, x4 = res.x.copy(), res.x.copy(), res.x.copy(), res.x.copy()
            x1[i] += eps
            x1[j] += eps
            x2[i] += eps
            x2[j] -= eps
            x3[i] -= eps
            x3[j] += eps
            x4[i] -= eps
            x4[j] -= eps

            f1 = total_objective_rc_unbinned_naive(x1, vec_data, draws)
            f2 = total_objective_rc_unbinned_naive(x2, vec_data, draws)
            f3 = total_objective_rc_unbinned_naive(x3, vec_data, draws)
            f4 = total_objective_rc_unbinned_naive(x4, vec_data, draws)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    try:
        se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)

    z = res.x / se
    p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


# =========================================================
# PART 2: UNBINNED FLAVOR SATIATION SPECIFICATIONS (WITH GAMMAS)
# =========================================================
def total_objective_unbinned_gamma(params, vec_data):
    n_inside = vec_data["n_inside"]
    const = params[0]
    beta_non_plain = params[1:n_inside]
    beta_vec = np.append(0.0, beta_non_plain)  # Plain beta is fixed to 0.0 (Base)

    gamma_vec = params[n_inside : 2 * n_inside]
    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]
    c_state = vec_data["c_states"]

    u_inside = (
        const
        + beta_vec
        + gamma_vec * np.log(1.0 + c_state)
        + alpha * prices
        + sigma * resids
    )
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def total_objective_rc_unbinned_gamma(params, vec_data, draws):
    n_inside = vec_data["n_inside"]
    n_non_plain = n_inside - 1
    n_draws = draws.shape[0]

    const = params[0]
    mu_beta_non_plain = params[1:n_inside]
    mu_gamma_vec = params[n_inside : 2 * n_inside]

    sd_beta_non_plain = params[2 * n_inside : 3 * n_inside - 1]
    sd_gamma_vec = params[3 * n_inside - 1 : 4 * n_inside - 1]

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]
    c_state = vec_data["c_states"]

    b_draws = (
        mu_beta_non_plain[None, :] + sd_beta_non_plain[None, :] * draws[:, :n_non_plain]
    )
    beta_draws_matrix = np.hstack([np.zeros((n_draws, 1)), b_draws])

    g_draws = (
        mu_gamma_vec[None, :]
        + sd_gamma_vec[None, :] * draws[:, n_non_plain : 2 * n_inside - 1]
    )

    u_base = const + alpha * prices + sigma * resids
    u_inside = (
        u_base[:, None, :]
        + beta_draws_matrix[None, :, :]
        + g_draws[None, :, :] * np.log(1.0 + c_state[:, None, :])
    )

    u_out = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_out], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll = logsumexp(log_probs, axis=1) - np.log(n_draws)
    total_ll = np.sum(choices * obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_unbinned_flavor_gamma_model(vec_data):
    n_inside = vec_data["n_inside"]
    total_params = 2 * n_inside + 2
    x0 = np.zeros(total_params)
    x0[-2] = -0.1
    bounds = [(None, None)] * (2 * n_inside) + [(None, 0.0), (None, None)]

    res = minimize(
        total_objective_unbinned_gamma,
        x0=x0,
        args=(vec_data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8, "gtol": 1e-5},
    )

    eps = 1e-5
    n = len(res.x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x1, x2, x3, x4 = res.x.copy(), res.x.copy(), res.x.copy(), res.x.copy()
            x1[i] += eps
            x1[j] += eps
            x2[i] += eps
            x2[j] -= eps
            x3[i] -= eps
            x3[j] += eps
            x4[i] -= eps
            x4[j] -= eps

            f1 = total_objective_unbinned_gamma(x1, vec_data)
            f2 = total_objective_unbinned_gamma(x2, vec_data)
            f3 = total_objective_unbinned_gamma(x3, vec_data)
            f4 = total_objective_unbinned_gamma(x4, vec_data)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    try:
        se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)

    z = res.x / se
    p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def estimate_rc_unbinned_flavor_gamma_model(vec_data, draws):
    n_inside = vec_data["n_inside"]

    total_params = 4 * n_inside + 1
    x0 = np.zeros(total_params)
    x0[2 * n_inside + 1 : 4 * n_inside] = 0.5
    x0[-2] = -0.1

    bounds = (
        [(-10.0, 10.0)] * n_inside  # const + mu_beta
        + [(-10.0, 0.0)] * n_inside  # mu_gamma
        + [(1e-4, 5.0)] * (2 * n_inside - 1)  # sd_beta + sd_gamma
        + [(-10.0, -0.01), (-10.0, 10.0)]  # price + cf
    )

    res = minimize(
        total_objective_rc_unbinned_gamma,
        x0=x0,
        args=(vec_data, draws),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-7, "gtol": 1e-5, "maxiter": 2000},
    )

    eps = 1e-4
    n = len(res.x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x1, x2, x3, x4 = res.x.copy(), res.x.copy(), res.x.copy(), res.x.copy()
            x1[i] += eps
            x1[j] += eps
            x2[i] += eps
            x2[j] -= eps
            x3[i] -= eps
            x3[j] += eps
            x4[i] -= eps
            x4[j] -= eps

            f1 = total_objective_rc_unbinned_gamma(x1, vec_data, draws)
            f2 = total_objective_rc_unbinned_gamma(x2, vec_data, draws)
            f3 = total_objective_rc_unbinned_gamma(x3, vec_data, draws)
            f4 = total_objective_rc_unbinned_gamma(x4, vec_data, draws)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    try:
        cov = np.linalg.inv(hessian + np.eye(n) * 1e-6)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)

    se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    z = res.x / np.where(se > 0, se, np.nan)
    p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def display_results_unbinned(
    results,
    title="UNBINNED SPECIFICATION RESULTS",
    vec_data=None,
    model_type="flavor_gamma",
):
    top_flavors = vec_data["top_flavors"]
    n_inside = vec_data["n_inside"]

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    alpha_val = results["params"][-2]

    # Explicit parameter labels relative to Plain
    if model_type == "naive":
        param_names = ["Constant"]
        param_names += [f"beta_{flv} (vs Plain Base)" for flv in top_flavors]
        param_names += ["Price", "Control Func."]
    elif model_type == "rc_naive":
        param_names = ["Constant"]
        param_names += [f"Mean beta_{flv} (vs Plain Base)" for flv in top_flavors]
        param_names += [f"SD beta_{flv}" for flv in top_flavors]
        param_names += ["Price", "Control Func."]
    elif model_type == "flavor_gamma":
        param_names = ["Constant"]
        param_names += [f"beta_{flv} (vs Plain Base)" for flv in top_flavors]
        param_names += ["gamma_plain"]
        param_names += [f"gamma_{flv}" for flv in top_flavors]
        param_names += ["Price", "Control Func."]
    elif model_type == "rc_flavor_gamma":
        param_names = ["Constant"]
        param_names += [f"Mean beta_{flv} (vs Plain Base)" for flv in top_flavors]
        param_names += ["Mean gamma_plain"]
        param_names += [f"Mean gamma_{flv}" for flv in top_flavors]
        param_names += [f"SD beta_{flv}" for flv in top_flavors]
        param_names += ["SD gamma_plain"]
        param_names += [f"SD gamma_{flv}" for flv in top_flavors]
        param_names += ["Price", "Control Func."]

    for name, val, se, z, p in zip(
        param_names,
        results["params"],
        results["se"],
        results["z_stat"],
        results["p_val"],
    ):
        p_str = f"{p:.4f}" if not np.isnan(p) else "NA"
        if not np.isnan(p):
            if p < 0.001:
                p_str += " ***"
            elif p < 0.05:
                p_str += " **"

        table.add_row(
            name,
            f"{val:.4f}",
            f"{se:.4f}" if not np.isnan(se) else "NA",
            f"{z:.3f}" if not np.isnan(z) else "NA",
            p_str,
        )

    console.print(table)
    console.print(f"[bold]Optimization Success:[/bold] {results['success']}")
    console.print(f"[bold]Final LL Objective:[/bold] {results['fun']:.4f}")

    if abs(alpha_val) > 1e-4:
        wtp_table = Table(
            title=f"{title} - WILLINGNESS TO PAY (WTP in $ vs Plain Base)",
            show_header=True,
            header_style="bold yellow",
        )
        wtp_table.add_column("Attribute / Parameter", style="cyan", justify="left")
        wtp_table.add_column("WTP ($)", justify="right")

        if "naive" in model_type:
            for idx, flv in enumerate(top_flavors):
                b_val = results["params"][1 + idx]
                wtp_table.add_row(
                    f"WTP Preference: {flv} (vs Plain Base)",
                    f"${b_val / alpha_val:.2f}",
                )
        elif "gamma" in model_type:
            for idx, flv in enumerate(top_flavors):
                b_val = results["params"][1 + idx]
                wtp_table.add_row(
                    f"WTP Preference: {flv} (vs Plain Base)",
                    f"${b_val / alpha_val:.2f}",
                )
            g_plain = results["params"][n_inside]
            wtp_table.add_row(
                "WTP Satiation Disutility: Plain", f"${g_plain / alpha_val:.2f}"
            )
            for idx, flv in enumerate(top_flavors):
                g_val = results["params"][n_inside + 1 + idx]
                wtp_table.add_row(
                    f"WTP Satiation Disutility: {flv}", f"${g_val / alpha_val:.2f}"
                )

        console.print(wtp_table)
    console.print("\n")


def generate_type_plots_unbinned(res_rc_flavor_gamma, vec_data, n_draws=5000):
    console.print(
        "[bold green]Generating Unbinned Flavor Heterogeneity Plots...[/bold green]"
    )
    top_flavors = vec_data["top_flavors"]
    n_inside = vec_data["n_inside"]

    params = res_rc_flavor_gamma["params"]
    alpha_val = abs(params[-2])

    mu_beta_non_plain = params[1:n_inside]
    mu_gamma_vec = params[n_inside : 2 * n_inside]
    sd_beta_non_plain = params[2 * n_inside : 3 * n_inside - 1]
    sd_gamma_vec = params[3 * n_inside - 1 : 4 * n_inside - 1]

    plot_rng = np.random.default_rng(2026)

    flv1_name, flv2_name = top_flavors[0], top_flavors[1]

    b_flv1 = plot_rng.normal(mu_beta_non_plain[0], sd_beta_non_plain[0], n_draws)
    b_flv2 = plot_rng.normal(mu_beta_non_plain[1], sd_beta_non_plain[1], n_draws)

    g_plain = plot_rng.normal(mu_gamma_vec[0], sd_gamma_vec[0], n_draws)
    g_flv1 = plot_rng.normal(mu_gamma_vec[1], sd_gamma_vec[1], n_draws)

    draws_df = pd.DataFrame(
        {
            f"beta_{flv1_name}": b_flv1,
            f"beta_{flv2_name}": b_flv2,
            "gamma_plain": g_plain,
            f"gamma_{flv1_name}": g_flv1,
            f"wtp_beta_{flv1_name}": b_flv1 / alpha_val,
            f"wtp_beta_{flv2_name}": b_flv2 / alpha_val,
            "wtp_gamma_plain": g_plain / alpha_val,
            f"wtp_gamma_{flv1_name}": g_flv1 / alpha_val,
        }
    )

    sns.set_theme(style="whitegrid")

    # Figure 4: Marginal Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(
        draws_df[f"beta_{flv1_name}"],
        ax=axes[0],
        color="#d73027",
        fill=True,
        alpha=0.4,
        label=f"{flv1_name.title()} Preference (vs Plain Base)",
    )
    sns.kdeplot(
        draws_df[f"beta_{flv2_name}"],
        ax=axes[0],
        color="#4575b4",
        fill=True,
        alpha=0.4,
        label=f"{flv2_name.title()} Preference (vs Plain Base)",
    )
    axes[0].set_title("Unbinned Flavor Preferences (vs Plain Base)", fontweight="bold")
    axes[0].set_xlabel("Preference Parameter (Utility)", fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white")

    sns.kdeplot(
        draws_df[f"gamma_{flv1_name}"],
        ax=axes[1],
        color="#d73027",
        fill=True,
        alpha=0.4,
        label=f"{flv1_name.title()} Satiation",
    )
    sns.kdeplot(
        draws_df["gamma_plain"],
        ax=axes[1],
        color="#4575b4",
        fill=True,
        alpha=0.4,
        label="Plain Satiation",
    )
    axes[1].set_title("Unbinned Satiation Parameters", fontweight="bold")
    axes[1].set_xlabel("Satiation Parameter (Utility Disutility)", fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_OUTPUT_DIR, "fig4_unbinned_marginal_type_distributions.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    # Figure 6: WTP Boxplots
    wtp_cols = [
        f"wtp_beta_{flv1_name}",
        f"wtp_beta_{flv2_name}",
        "wtp_gamma_plain",
        f"wtp_gamma_{flv1_name}",
    ]
    wtp_melted = draws_df[wtp_cols].melt(var_name="Metric", value_name="WTP")

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=wtp_melted,
        x="Metric",
        y="WTP",
        palette=["#d73027", "#4575b4", "#f46d43", "#74add1"],
        width=0.4,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.title(
        "Distribution of WTP ($) Across Unbinned Individual Flavors",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Willingness-to-Pay ($)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_OUTPUT_DIR, "fig6_unbinned_wtp_distributions.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    console.print(
        f"[bold green]Unbinned type distribution figures saved to {PLOT_OUTPUT_DIR}[/bold green]\n"
    )


def main():
    vec_data = load_and_preprocess_unbinned()

    # Dynamically scale simulation draws matrix to match unbinned dimensions:
    # Dimension = (n_inside - 1) [for Beta draws] + n_inside [for Gamma draws] = 2 * n_inside - 1
    n_inside = vec_data["n_inside"]
    sim_draws = np.random.default_rng(306).standard_normal((N_DRAWS, 2 * n_inside - 1))

    # =========================================================
    # PART 1: NAIVE UNBINNED SPECIFICATIONS (NO SATIATION BENCHMARK)
    # =========================================================
    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow] PART 1: NAIVE UNBINNED SPECIFICATIONS (NO GAMMAS / PLAIN = BASE)[/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    console.print("\n--- Estimating Standard Logit (Unbinned, No Gammas) ---")
    res_naive = estimate_unbinned_naive_model(vec_data)
    display_results_unbinned(
        res_naive,
        title="UNBINNED STANDARD LOGIT (NO SATIATION)",
        vec_data=vec_data,
        model_type="naive",
    )

    console.print("\n--- Estimating Random-Coefficient Logit (Unbinned, No Gammas) ---")
    res_rc_naive = estimate_rc_unbinned_naive_model(vec_data, draws=sim_draws)
    display_results_unbinned(
        res_rc_naive,
        title="UNBINNED RANDOM COEFFICIENT LOGIT (NO SATIATION)",
        vec_data=vec_data,
        model_type="rc_naive",
    )

    # =========================================================
    # PART 2: UNBINNED FLAVOR SATIATION SPECIFICATIONS
    # =========================================================
    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow] PART 2: UNBINNED FLAVOR SATIATION SPECIFICATIONS (WITH GAMMAS)[/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    console.print("\n--- Estimating Standard Unbinned Flavor Satiation Logit ---")
    res_flavor_gamma = estimate_unbinned_flavor_gamma_model(vec_data)
    display_results_unbinned(
        res_flavor_gamma,
        title="UNBINNED STANDARD FLAVOR SATIATION LOGIT",
        vec_data=vec_data,
        model_type="flavor_gamma",
    )

    console.print(
        "\n--- Estimating Random-Coefficient Unbinned Flavor Satiation Logit ---"
    )
    res_rc_flavor_gamma = estimate_rc_unbinned_flavor_gamma_model(
        vec_data, draws=sim_draws
    )
    display_results_unbinned(
        res_rc_flavor_gamma,
        title="UNBINNED RANDOM COEFFICIENT FLAVOR SATIATION LOGIT",
        vec_data=vec_data,
        model_type="rc_flavor_gamma",
    )

    # =========================================================
    # PART 3: GENERATE TYPE & PARAMETER HETEROGENEITY PLOTS
    # =========================================================
    generate_type_plots_unbinned(res_rc_flavor_gamma, vec_data)


if __name__ == "__main__":
    main()
