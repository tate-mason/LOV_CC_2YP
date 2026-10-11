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
PLOT_OUTPUT_DIR = "../../../../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}
LAMBDA_GRID = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)

# Standard normal draws for mixed logit simulation (50 draws, 4 dimensions: beta_ber, beta_pl, gamma_ber, gamma_pl)
GAMMA_DRAWS = np.random.default_rng(306).standard_normal((50, 4))


def compute_satiation_state(
    weekly_purchases,
    n_categories=3,
    lambda_mem=0.7,
    delta_discount=0.9,
    rho_weights=None,
):
    if rho_weights is None:
        rho_weights = np.ones(n_categories)

    sorted_df = weekly_purchases.sort_values(["household_code", "week_end"])
    inventory_records = []

    for hh_id, hh_group in sorted_df.groupby("household_code"):
        weeks = hh_group["week_end"].unique()
        C_history = {}
        prev_X = np.zeros(n_categories)

        for t_idx, week in enumerate(weeks):
            C_current = np.zeros(n_categories)
            memory_term = np.zeros(n_categories)

            if t_idx > 0:
                for s in range(t_idx):
                    discount = delta_discount ** (t_idx - s)
                    memory_term += discount * C_history[s]
            for j in range(n_categories):
                cross_term = 0.0
                for k in range(n_categories):
                    if k != j:
                        cross_term += rho_weights[k] * prev_X[k]
                C_current[j] = (lambda_mem * memory_term[j]) + (
                    (1.0 - lambda_mem) * cross_term
                )

            record = {"household_code": hh_id, "week_end": week}
            for j in range(n_categories):
                record[f"C_sat_{j}"] = C_current[j]
            inventory_records.append(record)

            C_history[t_idx] = C_current.copy()

            sub = hh_group[hh_group["week_end"] == week]
            curr_X = np.zeros(n_categories)
            for row in sub.itertuples():
                if row.choice_idx < n_categories:
                    curr_X[row.choice_idx] = row.units_bought
            total_inside_units = curr_X.sum()
            if total_inside_units > 0:
                curr_X = curr_X / total_inside_units
            prev_X = curr_X
    return pd.DataFrame(inventory_records)


def load_and_preprocess(weekly_capacity=14):
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

    merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)
    merged_df["flavor_cd"] = pd.to_numeric(
        merged_df["flavor_cd"], errors="coerce"
    ).fillna(0)

    merged_master = merged_df.copy()

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

    merged_master["iv_res"] = merged_master["iv_res"].fillna(0.0)

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

    weekly_purchases = (
        inside_df.groupby(["household_code", "store_code_uc", "week_end", "category"])
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["category"].map(cat_map)

    cat_choice_sets = (
        inside_df.groupby(["store_code_uc", "week_end", "category"])
        .agg(
            price=("price", "mean"),
            iv_res=("iv_res", "mean"),
        )
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
        mat = np.zeros((4, 2), dtype=np.float64)

        for cat, idx in cat_map.items():
            if cat != "outside" and cat in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat]

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    # Pre-compute satiation state across lambda grid
    console.print(
        "[bold green]Pre-computing satiation states across lambda grid...[/bold green]"
    )
    df_sat_grid = {}
    for lam in LAMBDA_GRID:
        df_sat = compute_satiation_state(
            weekly_purchases, lambda_mem=lam, delta_discount=0.9
        )
        for j in range(3):
            c_col = f"C_sat_{j}"
            c_sd = df_sat[c_col].std()
            if c_sd > 0:
                df_sat[c_col] = df_sat[c_col] / c_sd
        df_sat_grid[lam] = df_sat

    hh_weeks = weekly_purchases[
        ["household_code", "store_code_uc", "week_end"]
    ].drop_duplicates()

    for lam in LAMBDA_GRID:
        df_sat = df_sat_grid[lam]
        hh_weeks = hh_weeks.merge(
            df_sat.rename(
                columns={
                    "C_sat_0": f"C_sat_0_{lam:.1f}",
                    "C_sat_1": f"C_sat_1_{lam:.1f}",
                    "C_sat_2": f"C_sat_2_{lam:.1f}",
                }
            ),
            on=["household_code", "week_end"],
            how="left",
        ).fillna(0.0)

    hh_weeks = hh_weeks.sort_values(["household_code", "week_end"])

    hh_income_map = (
        merged_df.drop_duplicates(subset=["household_code"])
        .set_index("household_code")["household_income"]
        .to_dict()
    )

    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(3, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < 3:
                c_arr[row.choice_idx] = row.units_bought
        counts_dict[(hh_id, week)] = c_arr

    hh_packed_data = {}
    all_prices, all_resids, all_choices, all_c_grids, all_log_inc = (
        [],
        [],
        [],
        [],
        [],
    )

    hh_weeks_dict = hh_weeks.to_dict("records")

    for row_dict in hh_weeks_dict:
        store = row_dict["store_code_uc"]
        week = row_dict["week_end"]
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        raw_inc = hh_income_map.get(hh_id, 1)
        log_inc = np.log(max(float(raw_inc) if pd.notna(raw_inc) else 1.0, 1.0))

        inside_counts = counts_dict.get((hh_id, week), np.zeros(3, dtype=np.int64))
        inside_units = np.sum(inside_counts)

        effective_capacity = max(weekly_capacity, inside_units + 1)
        outside_count = effective_capacity - inside_units

        full_counts = np.append(inside_counts, outside_count)

        c_grid_mat = np.zeros((3, len(LAMBDA_GRID)), dtype=np.float64)
        for idx_lam, lam in enumerate(LAMBDA_GRID):
            c_grid_mat[0, idx_lam] = row_dict[f"C_sat_0_{lam:.1f}"]
            c_grid_mat[1, idx_lam] = row_dict[f"C_sat_1_{lam:.1f}"]
            c_grid_mat[2, idx_lam] = row_dict[f"C_sat_2_{lam:.1f}"]

        all_prices.append(choice_set_matrix[(store, week)][:3, 0])
        all_resids.append(choice_set_matrix[(store, week)][:3, 1])
        all_choices.append(full_counts)
        all_c_grids.append(c_grid_mat)
        all_log_inc.append(log_inc)

        if hh_id not in hh_packed_data:
            hh_packed_data[hh_id] = {
                "matrices": [],
                "choices": [],
                "c_states": [],
                "log_income": log_inc,
            }
        hh_packed_data[hh_id]["matrices"].append(choice_set_matrix[(store, week)])
        hh_packed_data[hh_id]["choices"].append(full_counts)
        hh_packed_data[hh_id]["c_states"].append(c_grid_mat[:, 3])  # Baseline lam=0.7

    n_hh = len(hh_packed_data)
    total_obs = sum(len(hh["choices"]) for hh in hh_packed_data.values())
    avg_obs_per_hh = total_obs / float(n_hh) if n_hh > 0 else 0.0

    console.print(
        "\n[bold green]=====================================================[/bold green]"
    )
    console.print(
        "[bold green]          PANEL SAMPLE HOUSEHOLD SUMMARY             [/bold green]"
    )
    console.print(
        "[bold green]=====================================================[/bold green]"
    )
    console.print(
        f"[bold white]Total Unique Households:[/bold white]       [bold cyan]{n_hh:,}[/bold cyan]"
    )
    console.print(
        f"[bold white]Total Choice Observations:[/bold white]     [bold cyan]{total_obs:,}[/bold cyan]"
    )
    console.print(
        f"[bold white]Mean Purchase Weeks / HH:[/bold white]      [bold cyan]{avg_obs_per_hh:.2f}[/bold cyan]"
    )
    console.print(
        "[bold green]=====================================================\n[/bold green]"
    )

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array([c[:, 3] for c in all_c_grids], dtype=np.float64),
        "c_grids": np.array(all_c_grids, dtype=np.float64),
        "log_inc": np.array(all_log_inc, dtype=np.float64),
    }

    return hh_packed_data, vec_data


# =========================================================
# PART 1: NAIVE SPECIFICATIONS (NO SATIATION BENCHMARK)
# =========================================================
def total_objective_naive(params, vec_data):
    const, beta_ber, beta_pl, alpha, sigma = params
    beta_vec = np.array([0.0, beta_ber, beta_pl])

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]

    u_inside = const + beta_vec + alpha * prices + sigma * resids
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def total_objective_rc_naive(params, vec_data, draws):
    (
        const,
        mu_beta_ber,
        mu_beta_pl,
        sd_beta_ber,
        sd_beta_pl,
        alpha,
        sigma,
    ) = params

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]

    n_draws = draws.shape[0]

    b_ber_draws = mu_beta_ber + sd_beta_ber * draws[:, 0]
    b_pl_draws = mu_beta_pl + sd_beta_pl * draws[:, 1]

    beta_draws_matrix = np.column_stack([np.zeros(n_draws), b_ber_draws, b_pl_draws])

    u_base = const + alpha * prices + sigma * resids
    u_inside = u_base[:, None, :] + beta_draws_matrix[None, :, :]
    u_out = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_out], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll = logsumexp(log_probs, axis=1) - np.log(n_draws)
    total_ll = np.sum(choices * obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_naive_model(vec_data):
    x0 = np.array([0.0, 0.0, 0.0, -0.1, 0.0])
    bounds = [(None, None)] * 3 + [(None, 0.0), (None, None)]

    res = minimize(
        total_objective_naive,
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

            f1 = total_objective_naive(x1, vec_data)
            f2 = total_objective_naive(x2, vec_data)
            f3 = total_objective_naive(x3, vec_data)
            f4 = total_objective_naive(x4, vec_data)

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


def estimate_rc_naive_model(vec_data, draws=GAMMA_DRAWS):
    x0 = np.array([0.0, 0.0, 0.0, 0.5, 0.5, -0.1, 0.0])
    bounds = [
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (0.05, 5.0),
        (0.05, 5.0),
        (-10.0, -0.01),
        (-10.0, 10.0),
    ]

    res = minimize(
        total_objective_rc_naive,
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

            f1 = total_objective_rc_naive(x1, vec_data, draws)
            f2 = total_objective_rc_naive(x2, vec_data, draws)
            f3 = total_objective_rc_naive(x3, vec_data, draws)
            f4 = total_objective_rc_naive(x4, vec_data, draws)

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
# PART 2: FLAVOR SATIATION SPECIFICATIONS (WITH GAMMAS)
# =========================================================
def total_objective_flavor_gamma(params, vec_data):
    const, beta_ber, beta_pl, g_ber, g_pl, alpha, sigma = params
    beta_vec = np.array([0.0, beta_ber, beta_pl])
    gamma_vec = np.array([0.0, g_ber, g_pl])

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


def total_objective_het_flavor_gamma_fixed_lambda(params, vec_data, draws):
    (
        const,
        mu_beta_ber,
        mu_beta_pl,
        mu_g_ber,
        mu_g_pl,
        sd_beta_ber,
        sd_beta_pl,
        sd_g_ber,
        sd_g_pl,
        alpha,
        sigma,
    ) = params

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]
    c_state = vec_data["c_states"]

    n_draws = draws.shape[0]

    b_ber_draws = mu_beta_ber + sd_beta_ber * draws[:, 0]
    b_pl_draws = mu_beta_pl + sd_beta_pl * draws[:, 1]
    g_ber_draws = mu_g_ber + sd_g_ber * draws[:, 2]
    g_pl_draws = mu_g_pl + sd_g_pl * draws[:, 3]

    beta_draws_matrix = np.column_stack([np.zeros(n_draws), b_ber_draws, b_pl_draws])
    gamma_draws_matrix = np.column_stack([np.zeros(n_draws), g_ber_draws, g_pl_draws])

    u_base = const + alpha * prices + sigma * resids
    u_inside = (
        u_base[:, None, :]
        + beta_draws_matrix[None, :, :]
        + gamma_draws_matrix[None, :, :] * np.log(1.0 + c_state[:, None, :])
    )

    u_out = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_out], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll = logsumexp(log_probs, axis=1) - np.log(n_draws)
    total_ll = np.sum(choices * obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_flavor_gamma_model(vec_data):
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0])
    bounds = [(None, None)] * 5 + [(None, 0.0), (None, None)]

    res = minimize(
        total_objective_flavor_gamma,
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

            f1 = total_objective_flavor_gamma(x1, vec_data)
            f2 = total_objective_flavor_gamma(x2, vec_data)
            f3 = total_objective_flavor_gamma(x3, vec_data)
            f4 = total_objective_flavor_gamma(x4, vec_data)

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


def estimate_het_flavor_gamma_model_fixed_lambda(vec_data, draws=GAMMA_DRAWS):
    x0 = np.array([0.0, -0.7, -1.3, -0.7, -0.1, 0.5, 0.5, 0.5, 0.5, -0.98, 0.16])
    bounds = [
        (-10.0, 10.0),  # const
        (-10.0, 10.0),  # mu_beta_ber
        (-10.0, 10.0),  # mu_beta_pl
        (-10.0, 0.0),  # mu_g_ber
        (-10.0, 0.0),  # mu_g_pl
        (1e-4, 5.0),  # sd_beta_ber
        (1e-4, 5.0),  # sd_beta_pl
        (1e-4, 5.0),  # sd_g_ber
        (1e-4, 5.0),  # sd_g_pl
        (-10.0, -0.01),  # alpha
        (-10.0, 10.0),  # sigma
    ]

    res = minimize(
        total_objective_het_flavor_gamma_fixed_lambda,
        x0=x0,
        args=(vec_data, draws[:, :4]),  # First 4 dimensions of simulation draws
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

            f1 = total_objective_het_flavor_gamma_fixed_lambda(
                x1, vec_data, draws[:, :4]
            )
            f2 = total_objective_het_flavor_gamma_fixed_lambda(
                x2, vec_data, draws[:, :4]
            )
            f3 = total_objective_het_flavor_gamma_fixed_lambda(
                x3, vec_data, draws[:, :4]
            )
            f4 = total_objective_het_flavor_gamma_fixed_lambda(
                x4, vec_data, draws[:, :4]
            )

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


def profile_likelihood_lambda_search(vec_data, draws=GAMMA_DRAWS):
    """Profile Likelihood Grid Search across candidate lambda layers in [0.1, 0.9]."""
    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow]   RUNNING PROFILE LIKELIHOOD GRID SEARCH FOR LAMBDA  [/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    best_ll = 1e10
    best_res = None
    best_lam = None

    grid_results = []

    for idx, lam in enumerate(LAMBDA_GRID):
        vec_data_single = vec_data.copy()
        vec_data_single["c_states"] = vec_data["c_grids"][:, :, idx]

        res = estimate_het_flavor_gamma_model_fixed_lambda(vec_data_single, draws=draws)

        grid_results.append((lam, res["fun"], res))
        console.print(
            f"[cyan]Grid Search λ = {lam:.1f}[/cyan] | [bold white]Log-Likelihood Objective:[/bold white] [bold yellow]{res['fun']:.4f}[/bold yellow]"
        )

        if res["fun"] < best_ll:
            best_ll = res["fun"]
            best_res = res
            best_lam = lam

    console.print(
        f"\n[bold green]Optimal Population Memory Weight (λ*): {best_lam:.1f} (LL = {best_ll:.4f})[/bold green]\n"
    )
    return best_res, best_lam


def display_results_with_wtp(
    results,
    title="SPECIFICATION RESULTS",
    param_names=None,
    model_type="flavor_gamma",
    lambda_opt=None,
):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    alpha_val = results["params"][-2]

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
    if lambda_opt is not None:
        console.print(
            f"[bold yellow]Profile-Likelihood Optimal Memory Weight (λ*):[/bold yellow] [bold cyan]{lambda_opt:.1f}[/bold cyan]"
        )
    console.print(f"[bold]Optimization Success:[/bold] {results['success']}")
    console.print(f"[bold]Final LL Objective:[/bold] {results['fun']:.4f}")

    if abs(alpha_val) > 1e-4:
        wtp_table = Table(
            title=f"{title} - WILLINGNESS TO PAY (WTP in $)",
            show_header=True,
            header_style="bold yellow",
        )
        wtp_table.add_column("Attribute / Parameter", style="cyan", justify="left")
        wtp_table.add_column("WTP ($)", justify="right")

        if model_type == "flavor_gamma":
            beta_ber, beta_pl = results["params"][1], results["params"][2]
            g_ber, g_pl = results["params"][3], results["params"][4]

            wtp_table.add_row(
                "WTP: Berry Preference (vs Other)", f"${beta_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Plain Preference (vs Other)", f"${beta_pl / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Berry)", f"${g_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Plain)", f"${g_pl / alpha_val:.2f}"
            )

        elif model_type == "naive":
            beta_ber, beta_pl = results["params"][1], results["params"][2]
            wtp_table.add_row(
                "WTP: Berry Preference (vs Other)", f"${beta_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Plain Preference (vs Other)", f"${beta_pl / alpha_val:.2f}"
            )

        console.print(wtp_table)
    console.print("\n")


# ==============================================================================
# INTEGRATED TYPE & PARAMETER HETEROGENEITY PLOTTING FUNCTIONS
# ==============================================================================
def plot_marginal_type_distributions(draws_df):
    console.print(
        "[bold green]Generating Figure 4: Marginal Type Distributions...[/bold green]"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(
        draws_df["beta_berry"],
        ax=axes[0],
        color="#d73027",
        fill=True,
        alpha=0.4,
        label=r"Berry Preference ($\beta_{berry}$)",
    )
    sns.kdeplot(
        draws_df["beta_plain"],
        ax=axes[0],
        color="#4575b4",
        fill=True,
        alpha=0.4,
        label=r"Plain Preference ($\beta_{plain}$)",
    )
    axes[0].set_title("Distribution of Flavor Preferences", fontweight="bold")
    axes[0].set_xlabel("Preference Parameter (Utility)", fontweight="bold")
    axes[0].set_ylabel("Density", fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white")

    sns.kdeplot(
        draws_df["gamma_berry"],
        ax=axes[1],
        color="#d73027",
        fill=True,
        alpha=0.4,
        label=r"Berry Satiation ($\gamma_{berry}$)",
    )
    sns.kdeplot(
        draws_df["gamma_plain"],
        ax=axes[1],
        color="#4575b4",
        fill=True,
        alpha=0.4,
        label=r"Plain Satiation ($\gamma_{plain}$)",
    )
    axes[1].set_title("Distribution of Satiation Parameters", fontweight="bold")
    axes[1].set_xlabel("Satiation Parameter (Utility Disutility)", fontweight="bold")
    axes[1].set_ylabel("Density", fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig_path = os.path.join(PLOT_OUTPUT_DIR, "fig4_marginal_type_distributions.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_joint_type_distributions(draws_df):
    console.print(
        "[bold green]Generating Figure 5: Joint Beta vs. Gamma Distribution...[/bold green]"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(
        data=draws_df,
        x="beta_berry",
        y="gamma_berry",
        ax=axes[0],
        cmap="Reds",
        fill=True,
        thresh=0.05,
    )
    axes[0].set_title("Berry: Preference vs. Satiation", fontweight="bold")
    axes[0].set_xlabel(r"Baseline Preference ($\beta_{berry}$)", fontweight="bold")
    axes[0].set_ylabel(r"Satiation Parameter ($\gamma_{berry}$)", fontweight="bold")

    sns.kdeplot(
        data=draws_df,
        x="beta_plain",
        y="gamma_plain",
        ax=axes[1],
        cmap="Blues",
        fill=True,
        thresh=0.05,
    )
    axes[1].set_title("Plain: Preference vs. Satiation", fontweight="bold")
    axes[1].set_xlabel(r"Baseline Preference ($\beta_{plain}$)", fontweight="bold")
    axes[1].set_ylabel(r"Satiation Parameter ($\gamma_{plain}$)", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(PLOT_OUTPUT_DIR, "fig5_joint_type_distributions.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_wtp_distributions(draws_df):
    console.print(
        "[bold green]Generating Figure 6: Willingness-to-Pay Distributions...[/bold green]"
    )

    wtp_melted = draws_df[
        ["wtp_beta_berry", "wtp_beta_plain", "wtp_gamma_berry", "wtp_gamma_plain"]
    ].melt(var_name="Metric", value_name="WTP")

    wtp_map = {
        "wtp_beta_berry": "Berry Preference",
        "wtp_beta_plain": "Plain Preference",
        "wtp_gamma_berry": "Berry Satiation Penalty",
        "wtp_gamma_plain": "Plain Satiation Penalty",
    }
    wtp_melted["Metric"] = wtp_melted["Metric"].map(wtp_map)

    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=wtp_melted,
        x="Metric",
        y="WTP",
        palette=["#d73027", "#4575b4", "#f46d43", "#74add1"],
        width=0.4,
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.title(
        "Distribution of Household Willingness-to-Pay (WTP in $)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("")
    plt.ylabel("Willingness-to-Pay ($)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=15, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(PLOT_OUTPUT_DIR, "fig6_wtp_distributions.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()


def generate_type_plots(res_rc_flavor_gamma, n_draws=5000):
    params = res_rc_flavor_gamma["params"]
    mu_b_ber, mu_b_pl = params[1], params[2]
    mu_g_ber, mu_g_pl = params[3], params[4]
    sd_b_ber, sd_b_pl = params[5], params[6]
    sd_g_ber, sd_g_pl = params[7], params[8]
    alpha_val = abs(params[9])

    plot_rng = np.random.default_rng(2026)

    draws_df = pd.DataFrame(
        {
            "beta_berry": plot_rng.normal(mu_b_ber, sd_b_ber, n_draws),
            "beta_plain": plot_rng.normal(mu_b_pl, sd_b_pl, n_draws),
            "gamma_berry": plot_rng.normal(mu_g_ber, sd_g_ber, n_draws),
            "gamma_plain": plot_rng.normal(mu_g_pl, sd_g_pl, n_draws),
        }
    )

    draws_df["wtp_beta_berry"] = draws_df["beta_berry"] / alpha_val
    draws_df["wtp_beta_plain"] = draws_df["beta_plain"] / alpha_val
    draws_df["wtp_gamma_berry"] = draws_df["gamma_berry"] / alpha_val
    draws_df["wtp_gamma_plain"] = draws_df["gamma_plain"] / alpha_val

    plot_marginal_type_distributions(draws_df)
    plot_joint_type_distributions(draws_df)
    plot_wtp_distributions(draws_df)

    console.print(
        f"[bold green]Type distribution figures saved to {PLOT_OUTPUT_DIR}[/bold green]\n"
    )


def main():
    hh_packed_data, vec_data = load_and_preprocess()

    # =========================================================
    # PART 1: NAIVE SPECIFICATIONS (NO SATIATION BENCHMARK)
    # =========================================================
    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow] PART 1: NAIVE SPECIFICATIONS (NO SATIATION / NO GAMMAS)[/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    console.print("\n--- Estimating Standard Logit (No Gammas) ---")
    res_naive = estimate_naive_model(vec_data)
    p_names_naive = ["Constant", "beta_ber", "beta_pl", "Price", "Control Func."]
    display_results_with_wtp(
        res_naive,
        title="STANDARD LOGIT (NO SATIATION)",
        param_names=p_names_naive,
        model_type="naive",
    )

    console.print("\n--- Estimating Random-Coefficient Logit (No Gammas) ---")
    res_rc_naive = estimate_rc_naive_model(vec_data, draws=GAMMA_DRAWS)
    rc_p_names_naive = [
        "Constant",
        "Mean beta_berry",
        "Mean beta_plain",
        "SD beta_berry",
        "SD beta_plain",
        "Price",
        "Control Func.",
    ]
    display_results_with_wtp(
        res_rc_naive,
        title="RANDOM COEFFICIENT LOGIT (NO SATIATION)",
        param_names=rc_p_names_naive,
        model_type="naive",
    )

    # =========================================================
    # PART 2: FLAVOR SATIATION SPECIFICATIONS (PROFILE LIKELIHOOD LAMBDA)
    # =========================================================
    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow] PART 2: FLAVOR SATIATION SPECIFICATIONS (WITH GAMMAS)[/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    console.print("\n--- Estimating Standard Flavor-Specific Satiation Logit ---")
    res_flavor_gamma = estimate_flavor_gamma_model(vec_data)
    p_names_gamma = [
        "Constant",
        "beta_ber",
        "beta_pl",
        "gamma_berry",
        "gamma_plain",
        "Price",
        "Control Func.",
    ]
    display_results_with_wtp(
        res_flavor_gamma,
        title="STANDARD FLAVOR SATIATION LOGIT",
        param_names=p_names_gamma,
        model_type="flavor_gamma",
    )

    # Run Profile Likelihood Grid Search across lambda = [0.1, 0.3, 0.5, 0.7, 0.9]
    res_rc_flavor_gamma, opt_lambda = profile_likelihood_lambda_search(
        vec_data, draws=GAMMA_DRAWS
    )

    rc_p_names_gamma = [
        "Constant",
        "Mean beta_berry",
        "Mean beta_plain",
        "Mean gamma_berry",
        "Mean gamma_plain",
        "SD beta_berry",
        "SD beta_plain",
        "SD gamma_berry",
        "SD gamma_plain",
        "Price",
        "Control Func.",
    ]
    display_results_with_wtp(
        res_rc_flavor_gamma,
        title="RANDOM COEFFICIENT FLAVOR SATIATION LOGIT (PROFILE LAMBDA)",
        param_names=rc_p_names_gamma,
        model_type="flavor_gamma",
        lambda_opt=opt_lambda,
    )

    # =========================================================
    # PART 3: GENERATE TYPE & PARAMETER HETEROGENEITY PLOTS
    # =========================================================
    generate_type_plots(res_rc_flavor_gamma)


if __name__ == "__main__":
    main()
