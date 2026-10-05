# =================================================
# 1. SETUP AND CONFIG
# =================================================

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
CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}
rng = np.random.default_rng(219)


# =================================================
# 2. DATA / INPUT HANDLING
# =================================================


def load_and_preprocess(weekly_capacity=7):
    merged_df = (
        pl.scan_parquet(MERGED_PATH)
        .with_columns(
            [
                pl.col("quantity").cast(pl.Int64),
                pl.col("head_age").cast(pl.Int64),
                pl.col("household_income").cast(pl.Int64),
                pl.col("household_size").cast(pl.Int64),
                pl.col("yogurt_purchase").cast(pl.Int64),
                pl.col("serving_per_container_cd").cast(pl.Int64),
                pl.col("product_module_code_hms").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
            ]
        )
        .filter(pl.col("household_size") == 1)
        .filter(pl.col("serving_per_container_cd").is_in([67181961, 65622705]))
        .collect()
        .to_pandas()
    )

    merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)
    merged_df["flavor_cd"] = pd.to_numeric(
        merged_df["flavor_cd"], errors="coerce"
    ).fillna(0)

    merged_master = merged_df.copy()
    merged_master["flavor"] = np.select(
        [
            merged_master["flavor_str"].str.contains("berry", case=False, na=False),
            merged_master["flavor_cd"].isin([67676592, 66987057]),
        ],
        [1, 2],
        default=0,
    )

    # Petrin & Train Instrument Construction (1st Stage OLS)
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
        "price ~ price_iv + brand_cd + C(week_end)",
        data=merged_master,
        missing="drop",
    ).fit()

    merged_master["iv_res"] = iv_res.resid
    merged_master = merged_master.dropna(subset=["iv_res"])

    def map_flavor_category(flavor):
        if pd.isna(flavor):
            return "outside"
        elif flavor == 0:
            return "other"
        elif flavor == 1:
            return "berry"
        elif flavor == 2:
            return "plain"

    def get_modal_flavor(group):
        if group.empty:
            return np.nan
        counts = group["flavor"].value_counts()
        if counts.empty:
            return np.nan
        max_count = counts.max()
        top_modes = counts[counts == max_count].index.to_numpy()
        return rng.choice(top_modes)

    merged_master["category"] = merged_master["flavor"].map(map_flavor_category)

    weekly_purchases = (
        merged_master[merged_master["category"] != "outside"]
        .groupby(["household_code", "store_code_uc", "week_end", "category"])
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["category"].map(cat_map)

    cat_choice_sets = (
        merged_master.groupby(["store_code_uc", "week_end", "category"])
        .agg(price=("price", "mean"), iv_res=("iv_res", "mean"))
        .reset_index()
    )

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((4, 2))
        mean_store_prices = group["price"].mean()
        mat[:3, 0] = mean_store_prices

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                res_val = 0.0 if np.isnan(row.iv_res) else row.iv_res
                mat[idx] = [row.price, res_val]
        choice_set_matrix[(store, week)] = mat

    hh_weeks = weekly_purchases[
        ["household_code", "store_code_uc", "week_end"]
    ].drop_duplicates()
    hh_weeks = hh_weeks.sort_values(["household_code", "week_end"])

    weekly_modal = (
        merged_master.groupby(["household_code", "week_end"])
        .apply(get_modal_flavor, include_groups=False)
        .reset_index(name="modal_x")
    )
    weekly_modal["theta_prev"] = (
        weekly_modal.groupby("household_code")["modal_x"].shift(1).fillna(0.0)
    )

    hh_weeks = hh_weeks.merge(
        weekly_modal[["household_code", "week_end", "theta_prev"]],
        on=["household_code", "week_end"],
        how="left",
    )

    hh_income_map = (
        merged_df.drop_duplicates(subset=["household_code"])
        .set_index("household_code")["household_income"]
        .to_dict()
    )

    hh_packed_data = {}
    for hh_id, group in hh_weeks.groupby("household_code"):
        stores = group["store_code_uc"].to_numpy()
        weeks = group["week_end"].to_numpy()
        thetas = group["theta_prev"].to_numpy(dtype=np.float64)

        raw_inc = hh_income_map.get(hh_id, 1)
        log_inc = np.log(max(float(raw_inc) if pd.notna(raw_inc) else 1.0, 1.0))

        valid_mask = np.array(
            [(s, w) in choice_set_matrix for s, w in zip(stores, weeks)]
        )
        if not valid_mask.any():
            continue

        matrices_list = []
        choice_counts_list = []
        thetas_list = []

        for store, week, theta in zip(
            stores[valid_mask], weeks[valid_mask], thetas[valid_mask]
        ):
            sub = weekly_purchases[
                (weekly_purchases["household_code"] == hh_id)
                & (weekly_purchases["week_end"] == week)
            ]
            counts = np.zeros(4, dtype=np.int64)
            for row in sub.itertuples():
                counts[row.choice_idx] += row.units_bought

            inside_units = np.sum(counts[:3])
            outside_count = max(0, weekly_capacity - inside_units)
            counts[3] = outside_count

            matrices_list.append(choice_set_matrix[(store, week)])
            choice_counts_list.append(counts)
            thetas_list.append(theta)

        hh_packed_data[hh_id] = {
            "matrices": matrices_list,
            "choices": choice_counts_list,
            "thetas": thetas_list,
            "log_income": log_inc,
        }

    # Diagnostics
    all_inside_units = [
        np.sum(hh_data["choices"][t][:3])
        for hh_data in hh_packed_data.values()
        for t in range(len(hh_data["choices"]))
    ]
    all_outside_counts = [
        hh_data["choices"][t][3]
        for hh_data in hh_packed_data.values()
        for t in range(len(hh_data["choices"]))
    ]

    console.print(
        f"[bold cyan]Mean inside units per week:[/bold cyan] {np.mean(all_inside_units):.2f}"
    )
    console.print(
        f"[bold cyan]Mean outside count:[/bold cyan]          {np.mean(all_outside_counts):.2f}"
    )
    console.print(
        f"[bold cyan]Inside choice share:[/bold cyan]         {np.mean(all_inside_units) / float(weekly_capacity):.1%}"
    )
    console.print(
        f"[bold cyan]Zero outside count share:[/bold cyan]    {np.mean(np.array(all_outside_counts) == 0):.1%}"
    )

    return hh_packed_data


# =================================================
# 3. CORE ESTIMATION LOGIC
# =================================================


def total_objective(params, hh_packed_data):
    beta_oth, beta_ber, beta_pl, gamma, alpha, sigma = params

    beta_vec = np.array([beta_oth, beta_ber, beta_pl])
    cat_flavors = np.array([0.0, 1.0, 2.0])

    total_ll = 0.0

    for hh_data in hh_packed_data.values():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        thetas = hh_data["thetas"]

        for X_mat, counts, theta in zip(matrices, choices, thetas):
            prices = X_mat[:3, 0]
            resids = X_mat[:3, 1]

            Xi = np.abs(cat_flavors - theta)

            u = np.zeros(4)
            u[:3] = beta_vec + gamma * Xi + alpha * prices + sigma * resids
            u[3] = 0.0

            log_prob = u - logsumexp(u)
            week_ll = np.dot(log_prob, counts)

            if not np.isfinite(week_ll):
                week_ll = -700.0

            total_ll += week_ll

    return -total_ll


def estimate_model(hh_packed_data):
    x0 = np.zeros(6)
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, 0.0),
        (None, None),
    ]

    res = minimize(
        total_objective,
        x0=x0,
        args=(hh_packed_data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8},
    )

    cov_matrix = (
        res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else res.hess_inv
    )
    se = np.sqrt(np.diag(cov_matrix))
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


def total_objective_mixed(params, hh_packed_data, n_draws=50):
    (
        mu_b_oth,
        mu_b_berry,
        mu_b_pl,
        mu_gamma,
        alpha_0,
        alpha_inc,
        sigma_cf,
        sd_b_oth,
        sd_b_berry,
        sd_b_pl,
        sd_gamma,
    ) = params

    cat_flavors = np.array([0.0, 1.0, 2.0])
    rng_sim = np.random.default_rng(306)
    total_ll = 0.0

    for hh_data in hh_packed_data.values():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        thetas = hh_data["thetas"]
        log_inc = hh_data["log_income"]

        alpha_i = alpha_0 + alpha_inc * log_inc

        draws_b_oth = mu_b_oth + sd_b_oth * rng_sim.standard_normal(n_draws)
        draws_b_ber = mu_b_berry + sd_b_berry * rng_sim.standard_normal(n_draws)
        draws_b_pl = mu_b_pl + sd_b_pl * rng_sim.standard_normal(n_draws)
        draws_gamma = mu_gamma + sd_gamma * rng_sim.standard_normal(n_draws)

        beta_matrix = np.column_stack([draws_b_oth, draws_b_ber, draws_b_pl])
        log_draw_probs = np.zeros(n_draws)

        for X_mat, counts, theta in zip(matrices, choices, thetas):
            prices = X_mat[:3, 0]
            resids = X_mat[:3, 1]
            Xi = np.abs(cat_flavors - theta)

            u_inside = (
                beta_matrix
                + np.outer(draws_gamma, Xi)
                + alpha_i * prices
                + sigma_cf * resids
            )

            u = np.hstack([u_inside, np.zeros((n_draws, 1))])

            log_probs = u - logsumexp(u, axis=1, keepdims=True)
            log_draw_probs += np.dot(log_probs, counts)

        hh_ll = logsumexp(log_draw_probs) - np.log(n_draws)

        if not np.isfinite(hh_ll):
            total_ll += -1000.0
        else:
            total_ll += hh_ll

    return -total_ll


def estimate_mixed_model(hh_packed_data, n_draws=50):
    x0 = np.array([0.0, 0.0, 0.0, 0.0, -1.5, 0.05, 0.0, 0.1, 0.1, 0.1, 0.1])

    bounds = [
        (None, None),  # mu_b_oth
        (None, None),  # mu_b_ber
        (None, None),  # mu_b_pl
        (None, None),  # mu_gamma
        (None, 0.0),  # alpha_0
        (None, None),  # alpha_inc
        (None, None),  # sigma_cf
        (1e-4, None),  # sd_b_oth
        (1e-4, None),  # sd_b_ber
        (1e-4, None),  # sd_b_pl
        (1e-4, None),  # sd_gamma
    ]

    res = minimize(
        total_objective_mixed,
        x0=x0,
        args=(hh_packed_data, n_draws),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxiter": 1000,
        },
    )

    cov_matrix = (
        res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else res.hess_inv
    )
    se = np.sqrt(np.diag(cov_matrix))
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


# =================================================
# 4. MAIN & PRINTING
# =================================================


def print_simple_results(results):
    param_names = [
        "beta_other",
        "beta_berry",
        "beta_plain",
        "gamma (switching)",
        "alpha (price)",
        "sigma (control_func)",
    ]

    table = Table(title="Simple Logit Estimation Results (Occasion-to-Occasion Only)")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Estimate", justify="right", style="green")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    for name, est, se, z, p in zip(
        param_names,
        results["params"],
        results["se"],
        results["z_stat"],
        results["p_val"],
    ):
        table.add_row(name, f"{est:.4f}", f"{se:.4f}", f"{z:.3f}", f"{p:.4f}")

    console.print(table)


def print_mixed_results(results):
    param_names = [
        "mu_beta_other",
        "mu_beta_berry",
        "mu_beta_plain",
        "mu_gamma (switching)",
        "alpha_0 (base price)",
        "alpha_inc (inc price interaction)",
        "sigma_cf (control_func)",
        "sd_beta_other",
        "sd_beta_berry",
        "sd_beta_plain",
        "sd_gamma",
    ]

    table = Table(title="Mixed Logit Estimation Results (Occasion-to-Occasion Only)")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Estimate", justify="right", style="green")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    for name, est, se, z, p in zip(
        param_names,
        results["params"],
        results["se"],
        results["z_stat"],
        results["p_val"],
    ):
        table.add_row(name, f"{est:.4f}", f"{se:.4f}", f"{z:.3f}", f"{p:.4f}")

    console.print(table)


def main():
    hh_packed_data = load_and_preprocess()

    console.print("\n[bold yellow]--- Estimating LOV Simple Logit ---[/bold yellow]")
    simple_results = estimate_model(hh_packed_data)
    console.print(f"Optimization Success: {simple_results['success']}")
    console.print(f"Final Log-Likelihood: {-simple_results['fun']:.4f}\n")
    print_simple_results(simple_results)

    console.print("\n[bold yellow]--- Estimating LOV Mixed Logit ---[/bold yellow]")
    mixed_results = estimate_mixed_model(hh_packed_data, n_draws=50)
    console.print(f"Mixed Optimization Success: {mixed_results['success']}")
    console.print(f"Final Log-Likelihood: {-mixed_results['fun']:.4f}\n")
    print_mixed_results(mixed_results)


if __name__ == "__main__":
    main()
