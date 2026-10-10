import os
import pandas as pd
import polars as pl

import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from rich.console import Console
from rich.table import Table
from rich.traceback import install

install()
console = Console()

MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Output/"
CATEGORIES = ["other", "berry", "plain", "outside"]

cat_map = {c: i for i, c in enumerate(CATEGORIES)}
rng = np.random.default_rng(219)
GAMMA_DRAWS = np.random.default_rng(306).standard_normal((50, 5))


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
            prev_X = curr_X
    return pd.DataFrame(inventory_records)


def load_and_preprocess(weekly_capacity=14, inside_only=False):
    # Load pre-filtered panel directly (no redundant merges required)
    # Load pre-filtered panel directly
    merged_df = pl.read_parquet(MERGED_PATH).to_pandas()

    # 1. Cast numeric quantities safely
    merged_df["quantity"] = (
        pd.to_numeric(merged_df["quantity"], errors="coerce").fillna(1).astype(int)
    )
    merged_df["total_price_paid"] = pd.to_numeric(
        merged_df["total_price_paid"], errors="coerce"
    ).fillna(0.0)

    if "price" not in merged_df.columns:
        merged_df["price"] = np.where(
            merged_df["quantity"] > 0,
            merged_df["total_price_paid"] / merged_df["quantity"],
            np.nan,
        )

    merged_df["household_income"] = (
        pd.to_numeric(merged_df["household_income"], errors="coerce")
        .fillna(1)
        .astype(int)
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
        "price  ~ price_iv + brand_cd + C(week_end)", data=merged_master, missing="drop"
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

    df_sat = compute_satiation_state(
        weekly_purchases, lambda_mem=0.7, delta_discount=0.9
    )
    for j in range(3):
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

    for hh_id, group in hh_weeks.groupby("household_code"):
        raw_inc = hh_income_map.get(hh_id, 1)
        log_inc = np.log(max(float(raw_inc) if pd.notna(raw_inc) else 1.0, 1.0))

        matrices_list = []
        choice_counts_list = []
        c_states_list = []

        for row in group.itertuples():
            store = row.store_code_uc
            week = row.week_end

            if (store, week) not in choice_set_matrix:
                continue

            inside_counts = counts_dict.get((hh_id, week), np.zeros(3, dtype=np.int64))
            inside_units = np.sum(inside_counts)

            if inside_only and inside_units == 0:
                continue

            cap = 1 if inside_only else weekly_capacity
            effective_capacity = max(cap, inside_units + (0 if inside_only else 1))
            outside_count = max(0, effective_capacity - inside_units)

            full_counts = np.append(inside_counts, outside_count)
            c_vec = np.array([row.C_sat_0, row.C_sat_1, row.C_sat_2], dtype=np.float64)

            matrices_list.append(choice_set_matrix[(store, week)])
            choice_counts_list.append(full_counts)
            c_states_list.append(c_vec)

            if len(matrices_list) > 0:
                hh_packed_data[hh_id] = {
                    "matrices": matrices_list,
                    "choices": choice_counts_list,
                    "c_states": c_states_list,
                    "log_income": log_inc,
                }

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

    all_prices = []
    all_resids = []
    all_choices = []
    all_c_states = []
    all_log_inc = []

    for hh_id, hh_data in hh_packed_data.items():
        for m, c, cs in zip(
            hh_data["matrices"], hh_data["choices"], hh_data["c_states"]
        ):
            all_prices.append(m[:3, 0])
            all_resids.append(m[:3, 1])
            all_choices.append(c)
            all_c_states.append(cs)
            all_log_inc.append(hh_data["log_income"])

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "log_inc": np.array(all_log_inc, dtype=np.float64),
    }

    return hh_packed_data, vec_data


# =========================================================
# PRODUCT-SPECIFIC GAMMAS & WTP ESTIMATION
# =========================================================
def total_objective_flavor_gamma(params, vec_data):
    const, beta_ber, beta_pl, g_oth, g_ber, g_pl, alpha, sigma = params
    beta_vec = np.array([0.0, beta_ber, beta_pl])
    gamma_vec = np.array([g_oth, g_ber, g_pl])

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]
    c_state = vec_data["c_states"]

    u_inside = const + beta_vec + gamma_vec * c_state + alpha * prices + sigma * resids
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def total_objective_het_flavor_gamma(params, vec_data, draws):
    (
        const,
        mu_beta_ber,
        mu_beta_pl,
        mu_g_oth,
        mu_g_ber,
        mu_g_pl,
        log_sd_beta_ber,
        log_sd_beta_pl,
        log_sd_g_oth,
        log_sd_g_ber,
        log_sd_g_pl,
        alpha,
        sigma,
    ) = params

    sd_beta_ber = np.exp(log_sd_beta_ber)
    sd_beta_pl = np.exp(log_sd_beta_pl)
    sd_g_oth = np.exp(log_sd_g_oth)
    sd_g_ber = np.exp(log_sd_g_ber)
    sd_g_pl = np.exp(log_sd_g_pl)

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]
    c_state = vec_data["c_states"]

    n_draws = draws.shape[0]

    b_ber_draws = mu_beta_ber + sd_beta_ber * draws[:, 0]
    b_pl_draws = mu_beta_pl + sd_beta_pl * draws[:, 1]
    g_oth_draws = mu_g_oth + sd_g_oth * draws[:, 2]
    g_ber_draws = mu_g_ber + sd_g_ber * draws[:, 3]
    g_pl_draws = mu_g_pl + sd_g_pl * draws[:, 4]

    beta_draws_matrix = np.column_stack([np.zeros(n_draws), b_ber_draws, b_pl_draws])
    gamma_draws_matrix = np.column_stack([g_oth_draws, g_ber_draws, g_pl_draws])

    u_base = const + alpha * prices + sigma * resids
    u_inside = (
        u_base[:, None, :]
        + beta_draws_matrix[None, :, :]
        + gamma_draws_matrix[None, :, :] * c_state[:, None, :]
    )
    u_out = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_out], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll = logsumexp(log_probs, axis=1) - np.log(n_draws)
    total_ll = np.sum(choices * obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_flavor_gamma_model(vec_data):
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])
    bounds = [(None, None)] * 6 + [(None, 0.0), (None, None)]

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


def estimate_het_flavor_gamma_model(vec_data, draws=GAMMA_DRAWS):
    x0 = np.array(
        [
            0.95,  # const
            -0.76,  # mu_beta_ber
            -1.39,  # mu_beta_pl
            -0.10,  # mu_g_oth
            -0.15,  # mu_g_ber
            -0.20,  # mu_g_pl
            -2.30,  # log_sd_beta_ber
            -2.30,  # log_sd_beta_pl
            -3.00,  # log_sd_g_oth
            -3.00,  # log_sd_g_ber
            -3.00,  # log_sd_g_pl
            -0.98,  # alpha
            0.16,  # sigma
        ]
    )

    bounds = [
        (-10.0, 10.0),  # const
        (-10.0, 10.0),  # mu_beta_ber
        (-10.0, 10.0),  # mu_beta_pl
        (-5.0, 2.0),  # mu_g_oth
        (-5.0, 2.0),  # mu_g_ber
        (-5.0, 2.0),  # mu_g_pl
        (-10.0, 5.0),  # log_sd_beta_ber
        (-10.0, 5.0),  # log_sd_beta_pl
        (-10.0, 5.0),  # log_sd_g_oth
        (-10.0, 5.0),  # log_sd_g_ber
        (-10.0, 5.0),  # log_sd_g_pl
        (-10.0, -0.01),  # alpha
        (-10.0, 10.0),  # sigma
    ]

    res = minimize(
        total_objective_het_flavor_gamma,
        x0=x0,
        args=(vec_data, draws),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-7, "gtol": 1e-5, "maxiter": 1000},
    )

    # Finite difference Hessian computation in unconstrained space
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

            f1 = total_objective_het_flavor_gamma(x1, vec_data, draws)
            f2 = total_objective_het_flavor_gamma(x2, vec_data, draws)
            f3 = total_objective_het_flavor_gamma(x3, vec_data, draws)
            f4 = total_objective_het_flavor_gamma(x4, vec_data, draws)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    # Invert Hessian using pseudo-inverse for numerical stability
    try:
        cov_unconstrained = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov_unconstrained = np.linalg.pinv(hessian)

    se_unconstrained = np.sqrt(np.maximum(0.0, np.diag(cov_unconstrained)))

    # Apply Delta Method: Var(exp(theta)) = exp(theta)^2 * Var(theta)
    sd_indices = [6, 7, 8, 9, 10]
    reported_params = res.x.copy()
    reported_se = se_unconstrained.copy()

    for idx in sd_indices:
        reported_params[idx] = np.exp(res.x[idx])
        reported_se[idx] = np.exp(res.x[idx]) * se_unconstrained[idx]

    z = reported_params / reported_se
    p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

    return {
        "params": reported_params,
        "se": reported_se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def display_results_with_wtp(
    results, title="SPECIFICATION RESULTS", param_names=None, is_flavor_gamma=False
):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    alpha_val = results["params"][-2]  # Price parameter is second to last

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

    # Compute and display Willingness-To-Pay (WTP) Table
    if abs(alpha_val) > 1e-4:
        wtp_table = Table(
            title=f"{title} - WILLINGNESS TO PAY (WTP in $)",
            show_header=True,
            header_style="bold yellow",
        )
        wtp_table.add_column("Attribute / Parameter", style="cyan", justify="left")
        wtp_table.add_column("WTP ($)", justify="right")

        if is_flavor_gamma:
            # Flavor gamma indices in parameters vector
            beta_ber, beta_pl = results["params"][1], results["params"][2]
            g_oth, g_ber, g_pl = (
                results["params"][3],
                results["params"][4],
                results["params"][5],
            )

            wtp_table.add_row(
                "WTP: Berry Preference (vs Other)", f"${-beta_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Plain Preference (vs Other)", f"${-beta_pl / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Other)", f"${-g_oth / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Berry)", f"${-g_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Plain)", f"${-g_pl / alpha_val:.2f}"
            )
        else:
            beta_ber, beta_pl, gamma_val = (
                results["params"][1],
                results["params"][2],
                results["params"][3],
            )
            wtp_table.add_row(
                "WTP: Berry Preference (vs Other)", f"${-beta_ber / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Plain Preference (vs Other)", f"${-beta_pl / alpha_val:.2f}"
            )
            wtp_table.add_row(
                "WTP: Satiation Disutility (Gamma)", f"${-gamma_val / alpha_val:.2f}"
            )

        console.print(wtp_table)
    console.print("\n")


def main():
    hh_packed_data, vec_data = load_and_preprocess(inside_only=False)

    console.print(
        "\n[bold yellow]=====================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow] STEP 2: PRODUCT-SPECIFIC GAMMAS & WTP ESTIMATION    [/bold yellow]"
    )
    console.print(
        "[bold yellow]=====================================================[/bold yellow]"
    )

    # 1. Fixed-Coefficient Flavor Gamma Model
    console.print("\n--- Estimating Standard Flavor-Specific Satiation Logit ---")
    res_flavor_gamma = estimate_flavor_gamma_model(vec_data)
    p_names = [
        "Constant",
        "beta_ber",
        "beta_pl",
        "gamma_other",
        "gamma_berry",
        "gamma_plain",
        "Price",
        "Control Func.",
    ]
    display_results_with_wtp(
        res_flavor_gamma,
        title="STANDARD FLAVOR SATIATION LOGIT",
        param_names=p_names,
        is_flavor_gamma=True,
    )

    # 2. Random-Coefficient Flavor Gamma Model
    console.print(
        "\n--- Estimating Random-Coefficient Flavor-Specific Satiation Logit ---"
    )
    res_rc_flavor_gamma = estimate_het_flavor_gamma_model(vec_data, draws=GAMMA_DRAWS)
    rc_p_names = [
        "Constant",
        "Mean beta_berry",
        "Mean beta_plain",
        "Mean gamma_other",
        "Mean gamma_berry",
        "Mean gamma_plain",
        "SD beta_berry",
        "SD beta_plain",
        "SD gamma_other",
        "SD gamma_berry",
        "SD gamma_plain",
        "Price",
        "Control Func.",
    ]
    display_results_with_wtp(
        res_rc_flavor_gamma,
        title="RANDOM COEFFICIENT FLAVOR SATIATION LOGIT",
        param_names=rc_p_names,
        is_flavor_gamma=True,
    )


if __name__ == "__main__":
    main()
