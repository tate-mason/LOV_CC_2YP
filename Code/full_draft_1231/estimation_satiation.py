# =================================================
# 1. SETUP AND CONFIG
# =================================================

# data loading
import polars as pl
import pandas as pd

# numerical and statistical analysis
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp
import statsmodels.formula.api as smf

# graphing
import matplotlib.pyplot as plt
import seaborn as sns

# output
from rich.console import Console
from rich.traceback import install

install()
from rich.table import Table

console = Console()

MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Output/"
CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}
rng = np.random.default_rng(219)
STATIC_DRAWS = np.random.default_rng(306).standard_normal((50, 4))


# =================================================
# 2. DATA / INPUT HANDLING
# =================================================


def compute_satiation_state(
    weekly_purchases,
    n_categories=3,
    lambda_mem=0.7,
    delta_discount=0.9,
    rho_weights=None,
):
    """
    Computes dynamic satiation state:
    C_jt = lambda * sum_s (delta^(t-s) * C_js) + (1 - lambda) * sum_{k != j} (rho_k * X_{k, t-1})
    """
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

            # Component 1: Discounted memory of prior satiation history
            memory_term = np.zeros(n_categories)
            if t_idx > 0:
                for s in range(t_idx):
                    discount = delta_discount ** (t_idx - s)
                    memory_term += discount * C_history[s]

            # Component 2: Cross-flavor purchase impact from prior period t-1
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

            # Record choices made in current week for next period t-1
            sub = hh_group[hh_group["week_end"] == week]
            curr_X = np.zeros(n_categories)
            for row in sub.itertuples():
                if row.choice_idx < n_categories:
                    curr_X[row.choice_idx] = row.units_bought
            prev_X = curr_X

    return pd.DataFrame(inventory_records)


def load_and_preprocess(weekly_capacity=28):
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
    # 1. Gather all potential description columns present in dataset
    descr_cols = [
        c
        for c in merged_df.columns
        if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"])
    ]

    # 2. Combine into a single lowercase text column
    merged_df["full_text"] = ""
    for c in descr_cols:
        merged_df["full_text"] += " " + merged_df[c].fillna("").astype(str)
    merged_df["full_text"] = merged_df["full_text"].str.lower()

    # 3. Apply regex across combined text
    berry_regex = r"berry|straw|blue|rasp|black|cran|cherry|wildberry"
    plain_regex = r"plain|unflavored"

    is_plain = merged_df["full_text"].str.contains(plain_regex, na=False)
    is_berry = merged_df["full_text"].str.contains(berry_regex, na=False) & (~is_plain)

    # 4. Assign Flavor Categories (2: Plain, 1: Berry, 0: Other)
    merged_master["flavor"] = np.select([is_plain, is_berry], [2, 1], default=0)

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

    # Choice Sets & Weekly Purchases
    weekly_purchases = (
        merged_master[merged_master["category"] != "outside"]
        .groupby(["household_code", "store_code_uc", "week_end", "category"])
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["category"].map(cat_map)

    # Filter to inside categories only before calculating choice set prices
    cat_choice_sets = (
        merged_master[merged_master["category"] != "outside"]
        .groupby(["store_code_uc", "week_end", "category"])
        .agg(price=("price", "mean"), iv_res=("iv_res", "mean"))
        .reset_index()
    )

    overall_cat_prices = (
        merged_master[merged_master["category"] != "outside"]
        .groupby("category")["price"]
        .mean()
        .to_dict()
    )

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((4, 2))

        # Calculate mean price across observed inside categories only
        valid_prices = group["price"].dropna()
        mean_store_prices = valid_prices.mean() if len(valid_prices) > 0 else 1.50

        # Fill default prices for all 3 inside options
        for c_name, c_idx in cat_map.items():
            if c_idx < 3:
                mat[c_idx, 0] = overall_cat_prices.get(c_name, mean_store_prices)

        # Overwrite with store-week specific observed prices & residuals
        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                p_val = (
                    row.price
                    if pd.notna(row.price)
                    else overall_cat_prices.get(row.category, mean_store_prices)
                )
                r_val = row.iv_res if pd.notna(row.iv_res) else 0.0
                mat[idx] = [p_val, r_val]

        choice_set_matrix[(store, week)] = mat

    # Compute dynamic Satiation States C_jt
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

    # Household Income Map
    hh_income_map = (
        merged_df.drop_duplicates(subset=["household_code"])
        .set_index("household_code")["household_income"]
        .to_dict()
    )

    hh_packed_data = {}
    for hh_id, group in hh_weeks.groupby("household_code"):
        stores = group["store_code_uc"].to_numpy()
        weeks = group["week_end"].to_numpy()
        c_0_vals = group["C_sat_0"].to_numpy(dtype=np.float64)
        c_1_vals = group["C_sat_1"].to_numpy(dtype=np.float64)
        c_2_vals = group["C_sat_2"].to_numpy(dtype=np.float64)

        raw_inc = hh_income_map.get(hh_id, 1)
        log_inc = np.log(max(float(raw_inc) if pd.notna(raw_inc) else 1.0, 1.0))

        valid_mask = np.array(
            [(s, w) in choice_set_matrix for s, w in zip(stores, weeks)]
        )
        if not valid_mask.any():
            continue

        matrices_list = []
        choice_counts_list = []
        c_states_list = []

        for store, week, c0, c1, c2 in zip(
            stores[valid_mask],
            weeks[valid_mask],
            c_0_vals[valid_mask],
            c_1_vals[valid_mask],
            c_2_vals[valid_mask],
        ):
            sub = weekly_purchases[
                (weekly_purchases["household_code"] == hh_id)
                & (weekly_purchases["week_end"] == week)
            ]
            counts = np.zeros(4, dtype=np.int64)
            for row in sub.itertuples():
                counts[row.choice_idx] += row.units_bought

            inside_units = np.sum(counts[:3])
            effective_capacity = max(weekly_capacity, inside_units + 1)
            outside_count = effective_capacity - inside_units
            counts[3] = outside_count

            matrices_list.append(choice_set_matrix[(store, week)])
            choice_counts_list.append(counts)
            c_states_list.append(np.array([c0, c1, c2]))

        hh_packed_data[hh_id] = {
            "matrices": matrices_list,
            "choices": choice_counts_list,
            "c_states": c_states_list,
            "log_income": log_inc,
        }

    # Summary diagnostics executed AFTER dataset is fully packed
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

    # -------------------------------------------------------------
    # ADDED DIAGNOSTICS: Check choice counts and parameter variation
    # -------------------------------------------------------------
    all_counts = np.sum(
        [counts for hh in hh_packed_data.values() for counts in hh["choices"]],
        axis=0,
    )
    console.print("\n[bold yellow]--- CHOICE CATEGORY TOTALS ---[/bold yellow]")
    console.print(
        f"Other: [cyan]{all_counts[0]}[/cyan] | "
        f"Berry: [cyan]{all_counts[1]}[/cyan] | "
        f"Plain: [cyan]{all_counts[2]}[/cyan] | "
        f"Outside: [cyan]{all_counts[3]}[/cyan]"
    )

    prices = [
        X[i, 0]
        for hh in hh_packed_data.values()
        for X in hh["matrices"]
        for i in range(3)
    ]
    resids = [
        X[i, 1]
        for hh in hh_packed_data.values()
        for X in hh["matrices"]
        for i in range(3)
    ]
    console.print(
        f"[bold yellow]Price Range:[/bold yellow] {np.min(prices):.2f} to {np.max(prices):.2f} (Std: {np.std(prices):.2f})"
    )
    console.print(
        f"[bold yellow]Resid Range:[/bold yellow] {np.min(resids):.2f} to {np.max(resids):.2f} (Std: {np.std(resids):.2f})\n"
    )

    return hh_packed_data


# =================================================
# CORE LOGIC / PROCESSING
# =================================================


def total_objective(params, hh_packed_data):
    beta_oth, beta_ber, beta_pl, gamma, alpha, sigma = params

    beta_vec = np.array([beta_oth, beta_ber, beta_pl])

    total_ll = 0.0

    for hh_data in hh_packed_data.values():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        c_states = hh_data["c_states"]

        for X_mat, counts, C_jt in zip(matrices, choices, c_states):
            prices = X_mat[:3, 0]
            resids = X_mat[:3, 1]

            u = np.zeros(4)
            u[:3] = beta_vec + gamma * C_jt + alpha * prices + sigma * resids
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

    # Check gradient magnitude at optimum
    grad = res.jac
    grad_norm = np.linalg.norm(grad)
    print(f"Gradient Norm at convergence: {grad_norm:.6f}")

    # Check Hessian condition number
    if hasattr(res, "hess_inv"):
        if hasattr(res.hess_inv, "todense"):
            hess_inv = res.hess_inv.todense()
        else:
            hess_inv = res.hess_inv
        cond = np.linalg.cond(hess_inv)
        print(f"Hessian Condition Number: {cond:.2e}")
    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def total_objective_mixed(params, hh_packed_data, static_draws):
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

    n_draws = static_draws.shape[0]

    draws_b_oth = mu_b_oth + sd_b_oth * static_draws[:, 0]
    draws_b_ber = mu_b_berry + sd_b_berry * static_draws[:, 1]
    draws_b_pl = mu_b_pl + sd_b_pl * static_draws[:, 2]
    draws_gamma = mu_gamma + sd_gamma * static_draws[:, 3]

    beta_matrix = np.column_stack([draws_b_oth, draws_b_ber, draws_b_pl])
    total_ll = 0.0

    for hh_data in hh_packed_data.values():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        c_states = hh_data["c_states"]
        log_inc = hh_data["log_income"]

        alpha_i = alpha_0 + alpha_inc * log_inc
        log_draw_probs = np.zeros(n_draws)

        for X_mat, counts, C_jt in zip(matrices, choices, c_states):
            prices = X_mat[:3, 0]
            resids = X_mat[:3, 1]

            u_inside = (
                beta_matrix
                + np.outer(draws_gamma, C_jt)
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


def estimate_mixed_model(hh_packed_data, static_draws=STATIC_DRAWS):
    x0 = np.array([0.0, 0.0, 0.0, 0.0, -1.5, 0.05, 0.0, 0.1, 0.1, 0.1, 0.1])

    bounds = [
        (None, None),  # mu_b_oth
        (None, None),  # mu_b_ber
        (None, None),  # mu_b_pl
        (None, None),  # mu_gamma
        (None, 0.0),  # alpha_0 (Must be non-positive)
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
        args=(hh_packed_data, static_draws),
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

    ## Check gradient magnitude at optimum
    # grad = res.jac
    # grad_norm = np.linalg.norm(grad)
    # print(f"Gradient Norm at convergence: {grad_norm:.6f}")

    ## Check Hessian condition number
    # if hasattr(res, "hess_inv"):
    #    if hasattr(res.hess_inv, "todense"):
    #        hess_inv = res.hess_inv.todense()
    #    else:
    #        hess_inv = res.hess_inv
    #    cond = np.linalg.cond(hess_inv)
    #    print(f"Hessian Condition Number: {cond:.2e}")
    return {
        "params": res.x,
        "se": se,
        "z_stat": z,
        "p_val": p,
        "success": res.success,
        "fun": res.fun,
    }


def display_results(results):
    table = Table(
        title="MCALLISTER & LATTIN SATIATION RESULTS",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = [
        "beta_oth",
        "beta_ber",
        "beta_pl",
        "gamma (Satiation)",
        "Price",
        "Control Func.",
    ]

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


def display_mixed_results(results):
    table = Table(
        title="MIXED LOGIT SATIATION RESULTS",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = [
        "μ_beta_oth",
        "μ_beta_ber",
        "μ_beta_pl",
        "μ_gamma (Satiation)",
        "α_0 (Base Price)",
        "α_inc (Price x Inc)",
        "σ_Control_Func",
        "σ_beta_oth (SD)",
        "σ_beta_ber (SD)",
        "σ_beta_pl (SD)",
        "σ_gamma (SD)",
    ]

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


def extract_individual_parameters(results, hh_packed_data, n_draws=500):
    params = results["params"]
    (
        const,
        mu_b_ber,
        mu_b_pl,
        mu_gamma,
        alpha_0,
        alpha_inc,
        sigma_cf,
        sd_b_ber,
        sd_b_pl,
        sd_gamma,
    ) = params

    d_berry = np.array([0.0, 1.0, 0.0])
    d_plain = np.array([0.0, 0.0, 1.0])

    rng_sim = np.random.default_rng(123)
    hh_posterior_means = []

    for hh_id, hh_data in hh_packed_data.items():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        c_states = hh_data["c_states"]
        log_inc = hh_data["log_income"]

        total_counts = np.sum(choices, axis=0)
        total_units = np.sum(total_counts)
        outside_units = total_counts[3]
        outside_share = outside_units / total_units if total_units > 0 else 0.0

        alpha_i = alpha_0 + alpha_inc * log_inc

        draws_b_ber = mu_b_ber + sd_b_ber * rng_sim.standard_normal(n_draws)
        draws_b_pl = mu_b_pl + sd_b_pl * rng_sim.standard_normal(n_draws)
        draws_gamma = mu_gamma + sd_gamma * rng_sim.standard_normal(n_draws)

        draw_probabilities = np.ones(n_draws)

        for X_mat, counts, C_jt in zip(matrices, choices, c_states):
            prices = X_mat[:3, 0]
            resids = X_mat[:3, 1]

            u_inside = (
                const
                + np.outer(draws_b_ber, d_berry)
                + np.outer(draws_b_pl, d_plain)
                + np.outer(draws_gamma, C_jt)
                + alpha_i * prices
                + sigma_cf * resids
            )
            u = np.hstack([u_inside, np.zeros((n_draws, 1))])

            log_probs = u - logsumexp(u, axis=1, keepdims=True)
            week_log_ll = np.dot(log_probs, counts)
            draw_probabilities *= np.exp(week_log_ll)

        total_prob = np.sum(draw_probabilities)
        if total_prob > 0:
            weights = draw_probabilities / total_prob
        else:
            weights = np.ones(n_draws) / n_draws

        hh_posterior_means.append(
            {
                "household_code": hh_id,
                "beta_berry": np.sum(weights * draws_b_ber),
                "beta_plain": np.sum(weights * draws_b_pl),
                "gamma_satiation": np.sum(weights * draws_gamma),
                "alpha_price": alpha_i,
                "wtp_berry": -1 * (np.sum(weights * draws_b_ber) / alpha_i),
                "wtp_plain": -1 * (np.sum(weights * draws_b_pl) / alpha_i),
                "outside_units": outside_units,
                "outside_share": outside_share,
            }
        )

    return pd.DataFrame(hh_posterior_means)


def display_type_distribution(df_types):
    table = Table(
        title="EMPIRICAL DISTRIBUTION OF HOUSEHOLD TYPES (POSTERIOR MEANS)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter / Type", style="cyan", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("10th Pctl", justify="right")
    table.add_column("50th (Median)", justify="right")
    table.add_column("90th Pctl", justify="right")
    table.add_column("Std Dev", justify="right")

    cols_to_summarize = [
        ("β_other (Other Preference)", "beta_oth"),
        ("β_berry (Berry Preference)", "beta_berry"),
        ("β_plain (Plain Preference)", "beta_plain"),
        ("γ (Satiation / Habit)", "gamma_satiation"),
        ("α (Price Sensitivity)", "alpha_price"),
        ("WTP Berry ($)", "wtp_berry"),
        ("WTP Plain ($)", "wtp_plain"),
    ]

    for label, col in cols_to_summarize:
        vals = df_types[col].to_numpy()
        table.add_row(
            label,
            f"{np.mean(vals):.4f}",
            f"{np.percentile(vals, 10):.4f}",
            f"{np.median(vals):.4f}",
            f"{np.percentile(vals, 90):.4f}",
            f"{np.std(vals):.4f}",
        )

    console.print(table)


def plot_type_distribution(df_types, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Density Distribution of Household Types (Satiation Model)",
        fontsize=16,
        fontweight="bold",
    )

    plots_config = [
        ("beta_berry", "Berry Preference", "skyblue", axes[0, 0]),
        ("beta_plain", "Plain Preference", "salmon", axes[0, 1]),
        ("gamma_satiation", "Satiation Parameter γ", "mediumpurple", axes[1, 0]),
        ("alpha_price", "Price Sensitivity α", "gold", axes[1, 1]),
    ]

    for col, title, color, ax in plots_config:
        sns.kdeplot(
            df_types[col],
            ax=ax,
            color=color,
            fill=True,
            alpha=0.3,
            linewidth=2.5,
            bw_adjust=0.8,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")

        mean_val = df_types[col].mean()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        console.print(f"[bold green]Saved Plot to PDF:[/bold green] {save_path}")


def plot_lov_vs_outside_option(df_types, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Outside Option Consumption vs Satiation Parameter",
        fontsize=16,
        fontweight="bold",
    )
    sns.kdeplot(
        df_types["outside_share"],
        ax=axes[0],
        color="slategray",
        fill=True,
        alpha=0.4,
        linewidth=2,
        label="Outside Option Share",
    )
    ax0_twin = axes[0].twinx()
    sns.kdeplot(
        df_types["gamma_satiation"],
        ax=ax0_twin,
        color="mediumpurple",
        fill=True,
        alpha=0.3,
        linewidth=2,
        label="gamma",
    )
    axes[0].set_title("Marginal Density Distributions", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density (Outside Share)", color="slategray")
    ax0_twin.set_ylabel("Density (Gamma)", color="mediumpurple")
    ax0_twin.grid(False)

    sns.regplot(
        data=df_types,
        x="gamma_satiation",
        y="outside_share",
        ax=axes[1],
        color="mediumpurple",
        scatter_kws={"alpha": 0.5, "s": 30, "color": "darkslateblue"},
        line_kws={"color": "crimson", "linewidth": 2},
    )
    corr = df_types["gamma_satiation"].corr(df_types["outside_share"])
    axes[1].set_title(
        f"Satiation Parameter vs Outside Choice Share (r={corr:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Gamma (Satiation)")
    axes[1].set_ylabel("HH Outside Option Share")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")


def main():
    hh_packed_data = load_and_preprocess()
    console.print("\n--- Estimating Standard Satiation Logit ---")
    results = estimate_model(hh_packed_data)
    display_results(results)

    console.print("\n--- Estimating Mixed Satiation Logit ---")
    mixed_results = estimate_mixed_model(hh_packed_data, static_draws=STATIC_DRAWS)
    display_mixed_results(mixed_results)

    df_types = extract_individual_parameters(mixed_results, hh_packed_data, n_draws=50)
    display_type_distribution(df_types)

    plot_path = OUT_PATH + "type_distribution_satiation.pdf"
    plot_type_distribution(df_types, save_path=plot_path)

    plot_path_lov_vs = OUT_PATH + "satiation_vs_oo.pdf"
    plot_lov_vs_outside_option(df_types, save_path=plot_path_lov_vs)


if __name__ == "__main__":
    main()
