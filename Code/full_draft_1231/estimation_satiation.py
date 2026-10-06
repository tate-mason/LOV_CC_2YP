# =================================================
# 1. SETUP AND CONFIG
# =================================================

# data loading
import pandas as pd
import polars as pl

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
from rich.table import Table
from rich.traceback import install

install()

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
                pl.col("protein_gram_cd").cast(pl.Int64),
                pl.col("sugar_gram_cd").cast(pl.Int64),
                pl.col("total_carbohydrate_gram_cd").cast(pl.Int64),
                pl.col("total_fat_gram_cd").cast(pl.Int64),
                pl.col("organic_claim_cd").cast(pl.Int64),
            ]
        )
        .filter(pl.col("product_module_code_hms").is_in([3612, 3603]))
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

    # 1. Gather description columns
    descr_cols = [
        c
        for c in merged_df.columns
        if any(k in c.lower() for k in ["descr", "flavor", "brand", "product", "upc"])
    ]

    # 2. Combine text
    merged_df["full_text"] = ""
    for c in descr_cols:
        merged_df["full_text"] += " " + merged_df[c].fillna("").astype(str)
    merged_df["full_text"] = merged_df["full_text"].str.lower()

    # 3. Regex matching
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
    merged_master["protein"] = (
        merged_master["protein_gram"].str.extract(r"(\d+)").astype(int)
    )
    merged_master["sugar"] = (
        merged_master["sugar_gram"].str.extract(r"(\d+)").astype(int)
    )
    merged_master["carbs"] = (
        merged_master["total_carbohydrate_gram"].str.extract(r"(\d+)").astype(int)
    )
    merged_master["fat"] = (
        merged_master["total_fat_gram"].str.extract(r"(\d+)").astype(int)
    )
    merged_master["organic"] = (
        merged_master["organic_claim"].str.extract(r"(\d+)").astype(int)
    )

    # Choice Sets & Weekly Purchases (Inside Categories Only)
    inside_df = merged_master[merged_master["category"] != "outside"].copy()

    weekly_purchases = (
        inside_df.groupby(["household_code", "store_code_uc", "week_end", "category"])
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["category"].map(cat_map)

    # Calculate choice sets excluding 'outside' to prevent NaN prices
    # Aggregate category-level characteristics per store-week
    cat_choice_sets = (
        inside_df.groupby(["store_code_uc", "week_end", "category"])
        .agg(
            price=("price", "mean"),
            iv_res=("iv_res", "mean"),
            protein=("protein", "mean"),
            carbs=("carbs", "mean"),
            fat=("fat", "mean"),
            sugar=("sugar", "mean"),
            organic=("organic", "mean"),
        )
        .reset_index()
    )

    overall_cat_prices = inside_df.groupby("category")["price"].mean().to_dict()

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        # Matrix shape: 4 options (3 inside + 1 outside), 7 features
        # Columns: [price, iv_res, protein, carbs, fat, sugar, organic]
        mat = np.zeros((4, 7), dtype=np.float64)

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                mat[idx] = [
                    row.price,
                    row.iv_res,
                    row.protein,
                    row.carbs,
                    row.fat,
                    row.sugar,
                    row.organic,
                ]

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

    # 1. Pre-aggregate choice counts per household-week into a dictionary
    # Key: (household_code, week_end) -> Value: np.array([count_0, count_1, count_2])
    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(3, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < 3:
                c_arr[row.choice_idx] = row.units_bought
        counts_dict[(hh_id, week)] = c_arr

    # 2. Build hh_packed_data by iterating over hh_weeks groups directly
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

            # Skip if market matrix is missing for this store-week
            if (store, week) not in choice_set_matrix:
                continue

            # Retrieve pre-computed choice counts in O(1) time
            inside_counts = counts_dict.get((hh_id, week), np.zeros(3, dtype=np.int64))
            inside_units = np.sum(inside_counts)

            # Compute outside option count
            effective_capacity = max(weekly_capacity, inside_units + 1)
            outside_count = effective_capacity - inside_units

            full_counts = np.append(inside_counts, outside_count)

            # Retrieve satiation vector C_jt
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

    # Print summary statistics
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

    # Pre-pack flat contiguous arrays for vectorized likelihood evaluation
    all_prices = []
    all_resids = []
    all_characts = []
    all_choices = []
    all_c_states = []
    all_log_inc = []

    for hh_id, hh_data in hh_packed_data.items():
        for m, c, cs in zip(
            hh_data["matrices"], hh_data["choices"], hh_data["c_states"]
        ):
            all_prices.append(m[:3, 0])
            all_resids.append(m[:3, 1])
            all_characts.append(m[:3, 2:])
            all_choices.append(c)
            all_c_states.append(cs)
            all_log_inc.append(hh_data["log_income"])

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "characts": np.array(all_characts, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "log_inc": np.array(all_log_inc, dtype=np.float64),
    }

    return hh_packed_data, vec_data


# =================================================
# 3. CORE ESTIMATION LOGIC (VECTORIZED)
# =================================================


def total_objective(params, vec_data):
    beta_oth, beta_ber, beta_pl, gamma, alpha, sigma, *beta_char = params
    beta_char = np.array(beta_char)
    beta_vec = np.array([beta_oth, beta_ber, beta_pl])

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    characts = vec_data["characts"]
    choices = vec_data["choices"]
    c_states = vec_data["c_states"]

    u_characts = characts @ beta_char

    u_inside = (
        beta_vec + gamma * c_states + alpha * prices + sigma * resids + u_characts
    )
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_model(vec_data):
    x0 = np.zeros(11)
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        *([(None, None)] * 5),
        (None, 0.0),
        (None, None),
    ]

    res = minimize(
        total_objective,
        x0=x0,
        args=(vec_data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8, "gtol": 1e-5},
    )

    # Finite-difference Hessian for reliable standard errors
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

            f1 = total_objective(x1, vec_data)
            f2 = total_objective(x2, vec_data)
            f3 = total_objective(x3, vec_data)
            f4 = total_objective(x4, vec_data)

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


def total_objective_mixed_vec(params, vec_data, static_draws):
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
        *beta_char,
    ) = params

    beta_char = np.array(beta_Char)
    n_draws = static_draws.shape[0]

    beta_draws = np.column_stack(
        [
            mu_b_oth + sd_b_oth * static_draws[:, 0],
            mu_b_berry + sd_b_berry * static_draws[:, 1],
            mu_b_pl + sd_b_pl * static_draws[:, 2],
        ]
    )
    gamma_draws = mu_gamma + sd_gamma * static_draws[:, 3]

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    characts = vec_data["chars"]
    choices = vec_data["choices"]
    c_states = vec_data["c_states"]
    log_inc = vec_data["log_inc"]

    u_characts = characts @ beta_char

    alpha_i = alpha_0 + alpha_inc * log_inc

    u_inside = (
        beta_draws[None, :, :]
        + gamma_draws[None, :, None] * c_states[:, None, :]
        + (alpha_i[:, None] * prices + sigma_cf * resids + u_characts)[:, None, :]
    )

    u_outside = np.zeros((u_inside.shape[0], n_draws, 1))
    u = np.concatenate([u_inside, u_outside], axis=2)

    log_probs = u - logsumexp(u, axis=2, keepdims=True)
    obs_ll_draws = np.sum(log_probs * choices[:, None, :], axis=2)

    obs_ll = logsumexp(obs_ll_draws, axis=1) - np.log(n_draws)
    total_ll = np.sum(obs_ll)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_mixed_model(vec_data, static_draws=STATIC_DRAWS[:30]):
    x0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5,
            0.05,
            0.0,
            0.1,
            0.1,
            0.1,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, 0.0),
        (None, None),
        (None, None),
        (1e-4, None),
        (1e-4, None),
        (1e-4, None),
        (1e-4, None),
    ] + [(None, None)] * 5

    res = minimize(
        total_objective_mixed_vec,
        x0=x0,
        args=(vec_data, static_draws),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-6,
            "gtol": 1e-3,
            "maxiter": 200,
        },
    )

    # Finite-difference Hessian for Mixed Model SEs
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

            f1 = total_objective_mixed_vec(x1, vec_data, static_draws)
            f2 = total_objective_mixed_vec(x2, vec_data, static_draws)
            f3 = total_objective_mixed_vec(x3, vec_data, static_draws)
            f4 = total_objective_mixed_vec(x4, vec_data, static_draws)

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


# =================================================
# 4. RESULTS DISPLAY AND POST-ESTIMATION
# =================================================


def display_results(results):
    table = Table(
        title="SATIATION SPECIFICATION RESULTS",
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


def extract_individual_parameters(results, hh_packed_data, n_draws=100):
    params = results["params"]
    (
        mu_b_oth,
        mu_b_ber,
        mu_b_pl,
        mu_gamma,
        alpha_0,
        alpha_inc,
        sigma_cf,
        sd_b_oth,
        sd_b_ber,
        sd_b_pl,
        sd_gamma,
    ) = params

    sim_draws = np.random.default_rng(123).standard_normal((n_draws, 4))
    draws_b_oth = mu_b_oth + sd_b_oth * sim_draws[:, 0]
    draws_b_ber = mu_b_ber + sd_b_ber * sim_draws[:, 1]
    draws_b_pl = mu_b_pl + sd_b_pl * sim_draws[:, 2]
    draws_gamma = mu_gamma + sd_gamma * sim_draws[:, 3]

    beta_matrix = np.column_stack([draws_b_oth, draws_b_ber, draws_b_pl])
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
        draw_log_probs = np.zeros(n_draws)

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
            draw_log_probs += np.dot(log_probs, counts)

        max_log_p = np.max(draw_log_probs)
        weights = np.exp(draw_log_probs - max_log_p)
        weights /= np.sum(weights)

        hh_posterior_means.append(
            {
                "household_code": hh_id,
                "beta_oth": np.sum(weights * draws_b_oth),
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


# =================================================
# 5. ENTRY POINT
# =================================================


def main():
    hh_packed_data, vec_data = load_and_preprocess()

    console.print("\n--- Estimating Standard Satiation Logit ---")
    results = estimate_model(vec_data)
    display_results(results)

    console.print("\n--- Estimating Mixed Satiation Logit ---")
    mixed_results = estimate_mixed_model(vec_data, static_draws=STATIC_DRAWS[:30])
    display_mixed_results(mixed_results)

    df_types = extract_individual_parameters(mixed_results, hh_packed_data, n_draws=100)
    display_type_distribution(df_types)

    plot_path = OUT_PATH + "type_distribution_satiation.pdf"
    plot_type_distribution(df_types, save_path=plot_path)

    plot_path_lov_vs = OUT_PATH + "satiation_vs_oo.pdf"
    plot_lov_vs_outside_option(df_types, save_path=plot_path_lov_vs)


if __name__ == "__main__":
    main()
