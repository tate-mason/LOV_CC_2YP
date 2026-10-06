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
STATIC_DRAWS = np.random.default_rng(306).standard_normal((50, 4))


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


def load_and_preprocess(weekly_capacity=14):
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

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((4, 2), dtype=np.float64)

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

            effective_capacity = max(weekly_capacity, inside_units + 1)
            outside_count = effective_capacity - inside_units

            full_counts = np.append(inside_units, outside_count)
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

    # Pre-pack flat contiguous arrays
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


def total_objective(params, vec_data):
    const, beta_ber, beta_pl, alpha, sigma = params
    beta_vec = np.array([beta_ber, beta_pl])

    prices = vec_data["prices"]
    resids = vec_data["resids"]
    choices = vec_data["choices"]

    u_inside = const + beta_vec + alpha * prices + sigma * resids
    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_model(vec_data):
    x0 = np.zeros(5)
    bounds = [(None, None)] * 3 + [(None, 0.0), (None, None)]

    res = minimize(
        total_objective,
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
            x2[i] -= eps
            x2[j] -= eps
            x3[i] += eps
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


def display_results(results):
    table = Table(
        title="SIMPLE SATIATION SPECIFICATION RESULTS",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = [
        "Constant",
        "beta_ber",
        "beta_pl",
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


def main():
    hh_packed_data, vec_data = load_and_preprocess()

    console.print("\n--- Estimating Standard Satiation Logit ---")
    results = estimate_model(vec_data)
    display_results(results)


if __name__ == "__main__":
    main()
