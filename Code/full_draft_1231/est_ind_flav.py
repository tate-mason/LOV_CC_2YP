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

TOP_K_FLAVORS = 10  # Top K individual flavors to estimate alongside Plain
CALIBRATED_LAMBDA = 0.7  # Memory weight


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
        merged_df["flavor_descr"].fillna("").astype(str).str.lower().str.strip()
    )

    plain_regex = r"plain|unflavored"
    is_plain = merged_df["flavor_str"].str.contains(plain_regex, na=False)

    # Filter non-plain flavors and find the top K individual flavor strings
    non_plain_df = merged_df[~is_plain]
    top_flavors = non_plain_df["flavor_str"].value_counts().head(top_k).index.tolist()

    # Categories: plain (index 0 - BASE), top individual flavors, and outside option
    categories = ["plain"] + top_flavors + ["outside"]
    cat_map = {c: i for i, c in enumerate(categories)}
    n_inside = len(categories) - 1

    console.print(
        f"[bold yellow]Identified Categories (Base = Plain):[/bold yellow] {categories}"
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

    # Satiation State Calculation across individual unbinned flavors
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
                    discount = 0.9 ** (t_idx - s)
                    memory_term += discount * C_history[s]
            for j in range(n_inside):
                cross_term = np.sum(prev_X) - prev_X[j]
                C_current[j] = (CALIBRATED_LAMBDA * memory_term[j]) + (
                    (1.0 - CALIBRATED_LAMBDA) * cross_term
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

    df_sat = pd.DataFrame(inventory_records)
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

    all_prices, all_resids, all_choices, all_c_states = [], [], [], []

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = row_dict["week_end"]
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

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

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "top_flavors": top_flavors,
        "n_inside": n_inside,
    }

    return vec_data


def total_objective_unbinned_gamma(params, vec_data):
    n_inside = vec_data["n_inside"]
    # params: [const, beta_1...beta_{K}, gamma_0 (plain), gamma_1...gamma_{K}, alpha, sigma]
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


def estimate_unbinned_flavor_gamma_model(vec_data):
    n_inside = vec_data["n_inside"]
    # Initial parameters
    x0 = np.zeros(2 * n_inside + 1)
    x0[-2] = -0.1  # alpha price
    bounds = [(None, None)] * (2 * n_inside - 1) + [(None, 0.0), (None, None)]

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


def display_unbinned_results(results, vec_data):
    top_flavors = vec_data["top_flavors"]
    n_inside = vec_data["n_inside"]

    table = Table(
        title="UNBINNED FLAVOR ESTIMATION (BASE FLAVOR = PLAIN)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter / Flavor", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = ["Constant"]
    param_names += [f"beta_{flv} (vs Plain)" for flv in top_flavors]
    param_names += ["gamma_plain"]
    param_names += [f"gamma_{flv}" for flv in top_flavors]
    param_names += ["Price (alpha)", "Control Func. (sigma)"]

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
    console.print(f"[bold]Final LL Objective:[/bold] {results['fun']:.4f}\n")


def main():
    console.print(
        "\n[bold yellow]--- Running Unbinned Individual Flavor Satiation Model ---[/bold yellow]\n"
    )
    vec_data = load_and_preprocess_unbinned()
    results = estimate_unbinned_flavor_gamma_model(vec_data)
    display_unbinned_results(results, vec_data)


if __name__ == "__main__":
    main()
