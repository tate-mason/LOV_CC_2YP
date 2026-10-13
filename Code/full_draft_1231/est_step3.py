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
import statsmodels.formula.api as smf

install()
console = Console()

MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Output/"
PLOT_OUTPUT_DIR = "../Output/Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

CALIBRATED_LAMBDA = 0.7
BASE_DELTA_DISCOUNT = 0.9
TOP_N_FLAVORS = 8  # Top N individual flavors retained as unbinned options

INCOME_CODE_MAP = {
    3: 2500,  # Under $5,000
    4: 6500,  # $5,000 - $7,999
    6: 9000,  # $8,000 - $9,999
    8: 11000,  # $10,000 - $11,999
    10: 13500,  # $12,000 - $14,999
    11: 17500,  # $15,000 - $19,999
    13: 22500,  # $20,000 - $24,999
    15: 27500,  # $25,000 - $29,999
    16: 32500,  # $30,000 - $34,999
    17: 37500,  # $35,000 - $39,999
    18: 42500,  # $40,000 - $44,999
    19: 47500,  # $45,000 - $49,999
    21: 55000,  # $50,000 - $59,999
    23: 65000,  # $60,000 - $69,999
    26: 85000,  # $70,000 - $99,999
    27: 100000,  # $100,000+
}


def compute_unbinned_satiation_state(
    weekly_purchases,
    n_categories,
    lambda_mem=CALIBRATED_LAMBDA,
    delta_discount=BASE_DELTA_DISCOUNT,
):
    sorted_df = weekly_purchases.sort_values(["household_code", "week_end"])
    inventory_records = []

    for hh_id, hh_group in sorted_df.groupby("household_code"):
        weeks = pd.to_datetime(hh_group["week_end"].unique())
        weeks = np.sort(weeks)

        C_history = {}
        prev_X = np.zeros(n_categories)

        for t_idx, week in enumerate(weeks):
            C_current = np.zeros(n_categories)
            memory_term = np.zeros(n_categories)

            if t_idx > 0:
                for s in range(t_idx):
                    weeks_elapsed = pd.Timedelta(week - weeks[s]).days / 7.0
                    discount = delta_discount**weeks_elapsed
                    memory_term += discount * C_history[s]

            for j in range(n_categories):
                # Pure Own-Flavor Stock per unbinned flavor
                C_current[j] = (lambda_mem * memory_term[j]) + (
                    (1.0 - lambda_mem) * prev_X[j]
                )

            record = {
                "household_code": hh_id,
                "week_end": pd.Timestamp(week).strftime("%Y-%m-%d"),
            }
            for j in range(n_categories):
                record[f"C_sat_{j}"] = C_current[j]
            inventory_records.append(record)

            C_history[t_idx] = C_current.copy()

            sub = hh_group[hh_group["week_end"] == week]
            curr_X = np.zeros(n_categories)
            for row in sub.itertuples():
                if hasattr(row, "choice_idx") and row.choice_idx < n_categories:
                    curr_X[int(row.choice_idx)] = row.units_bought
            total_inside_units = curr_X.sum()
            if total_inside_units > 0:
                curr_X = curr_X / total_inside_units
            prev_X = curr_X

    df_sat = pd.DataFrame(inventory_records)

    # Standardize column variance
    for j in range(n_categories):
        c_col = f"C_sat_{j}"
        std_val = df_sat[c_col].std()
        if std_val > 0:
            df_sat[c_col] = df_sat[c_col] / std_val

    return df_sat


def load_and_preprocess_unbinned(weekly_capacity=14, top_n=TOP_N_FLAVORS):
    schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

    exprs = [
        pl.col("quantity").cast(pl.Int64),
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

    # 1. Income Mapping & Distribution
    hh_unique_raw = (
        merged_df.groupby("household_code")["household_income"].first().dropna()
    )
    tot_hh = len(hh_unique_raw)

    merged_df["household_income_dollars"] = merged_df["household_income"].map(
        INCOME_CODE_MAP
    )
    median_dollars = merged_df["household_income_dollars"].median()
    merged_df["household_income_dollars"] = merged_df[
        "household_income_dollars"
    ].fillna(median_dollars)

    # 2. Extract Unbinned Raw Flavors
    descr_col = "flavor_descr" if "flavor_descr" in merged_df.columns else "flavor_cd"
    merged_df["flavor_raw"] = (
        merged_df[descr_col].fillna("UNSPECIFIED").astype(str).str.upper().str.strip()
    )

    # Identify Top N Flavors
    flavor_counts = merged_df["flavor_raw"].value_counts()
    top_flavors = list(flavor_counts.head(top_n).index)

    def assign_unbinned_category(f_raw):
        if f_raw in top_flavors:
            return f_raw
        return "OTHER_INSIDE"

    merged_df["unbinned_cat"] = merged_df["flavor_raw"].apply(assign_unbinned_category)

    unbinned_categories = top_flavors + ["OTHER_INSIDE"]
    cat_map = {c: i for i, c in enumerate(unbinned_categories)}
    n_inside = len(unbinned_categories)

    table_flavors = Table(
        title=f"TOP {top_n} UNBINNED FLAVORS (+ OTHER INSIDE)",
        show_header=True,
        header_style="bold magenta",
    )
    table_flavors.add_column("Choice Index", style="cyan", justify="center")
    table_flavors.add_column(
        "Unbinned Flavor String", style="bold white", justify="left"
    )
    table_flavors.add_column("Observation Count", style="yellow", justify="right")

    for idx, f_name in enumerate(unbinned_categories):
        cnt = (merged_df["unbinned_cat"] == f_name).sum()
        table_flavors.add_row(str(idx), f_name, f"{cnt:,}")

    console.print(table_flavors)

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

    # Standardize Income
    hh_stats = (
        merged_df.groupby("household_code")
        .agg(income=("household_income_dollars", "first"))
        .reset_index()
    )
    hh_stats["income_std"] = (hh_stats["income"] - hh_stats["income"].mean()) / (
        hh_stats["income"].std() + 1e-6
    )
    hh_dict = hh_stats.set_index("household_code")["income_std"].to_dict()

    weekly_purchases = (
        merged_df.groupby(
            ["household_code", "store_code_uc", "week_end", "unbinned_cat"]
        )
        .agg(units_bought=("quantity", "sum"))
        .reset_index()
    )
    weekly_purchases["choice_idx"] = weekly_purchases["unbinned_cat"].map(cat_map)

    cat_choice_sets = (
        merged_df.groupby(["store_code_uc", "week_end", "unbinned_cat"])
        .agg(price=("price", "mean"), iv_res=("iv_res", "mean"))
        .reset_index()
    )

    cat_choice_sets["price"] = cat_choice_sets.groupby(
        ["store_code_uc", "unbinned_cat"]
    )["price"].transform(lambda x: x.fillna(x.mean()))
    cat_choice_sets["price"] = cat_choice_sets.groupby("unbinned_cat")[
        "price"
    ].transform(lambda x: x.fillna(x.mean()))
    cat_choice_sets["iv_res"] = cat_choice_sets["iv_res"].fillna(0.0)

    overall_cat_prices = merged_df.groupby("unbinned_cat")["price"].mean().to_dict()

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((n_inside, 2), dtype=np.float64)

        for cat_name, idx in cat_map.items():
            if cat_name in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat_name]

        for row in group.itertuples():
            if row.unbinned_cat in cat_map:
                idx = cat_map[row.unbinned_cat]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    console.print(
        "\n[bold green]Computing unbinned own-flavor satiation states...[/bold green]"
    )
    df_sat = compute_unbinned_satiation_state(
        weekly_purchases,
        n_categories=n_inside,
        lambda_mem=CALIBRATED_LAMBDA,
        delta_discount=0.9,
    )

    hh_weeks = weekly_purchases[
        ["household_code", "store_code_uc", "week_end"]
    ].drop_duplicates()
    hh_weeks["week_end_str"] = pd.to_datetime(hh_weeks["week_end"]).dt.strftime(
        "%Y-%m-%d"
    )
    df_sat["week_end_str"] = pd.to_datetime(df_sat["week_end"]).dt.strftime("%Y-%m-%d")

    hh_weeks = hh_weeks.merge(
        df_sat,
        left_on=["household_code", "week_end_str"],
        right_on=["household_code", "week_end_str"],
        how="left",
    ).fillna(0.0)

    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(n_inside, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < n_inside:
                c_arr[int(row.choice_idx)] = row.units_bought
        counts_dict[(hh_id, week)] = c_arr

    all_prices, all_resids, all_choices, all_c_states, all_inc = [], [], [], [], []

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = (
            row_dict["week_end_x"] if "week_end_x" in row_dict else row_dict["week_end"]
        )
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        inc_val = hh_dict.get(hh_id, 0.0)

        inside_counts = counts_dict.get(
            (hh_id, week), np.zeros(n_inside, dtype=np.int64)
        )
        inside_units = np.sum(inside_counts)

        effective_capacity = max(weekly_capacity, inside_units + 1)
        outside_count = effective_capacity - inside_units

        full_counts = np.append(inside_counts, outside_count)
        c_vec = np.array(
            [row_dict.get(f"C_sat_{j}", 0.0) for j in range(n_inside)],
            dtype=np.float64,
        )

        all_prices.append(choice_set_matrix[(store, week)][:n_inside, 0])
        all_resids.append(choice_set_matrix[(store, week)][:n_inside, 1])
        all_choices.append(full_counts)
        all_c_states.append(c_vec)
        all_inc.append(inc_val)

    return {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "income": np.array(all_inc, dtype=np.float64)[:, None],
        "categories": unbinned_categories,
    }


def total_objective_unbinned(params, vec_data):
    n_inside = vec_data["prices"].shape[1]

    # Base option (j=0) fixed to 0.0 for identification
    const = params[0]
    b0 = params[1:n_inside]  # (n_inside - 1)
    b_inc = params[n_inside : 2 * n_inside - 1]  # (n_inside - 1)

    g0 = params[2 * n_inside - 1]
    g_inc = params[2 * n_inside]

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]  # (N, n_inside)
    resids = vec_data["resids"]  # (N, n_inside)
    choices = vec_data["choices"]  # (N, n_inside + 1)
    c_state = vec_data["c_states"]  # (N, n_inside)
    inc = vec_data["income"]  # (N, 1)

    beta_non_base = b0[None, :] + b_inc[None, :] * inc
    beta_mat = np.hstack([np.zeros((inc.shape[0], 1)), beta_non_base])

    gamma_i = g0 + g_inc * inc

    u_inside = (
        const
        + beta_mat
        + gamma_i * np.log(1.0 + c_state)
        + alpha * prices
        + sigma * resids
    )

    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_unbinned_model(vec_data):
    n_inside = vec_data["prices"].shape[1]
    total_params = 1 + 2 * (n_inside - 1) + 2 + 2

    x0 = np.zeros(total_params)

    # Asymmetric initialization across unbinned flavors to break gradient traps
    for k in range(n_inside - 1):
        x0[1 + k] = 0.1 * (k + 1)

    x0[2 * n_inside - 1] = -0.1  # gamma_0
    x0[-2] = -0.5  # alpha price

    bounds = (
        [(-20.0, 20.0)] * (1 + 2 * (n_inside - 1))
        + [(-10.0, 10.0)] * 2  # gamma_0, gamma_inc
        + [(-30.0, -0.01), (-30.0, 30.0)]  # price + control func
    )

    res = minimize(
        total_objective_unbinned,
        x0=x0,
        args=(vec_data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8, "gtol": 1e-5, "maxiter": 2000},
    )

    n = len(res.x)
    eps = 1e-4
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

            f1 = total_objective_unbinned(x1, vec_data)
            f2 = total_objective_unbinned(x2, vec_data)
            f3 = total_objective_unbinned(x3, vec_data)
            f4 = total_objective_unbinned(x4, vec_data)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    try:
        cov = np.linalg.inv(hessian + np.eye(n) * 1e-6)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)

    se = np.sqrt(np.maximum(1e-8, np.diag(cov)))
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


def display_results(results, categories):
    n_inside = len(categories)

    table = Table(
        title="UNBINNED ALL-FLAVORS LOGIT MODEL ESTIMATION",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = ["Constant"]
    base_cat = categories[0]

    for cat in categories[1:]:
        param_names.append(f"beta_{cat} (vs {base_cat})")
    for cat in categories[1:]:
        param_names.append(f"beta_{cat} x Income")

    param_names.extend(
        [
            "gamma_0 (Global Base Satiation)",
            "gamma x Income",
            "Price (alpha)",
            "Control Func. (sigma)",
        ]
    )

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
        "\n[bold yellow]--- Preprocessing Data for Unbinned All-Flavors Model ---[/bold yellow]"
    )
    vec_data = load_and_preprocess_unbinned(top_n=TOP_N_FLAVORS)

    console.print("\n[bold yellow]--- Estimating Unbinned Model ---[/bold yellow]")
    res_model = estimate_unbinned_model(vec_data)
    display_results(res_model, vec_data["categories"])


if __name__ == "__main__":
    main()
