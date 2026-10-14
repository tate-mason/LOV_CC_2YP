import os
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import scipy as sp
from scipy.optimize import minimize
import statsmodels.formula.api as smf

install()
console = Console()

MERGED_PATH = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
OUT_PATH = "/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Output/"

CATEGORIES = ["other", "berry", "plain"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}

BASE_DELTA_DISCOUNT = 0.85
LAMBDA_OWN_WEIGHT = 0.70

INCOME_CODE_MAP = {
    3: 7500,
    4: 9000,
    6: 11000,
    8: 13500,
    10: 17500,
    11: 22500,
    13: 27500,
    15: 32500,
    16: 37500,
    17: 42500,
    18: 47500,
    19: 55000,
    21: 65000,
    23: 85000,
    26: 112500,
    27: 137500,
    28: 175000,
}


def compute_lattin_intertrip_stock(
    weekly_purchases,
    n_categories=3,
    delta_discount=BASE_DELTA_DISCOUNT,
    lambda_own=LAMBDA_OWN_WEIGHT,
):
    sorted_df = weekly_purchases.sort_values(["household_code", "week_end"])
    purchase_records = []

    for hh_id, hh_group in sorted_df.groupby("household_code"):
        weeks = pd.to_datetime(hh_group["week_end"].unique())
        weeks = np.sort(weeks)

        choice_history = {}

        for t_idx, week in enumerate(weeks):
            continuation = np.zeros(n_categories, dtype=np.float64)
            switch = np.zeros(n_categories, dtype=np.float64)

            if t_idx > 0:
                for s in range(t_idx):
                    weeks_elapsed = pd.Timedelta(week - weeks[s]).days / 7.0
                    discount = delta_discount**weeks_elapsed
                    y_s = choice_history[s]

                    continuation += discount * y_s
                    if np.sum(y_s) > 0:
                        for j in range(n_categories):
                            prod_k = 1.0 if y_s[j] == 0 else 0.0
                            switch[j] += discount * prod_k

            C_t = (lambda_own * continuation) + ((1.0 - lambda_own) * switch)

            record = {
                "household_code": hh_id,
                "week_end": pd.Timestamp(week).strftime("%Y-%m-%d"),
            }
            for j in range(n_categories):
                record[f"C_sat_{j}"] = C_t[j]
            purchase_records.append(record)

            sub = hh_group[hh_group["week_end"] == week]
            Y_t = np.zeros(n_categories, dtype=np.float64)
            for row in sub.itertuples():
                if hasattr(row, "choice_idx") and row.choice_idx < n_categories:
                    Y_t[int(row.choice_idx)] = 1.0
            choice_history[t_idx] = Y_t.copy()

    df_sat = pd.DataFrame(purchase_records)

    for j in range(n_categories):
        c_col = f"C_sat_{j}"
        std_val = df_sat[c_col].std()
        if std_val > 0:
            df_sat[c_col] = df_sat[c_col] / std_val

    return df_sat


def load_and_preprocess(weekly_budget=50.0):
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

    merged_df["household_income_dollars"] = merged_df["household_income"].map(
        INCOME_CODE_MAP
    )
    median_inc = merged_df["household_income_dollars"].median()
    merged_df["household_income_dollars"] = merged_df[
        "household_income_dollars"
    ].fillna(median_inc)

    descr_col = "flavor_descr" if "flavor_descr" in merged_df.columns else "flavor_cd"
    merged_df["flavor_str"] = (
        merged_df[descr_col].fillna("").astype(str).str.lower().str.strip()
    )

    plain_regex = r"plain|unflavored"
    berry_regex = r"berry|strawb|blueb|rasp|blackb|cherry|cran|wildberry"

    def assign_category(f_str):
        if pd.isna(f_str) or f_str == "":
            return "outside"
        if pd.Series(f_str).str.contains(plain_regex, regex=True).iloc[0]:
            return "plain"
        if pd.Series(f_str).str.contains(berry_regex, regex=True).iloc[0]:
            return "berry"
        return "other"

    merged_df["category"] = merged_df["flavor_str"].apply(assign_category)

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

    hh_stats = (
        merged_df.groupby("household_code")
        .agg(income=("household_income_dollars", "first"))
        .reset_index()
    )
    hh_stats["income_std"] = (hh_stats["income"] - hh_stats["income"].mean()) / (
        hh_stats["income"].std() + 1e-6
    )
    hh_dict = hh_stats.set_index("household_code")["income_std"].to_dict()

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
        mat = np.zeros((3, 2), dtype=np.float64)

        for cat, idx in cat_map.items():
            if cat in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat]

        for row in group.itertuples():
            if row.category in cat_map:
                idx = cat_map[row.category]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    # CONTINUING FROM WHERE YOU LEFT OFF:
    console.print("\n[bold green]Computing Inter-Trip Lattin Stock...[/bold green]")
    df_sat = compute_lattin_intertrip_stock(
        weekly_purchases,
        n_categories=3,
        delta_discount=BASE_DELTA_DISCOUNT,
        lambda_own=LAMBDA_OWN_WEIGHT,
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

    quantities_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        q_arr = np.zeros(3, dtype=np.float64)
        for row in grp.itertuples():
            if row.choice_idx < 3:
                q_arr[int(row.choice_idx)] = float(row.units_bought)
        quantities_dict[(hh_id, week)] = q_arr

    all_prices, all_resids, all_quantities, all_c_states, all_inc, all_budgets = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = (
            row_dict["week_end_x"] if "week_end_x" in row_dict else row_dict["week_end"]
        )
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        inc_val = hh_dict.get(hh_id, 0.0)
        p_vec = choice_set_matrix[(store, week)][:3, 0]
        r_vec = choice_set_matrix[(store, week)][:3, 1]

        q_vec = quantities_dict.get((hh_id, week), np.zeros(3, dtype=np.float64))
        inside_spend = np.sum(q_vec * p_vec)

        budget = max(weekly_budget, inside_spend + 5.0)

        c_vec = np.array(
            [
                row_dict.get("C_sat_0", 0.0),
                row_dict.get("C_sat_1", 0.0),
                row_dict.get("C_sat_2", 0.0),
            ],
            dtype=np.float64,
        )

        all_prices.append(p_vec)
        all_resids.append(r_vec)
        all_quantities.append(q_vec)
        all_c_states.append(c_vec)
        all_inc.append(inc_val)
        all_budgets.append(budget)

    return {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "quantities": np.array(all_quantities, dtype=np.float64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "income": np.array(all_inc, dtype=np.float64)[:, None],
        "budgets": np.array(all_budgets, dtype=np.float64)[:, None],
    }


def total_objective_bhat_lattin(params, vec_data):
    """
    BHAT (2008) MDCEV Likelihood with LATTIN (1987) Inter-Trip Reference Stock
    Parameter Order (11 parameters):
    0: Constant
    1..2: Beta_Berry, Beta_Plain (vs Other Base = 0.0)
    3..4: Beta_Berry x Inc, Beta_Plain x Inc
    5: Gamma_0 (Global Inter-Trip Lattin Satiation)
    6: Gamma_Inc (Inter-Trip Satiation x Income)
    7: Price alpha_p
    8: Control Func sigma
    9..10: Intra-Trip Translation Theta (Berry, Plain)
    """
    const = params[0]
    b0 = params[1:3]
    b_inc = params[3:5]
    g0 = params[5]
    g_inc = params[6]
    alpha_p = params[7]
    sigma = params[8]
    theta_inside = np.exp(params[9:11])

    prices = vec_data["prices"]  # (N, 3)
    resids = vec_data["resids"]  # (N, 3)
    quantities = vec_data["quantities"]  # (N, 3)
    c_state = vec_data["c_states"]  # (N, 3)
    inc = vec_data["income"]  # (N, 1)
    budgets = vec_data["budgets"]  # (N, 1)

    N = prices.shape[0]

    beta_non_other = b0[None, :] + b_inc[None, :] * inc
    beta_mat = np.hstack([np.zeros((inc.shape[0], 1)), beta_non_other])
    gamma_i = g0 + g_inc * inc

    v_inside = (
        const
        + beta_mat
        + gamma_i * np.log(1.0 + c_state)
        + alpha_p * prices
        + sigma * resids
    )

    inside_spend = np.sum(prices * quantities, axis=1, keepdims=True)
    z_outside = np.maximum(1e-4, budgets - inside_spend)

    theta_full = np.hstack([np.ones((N, 1)), np.tile(theta_inside[None, :], (N, 1))])

    f_inside = v_inside - np.log(prices) - np.log(1.0 + quantities / theta_full)
    f_outside = -np.log(z_outside)

    is_chosen = (quantities > 0).astype(np.float64)
    M = np.sum(is_chosen, axis=1) + 1.0

    c_j = (1.0 / (quantities + theta_full)) * is_chosen
    sum_c = np.sum(c_j / prices, axis=1) + (1.0 / z_outside.squeeze())
    prod_c = np.prod(np.where(is_chosen == 1, c_j, 1.0), axis=1)
    jacobian_det = prod_c * sum_c

    diff_f = f_inside - f_outside
    sum_exp_f = 1.0 + np.sum(np.exp(diff_f), axis=1)

    ll_i = (
        np.log(np.maximum(1e-12, jacobian_det))
        + np.sum(diff_f * is_chosen, axis=1)
        - M * np.log(sum_exp_f)
    )
    total_ll = np.sum(ll_i)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_bhat_lattin_model(vec_data):
    total_params = 11
    x0 = np.zeros(total_params)
    x0[0] = 0.5
    x0[1] = 0.5
    x0[2] = -0.5
    x0[5] = -0.1
    x0[7] = -0.5
    x0[9] = 0.0
    x0[10] = 0.0

    bounds = (
        [(-20.0, 20.0)] * 5
        + [(-10.0, 0.0)]  # Enforce gamma_0 <= 0 for satiation
        + [(-10.0, 10.0)]  # gamma_inc
        + [(-30.0, -0.01), (-30.0, 30.0)]
        + [(-5.0, 5.0)] * 2
    )

    res = minimize(
        total_objective_bhat_lattin,
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

            f1 = total_objective_bhat_lattin(x1, vec_data)
            f2 = total_objective_bhat_lattin(x2, vec_data)
            f3 = total_objective_bhat_lattin(x3, vec_data)
            f4 = total_objective_bhat_lattin(x4, vec_data)

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


def display_results(results):
    table = Table(
        title="BHAT (2008) MDCEV + LATTIN (1987) INTER/INTRA TRIP MODEL",
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
        "beta_Berry (vs Other Base)",
        "beta_Plain (vs Other Base)",
        "beta_Berry x Income",
        "beta_Plain x Income",
        "gamma_0 (Inter-Trip Lattin Satiation)",
        "gamma x Income",
        "Price (alpha_p)",
        "Control Func. (sigma)",
        "theta_Berry (Intra-Trip Translation)",
        "theta_Plain (Intra-Trip Translation)",
    ]

    estimates = results["params"].copy()
    estimates[9] = np.exp(estimates[9])
    estimates[10] = np.exp(estimates[10])

    for name, val, se, z, p in zip(
        param_names,
        estimates,
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
        "\n[bold yellow]--- Preprocessing Data for Bhat + Lattin Joint Model ---[/bold yellow]"
    )
    vec_data = load_and_preprocess()

    console.print("\n[bold yellow]--- Estimating Model ---[/bold yellow]")
    res_model = estimate_bhat_lattin_model(vec_data)
    display_results(res_model)


if __name__ == "__main__":
    main()
