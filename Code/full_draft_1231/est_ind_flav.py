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

TOP_K_FLAVORS = 10  # Top 10 individual non-plain flavor codes
CALIBRATED_LAMBDA = 0.7  # Calibrated memory decay parameter


def compute_satiation_state_pairwise(
    weekly_purchases,
    lambda_mem=CALIBRATED_LAMBDA,
    delta_discount=0.9,
):
    """
    Computes satiation states C_sat for a 2-choice inside set:
    Index 0: Plain
    Index 1: Focal Flavor Code
    """
    sorted_df = weekly_purchases.sort_values(["household_code", "week_end"])
    inventory_records = []

    for hh_id, hh_group in sorted_df.groupby("household_code"):
        weeks = hh_group["week_end"].unique()
        C_history = {}
        prev_X = np.zeros(2)

        for t_idx, week in enumerate(weeks):
            C_current = np.zeros(2)
            memory_term = np.zeros(2)

            if t_idx > 0:
                for s in range(t_idx):
                    discount = delta_discount ** (t_idx - s)
                    memory_term += discount * C_history[s]

            for j in range(2):
                cross_term = np.sum(prev_X) - prev_X[j]
                C_current[j] = (lambda_mem * memory_term[j]) + (
                    (1.0 - lambda_mem) * cross_term
                )

            record = {
                "household_code": hh_id,
                "week_end": week,
                "C_sat_plain": C_current[0],
                "C_sat_focal": C_current[1],
            }
            inventory_records.append(record)

            C_history[t_idx] = C_current.copy()

            sub = hh_group[hh_group["week_end"] == week]
            curr_X = np.zeros(2)
            for row in sub.itertuples():
                if row.choice_idx < 2:
                    curr_X[row.choice_idx] = row.units_bought
            total_inside_units = curr_X.sum()
            if total_inside_units > 0:
                curr_X = curr_X / total_inside_units
            prev_X = curr_X

    return pd.DataFrame(inventory_records)


def extract_top_flavors_and_mapping(merged_df, top_k=TOP_K_FLAVORS):
    descr_col = "flavor_descr" if "flavor_descr" in merged_df.columns else "flavor_cd"
    merged_df["flavor_str"] = (
        merged_df[descr_col].fillna("").astype(str).str.lower().str.strip()
    )

    plain_regex = r"plain|unflavored"
    is_plain = merged_df["flavor_str"].str.contains(plain_regex, na=False)

    if "flavor_cd" in merged_df.columns:
        merged_df["code_val"] = merged_df["flavor_cd"].astype(str)
    else:
        merged_df["code_val"] = merged_df["flavor_str"]

    non_plain_df = merged_df[~is_plain]
    top_flavor_codes = (
        non_plain_df["code_val"].value_counts().head(top_k).index.tolist()
    )

    flavor_mapping = {}
    for code in top_flavor_codes:
        sub_df = merged_df[merged_df["code_val"] == code]
        if (
            "flavor_descr" in sub_df.columns
            and sub_df["flavor_descr"].dropna().count() > 0
        ):
            best_descr = sub_df["flavor_descr"].dropna().mode()[0]
        else:
            best_descr = f"Flavor Code {code}"
        flavor_mapping[code] = str(best_descr).title()

    flavor_mapping["plain"] = "Plain / Unflavored (Base)"

    return top_flavor_codes, flavor_mapping, merged_df


def load_and_preprocess_pairwise_code(focal_code, merged_df, weekly_capacity=14):
    plain_regex = r"plain|unflavored"

    def assign_category(row):
        f_str = row["flavor_str"]
        c_val = row["code_val"]
        if pd.isna(f_str) or f_str == "":
            return "outside"
        if pd.Series(f_str).str.contains(plain_regex, regex=True).iloc[0]:
            return "plain"
        if c_val == focal_code:
            return focal_code
        return "outside"

    sub_df = merged_df.copy()
    sub_df["category"] = sub_df.apply(assign_category, axis=1)

    categories = ["plain", focal_code, "outside"]
    cat_map = {c: i for i, c in enumerate(categories)}

    # Price IV Construction
    market_price = (
        sub_df.groupby(["upc", "week_end", "market_name"])["price"].mean().reset_index()
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

    sub_df = sub_df.merge(
        market_price[["upc", "week_end", "market_name", "price_iv"]],
        on=["upc", "week_end", "market_name"],
        how="left",
    )

    sub_df = sub_df[(sub_df["price"] > 0.1) | (sub_df["price"].isna())]
    iv_res = smf.ols(
        "price ~ price_iv + brand_cd + C(week_end)", data=sub_df, missing="drop"
    ).fit()

    sub_df["iv_res"] = np.nan
    sub_df.loc[iv_res.model.data.row_labels, "iv_res"] = iv_res.resid
    valid_resids = sub_df["iv_res"].dropna()
    if len(valid_resids) > 0:
        res_mean = valid_resids.mean()
        res_std = valid_resids.std()
        sub_df["iv_res"] = (sub_df["iv_res"] - res_mean) / (
            res_std if res_std > 0 else 1.0
        )
    sub_df["iv_res"] = sub_df["iv_res"].fillna(0.0)

    # Standardize Household Income
    hh_demo = (
        sub_df.groupby("household_code")
        .agg(income=("household_income", "first"))
        .reset_index()
    )

    hh_demo["income_std"] = (hh_demo["income"] - hh_demo["income"].mean()) / (
        hh_demo["income"].std() + 1e-6
    )
    demo_dict = hh_demo.set_index("household_code")["income_std"].to_dict()

    inside_df = sub_df[sub_df["category"] != "outside"].copy()

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
        mat = np.zeros((2, 2), dtype=np.float64)

        for cat, idx in cat_map.items():
            if cat != "outside" and cat in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat]

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    df_sat = compute_satiation_state_pairwise(
        weekly_purchases, lambda_mem=CALIBRATED_LAMBDA, delta_discount=0.9
    )

    for col in ["C_sat_plain", "C_sat_focal"]:
        c_sd = df_sat[col].std()
        if c_sd > 0:
            df_sat[col] = df_sat[col] / c_sd

    hh_weeks = weekly_purchases[
        ["household_code", "store_code_uc", "week_end"]
    ].drop_duplicates()
    hh_weeks = hh_weeks.merge(
        df_sat, on=["household_code", "week_end"], how="left"
    ).fillna(0.0)

    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(2, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < 2:
                c_arr[row.choice_idx] = row.units_bought
        counts_dict[(hh_id, week)] = c_arr

    all_prices, all_resids, all_choices, all_c_states, all_inc = [], [], [], [], []

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = row_dict["week_end"]
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        inc_val = demo_dict.get(hh_id, 0.0)

        inside_counts = counts_dict.get((hh_id, week), np.zeros(2, dtype=np.int64))
        inside_units = np.sum(inside_counts)

        effective_capacity = max(weekly_capacity, inside_units + 1)
        outside_count = effective_capacity - inside_units

        full_counts = np.append(inside_counts, outside_count)
        c_vec = np.array(
            [row_dict["C_sat_plain"], row_dict["C_sat_focal"]], dtype=np.float64
        )

        all_prices.append(choice_set_matrix[(store, week)][:2, 0])
        all_resids.append(choice_set_matrix[(store, week)][:2, 1])
        all_choices.append(full_counts)
        all_c_states.append(c_vec)
        all_inc.append(inc_val)

    vec_data = {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "income": np.array(all_inc, dtype=np.float64)[:, None],
        "focal_code": focal_code,
    }

    return vec_data


# =========================================================
# PAIRWISE MODEL OBJECTIVE & ESTIMATION
# =========================================================
def total_objective_pairwise(params, vec_data):
    const = params[0]
    b0 = params[1]
    b_inc = params[2]

    g0 = params[3]
    g_inc = params[4]

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]  # (N, 2)
    resids = vec_data["resids"]  # (N, 2)
    choices = vec_data["choices"]  # (N, 3)
    c_state = vec_data["c_states"]  # (N, 2)
    inc = vec_data["income"]  # (N, 1)

    beta_focal = b0 + b_inc * inc
    beta_mat = np.hstack([np.zeros((inc.shape[0], 1)), beta_focal])

    gamma_focal = g0 + g_inc * inc
    gamma_mat = np.hstack([np.zeros((inc.shape[0], 1)), gamma_focal])

    u_inside = (
        const
        + beta_mat
        + gamma_mat * np.log(1.0 + c_state)
        + alpha * prices
        + sigma * resids
    )

    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_pairwise_model(vec_data):
    total_params = 7
    x0 = np.zeros(total_params)
    x0[-2] = -0.1

    bounds = (
        [(-30.0, 30.0)] * 3  # const + beta_0, beta_inc
        + [(-30.0, 0.0)]  # gamma_0 (negative disutility)
        + [(-30.0, 30.0)]  # gamma_inc
        + [(-30.0, -0.01), (-30.0, 30.0)]  # price + control function
    )

    res = minimize(
        total_objective_pairwise,
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

            f1 = total_objective_pairwise(x1, vec_data)
            f2 = total_objective_pairwise(x2, vec_data)
            f3 = total_objective_pairwise(x3, vec_data)
            f4 = total_objective_pairwise(x4, vec_data)

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


def display_consolidated_top_k_results(all_results, flavor_mapping):
    table = Table(
        title="TOP 10 FLAVOR CODES: PAIRWISE ESTIMATION WITH INCOME INTERACTION (PLAIN = BASE)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Flavor Code", style="cyan", justify="center")
    table.add_column("Description", style="white", justify="left")
    table.add_column("beta_0 (Main)", justify="right")
    table.add_column("beta x Inc", justify="right")
    table.add_column("gamma_0 (Main)", justify="right")
    table.add_column("gamma x Inc", justify="right")
    table.add_column("Price (alpha)", justify="right")
    table.add_column("LL Objective", justify="right")

    for code, res in all_results.items():
        p = res["params"]
        p_vals = res["p_val"]

        def format_val_p(val, p_val):
            stars = ""
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.05:
                stars = "**"
            return f"{val:.4f} {stars}".strip()

        table.add_row(
            str(code),
            flavor_mapping.get(code, "Unknown"),
            format_val_p(p[1], p_vals[1]),
            format_val_p(p[2], p_vals[2]),
            format_val_p(p[3], p_vals[3]),
            format_val_p(p[4], p_vals[4]),
            format_val_p(p[-2], p_vals[-2]),
            f"{res['fun']:.2f}",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

    exprs = [
        pl.col("quantity").cast(pl.Int64),
        pl.col("household_income").cast(pl.Float64),
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

    top_codes, flavor_mapping, merged_df = extract_top_flavors_and_mapping(
        merged_df, top_k=TOP_K_FLAVORS
    )

    console.print(
        f"\n[bold yellow]Estimating Pairwise Models for Top {TOP_K_FLAVORS} Flavor Codes:[/bold yellow] {top_codes}\n"
    )

    all_results = {}

    for idx, code in enumerate(top_codes):
        console.print(
            f"[bold green][{idx + 1}/{TOP_K_FLAVORS}] Estimating Pairwise Model: Plain vs Code {code} ({flavor_mapping.get(code)})...[/bold green]"
        )
        vec_data = load_and_preprocess_pairwise_code(code, merged_df)
        res = estimate_pairwise_model(vec_data)
        all_results[code] = res

    display_consolidated_top_k_results(all_results, flavor_mapping)


if __name__ == "__main__":
    main()
