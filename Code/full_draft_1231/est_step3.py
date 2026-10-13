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

CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}
CALIBRATED_LAMBDA = 0.7
BASE_DELTA_DISCOUNT = 0.9

# ==============================================================================
# OPTIONAL TEMPLATE: CUSTOM HOUSEHOLD INCOME REMAPPING TABLE
# Un-comment and modify this dictionary to recode raw Nielsen income codes
# into custom continuous values or binned brackets.
# ==============================================================================
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


def compute_satiation_state(
    weekly_purchases,
    n_categories=3,
    lambda_mem=CALIBRATED_LAMBDA,
    delta_discount=BASE_DELTA_DISCOUNT,
):
    """
    Computes PURE OWN-FLAVOR satiation states C_sat.
    Only past purchases of flavor j feed C_{i,j,t}.
    """
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
                # Pure Own-Flavor Stock: prev_X[j] ONLY
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

    # Standard scale across column to prevent optimization gradient issues
    for j in range(n_categories):
        c_col = f"C_sat_{j}"
        std_val = df_sat[c_col].std()
        if std_val > 0:
            df_sat[c_col] = df_sat[c_col] / std_val

    return df_sat


def load_and_preprocess(weekly_capacity=14):
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

    # ==========================================================================
    # PRINT HOUSEHOLD INCOME DISTRIBUTION AT THE TOP
    # ==========================================================================
    hh_unique_inc = (
        merged_df.groupby("household_code")["household_income"].first().dropna()
    )
    inc_counts = hh_unique_inc.value_counts().sort_index()

    table_inc = Table(
        title="PANEL HOUSEHOLD INCOME DISTRIBUTION (VALUE COUNTS)",
        show_header=True,
        header_style="bold green",
    )
    table_inc.add_column("Income Code / Value", style="cyan", justify="center")
    table_inc.add_column("Household Count", style="bold white", justify="right")
    table_inc.add_column("Percentage (%)", style="yellow", justify="right")

    tot_hh = len(hh_unique_inc)
    for inc_val, cnt in inc_counts.items():
        pct = (cnt / tot_hh) * 100.0
        table_inc.add_row(str(int(inc_val)), f"{cnt:,}", f"{pct:.2f}%")

    console.print(table_inc)
    console.print(
        f"[bold white]Total Unique Households:[/bold white] [bold cyan]{tot_hh:,}[/bold cyan]\n"
    )

    if "INCOME_CODE_MAP" in globals():
        merged_df["household_income"] = (
            merged_df["household_income"]
            .map(INCOME_CODE_MAP)
            .fillna(merged_df["household_income"])
        )

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
        .agg(income=("household_income", "first"))
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
            if cat != "outside" and cat in overall_cat_prices:
                mat[idx, 0] = overall_cat_prices[cat]

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                mat[idx] = [row.price, row.iv_res]

        choice_set_matrix[(store, week)] = mat

    console.print(
        "[bold green]Computing pure own-flavor satiation states (C_sat)...[/bold green]"
    )
    df_sat = compute_satiation_state(
        weekly_purchases,
        n_categories=3,
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
        c_arr = np.zeros(3, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < 3:
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

        inside_counts = counts_dict.get((hh_id, week), np.zeros(3, dtype=np.int64))
        inside_units = np.sum(inside_counts)

        effective_capacity = max(weekly_capacity, inside_units + 1)
        outside_count = effective_capacity - inside_units

        full_counts = np.append(inside_counts, outside_count)
        c_vec = np.array(
            [
                row_dict.get("C_sat_0", 0.0),
                row_dict.get("C_sat_1", 0.0),
                row_dict.get("C_sat_2", 0.0),
            ],
            dtype=np.float64,
        )

        all_prices.append(choice_set_matrix[(store, week)][:3, 0])
        all_resids.append(choice_set_matrix[(store, week)][:3, 1])
        all_choices.append(full_counts)
        all_c_states.append(c_vec)
        all_inc.append(inc_val)

    return {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "income": np.array(all_inc, dtype=np.float64)[:, None],
    }


# =========================================================
# LOGIT OBJECTIVE: OWN-FLAVOR SATIATION C_SAT
# =========================================================
def total_objective_own_flavor_satiation(params, vec_data):
    """
    Parameter Order (9 parameters):
    0: Constant
    1..2: Beta_0 [Berry, Plain] (vs Other base = 0.0)
    3..4: Beta_Inc [Berry, Plain]
    5: Gamma_0 (Global Scalar Base Satiation)
    6: Gamma_Inc (Global Satiation x Income)
    7: Price (alpha)
    8: Control Function (sigma)
    """
    const = params[0]

    b0 = params[1:3]
    b_inc = params[3:5]

    g0 = params[5]
    g_inc = params[6]

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]  # (N, 3)
    resids = vec_data["resids"]  # (N, 3)
    choices = vec_data["choices"]  # (N, 4)
    c_state = vec_data["c_states"]  # (N, 3) - Own-flavor C_sat
    inc = vec_data["income"]  # (N, 1)

    # Flavor-Specific Preference Beta Matrix (N, 3): Other (index 0) fixed to 0.0
    beta_non_other = b0[None, :] + b_inc[None, :] * inc
    beta_mat = np.hstack([np.zeros((inc.shape[0], 1)), beta_non_other])

    # Global Satiation Gamma_i (N, 1) applied across all inside options
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


def estimate_model(vec_data):
    total_params = 9
    x0 = np.zeros(total_params)
    x0[5] = -0.1  # Initial gamma_0 guess
    x0[-2] = -0.1  # Initial price alpha guess

    bounds = (
        [(-30.0, 30.0)] * 5  # const, beta_0, beta_inc for Berry & Plain
        + [(-30.0, 30.0)]  # gamma_0
        + [(-30.0, 30.0)]  # gamma_inc
        + [(-30.0, -0.01), (-30.0, 30.0)]  # price + control function
    )

    res = minimize(
        total_objective_own_flavor_satiation,
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

            f1 = total_objective_own_flavor_satiation(x1, vec_data)
            f2 = total_objective_own_flavor_satiation(x2, vec_data)
            f3 = total_objective_own_flavor_satiation(x3, vec_data)
            f4 = total_objective_own_flavor_satiation(x4, vec_data)

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
        title="PURE OWN-FLAVOR SATIATION LOGIT (BASE = OTHER)",
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
        "gamma_0 (Global Base Satiation)",
        "gamma x Income",
        "Price (alpha)",
        "Control Func. (sigma)",
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
    console.print(f"[bold]Final LL Objective:[/bold] {results['fun']:.4f}\n")


def main():
    console.print(
        "\n[bold yellow]--- Preprocessing Data with Pure Own-Flavor Satiation ---[/bold yellow]"
    )
    vec_data = load_and_preprocess()

    console.print(
        "\n[bold yellow]--- Estimating Pure Own-Flavor Model ---[/bold yellow]"
    )
    res_model = estimate_model(vec_data)
    display_results(res_model)


if __name__ == "__main__":
    main()
