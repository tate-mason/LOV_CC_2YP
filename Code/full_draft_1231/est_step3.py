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

CATEGORIES = ["other", "berry", "plain", "outside"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}
CALIBRATED_LAMBDA = 0.7
BASE_DELTA_DISCOUNT = 0.9


def compute_dynamic_satiation_state(
    weekly_purchases,
    n_categories=3,
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
                diff_days = pd.Timedelta(week - weeks[t_idx - 1]).days
                delta_t = max(1.0, diff_days / 7.0)
            else:
                delta_t = 1.0

            if t_idx > 0:
                for s in range(t_idx):
                    weeks_elapsed = pd.Timedelta(week - weeks[s]).days / 7.0
                    discount = delta_discount**weeks_elapsed
                    memory_term += discount * C_history[s]

            for j in range(n_categories):
                cross_term = np.sum(prev_X) - prev_X[j]
                C_current[j] = (lambda_mem * memory_term[j]) + (
                    (1.0 - lambda_mem) * cross_term
                )

            record = {
                "household_code": hh_id,
                "week_end": pd.Timestamp(week).strftime("%Y-%m-%d"),
                "delta_t": delta_t,
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

    return pd.DataFrame(inventory_records)


def load_and_preprocess_interactive_gamma(weekly_capacity=14):
    schema_names = pl.scan_parquet(MERGED_PATH).collect_schema().names()

    exprs = [
        pl.col("quantity").cast(pl.Int64),
        pl.col("household_income").cast(pl.Float64),
        pl.col("household_size").cast(pl.Int64),
        pl.col("serving_per_container_cd").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
    ]

    if "type_of_residence" in schema_names:
        exprs.append(pl.col("type_of_residence").cast(pl.Utf8))
    elif "residence_type" in schema_names:
        exprs.append(pl.col("residence_type").cast(pl.Utf8))

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

    res_col = (
        "type_of_residence"
        if "type_of_residence" in merged_df.columns
        else "residence_type"
    )
    if res_col in merged_df.columns:
        merged_df["single_family_res"] = np.where(
            merged_df[res_col]
            .astype(str)
            .str.contains("1|single|detached", case=False, na=False),
            1.0,
            0.0,
        )
    else:
        merged_df["single_family_res"] = 0.0

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

    # Standardize Demographics & Purchase Velocity
    hh_stats = (
        merged_df.groupby("household_code")
        .agg(
            income=("household_income", "first"),
            single_fam=("single_family_res", "first"),
            obs_weeks=("week_end", "nunique"),
        )
        .reset_index()
    )

    hh_stats["income_std"] = (hh_stats["income"] - hh_stats["income"].mean()) / (
        hh_stats["income"].std() + 1e-6
    )
    hh_stats["freq_std"] = (hh_stats["obs_weeks"] - hh_stats["obs_weeks"].mean()) / (
        hh_stats["obs_weeks"].std() + 1e-6
    )

    hh_dict = hh_stats.set_index("household_code")[
        ["income_std", "single_fam", "freq_std"]
    ].to_dict("index")

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

    df_sat = compute_dynamic_satiation_state(
        weekly_purchases,
        n_categories=3,
        lambda_mem=CALIBRATED_LAMBDA,
        delta_discount=0.9,
    )
    for j in range(3):
        c_col = f"C_sat_{j}"
        c_sd = df_sat[c_col].std()
        if c_sd > 0:
            df_sat[c_col] = df_sat[c_col] / c_sd

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

    all_prices, all_resids, all_choices, all_c_states = [], [], [], []
    all_inc, all_res_type, all_freq, all_delta_t = [], [], [], []

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = (
            row_dict["week_end_x"] if "week_end_x" in row_dict else row_dict["week_end"]
        )
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        h_info = hh_dict.get(
            hh_id, {"income_std": 0.0, "single_fam": 0.0, "freq_std": 0.0}
        )

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
        all_inc.append(h_info["income_std"])
        all_res_type.append(h_info["single_fam"])
        all_freq.append(h_info["freq_std"])
        all_delta_t.append(row_dict.get("delta_t", 1.0))

    return {
        "prices": np.array(all_prices, dtype=np.float64),
        "resids": np.array(all_resids, dtype=np.float64),
        "choices": np.array(all_choices, dtype=np.int64),
        "c_states": np.array(all_c_states, dtype=np.float64),
        "income": np.array(all_inc, dtype=np.float64)[:, None],
        "res_type": np.array(all_res_type, dtype=np.float64)[:, None],
        "freq": np.array(all_freq, dtype=np.float64)[:, None],
        "delta_t": np.array(all_delta_t, dtype=np.float64)[:, None],
    }


# =========================================================
# HOUSEHOLD-TRIP INTERACTIVE GAMMA LOGIT OBJECTIVE
# =========================================================
def total_objective_interactive_gamma(params, vec_data):
    """
    Parameter Order (14 parameters):
    0: Constant
    1..2: Beta_0 [Berry, Plain] (vs Other base = 0.0)
    3..4: Beta_Inc [Berry, Plain]
    5..6: Beta_ResType [Berry, Plain]
    7: Gamma_0 (Base Scalar Satiation)
    8: Gamma_Inc
    9: Gamma_DeltaT (Inter-trip gap)
    10: Gamma_Freq (Panel purchase velocity)
    11: Gamma_ResType (Residence type)
    12: Price (alpha)
    13: Control Function (sigma)
    """
    const = params[0]

    b0 = params[1:3]
    b_inc = params[3:5]
    b_res = params[5:7]

    g0 = params[7]
    g_inc = params[8]
    g_dt = params[9]
    g_freq = params[10]
    g_res = params[11]

    alpha = params[-2]
    sigma = params[-1]

    prices = vec_data["prices"]  # (N, 3)
    resids = vec_data["resids"]  # (N, 3)
    choices = vec_data["choices"]  # (N, 4)
    c_state = vec_data["c_states"]  # (N, 3)
    inc = vec_data["income"]  # (N, 1)
    res_type = vec_data["res_type"]  # (N, 1)
    freq = vec_data["freq"]  # (N, 1)
    delta_t = vec_data["delta_t"]  # (N, 1)

    # Beta Matrix (N, 3): Other fixed to 0.0
    beta_non_other = b0[None, :] + b_inc[None, :] * inc + b_res[None, :] * res_type
    beta_mat = np.hstack([np.zeros((inc.shape[0], 1)), beta_non_other])

    # Household-Trip Specific Scalar Satiation Gamma_i,t (N, 1)
    gamma_i_t = g0 + g_inc * inc + g_dt * delta_t + g_freq * freq + g_res * res_type

    u_inside = (
        const
        + beta_mat
        + gamma_i_t * np.log(1.0 + c_state)
        + alpha * prices
        + sigma * resids
    )

    u_outside = np.zeros((u_inside.shape[0], 1))
    u = np.hstack([u_inside, u_outside])

    log_probs = u - logsumexp(u, axis=1, keepdims=True)
    total_ll = np.sum(log_probs * choices)

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_interactive_gamma_model(vec_data):
    total_params = 14
    x0 = np.zeros(total_params)
    x0[7] = -0.1  # Initial gamma_0 guess
    x0[-2] = -0.1  # Initial price alpha guess

    bounds = (
        [(-30.0, 30.0)] * 7  # const + betas (0, inc, res_type)
        + [(-30.0, 0.0)]  # gamma_0 bound (disutility <= 0)
        + [(-30.0, 30.0)] * 4  # gamma modifiers (inc, delta_t, freq, res_type)
        + [(-30.0, -0.01), (-30.0, 30.0)]  # price + control function
    )

    res = minimize(
        total_objective_interactive_gamma,
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

            f1 = total_objective_interactive_gamma(x1, vec_data)
            f2 = total_objective_interactive_gamma(x2, vec_data)
            f3 = total_objective_interactive_gamma(x3, vec_data)
            f4 = total_objective_interactive_gamma(x4, vec_data)

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
        title="HOUSEHOLD-TRIP INTERACTIVE SATIATION LOGIT (GAMMA_I,T)",
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
        "beta_Berry x SingleFamilyRes",
        "beta_Plain x SingleFamilyRes",
        "gamma_0 (Base Scalar Satiation)",
        "gamma x Income",
        "gamma x Delta_t (Inter-Trip Gap)",
        "gamma x Trip Frequency",
        "gamma x SingleFamilyRes",
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
        "\n[bold yellow]--- Preprocessing Data for Household-Trip Interactive Gamma Model ---[/bold yellow]"
    )
    vec_data = load_and_preprocess_interactive_gamma()

    console.print(
        "\n[bold yellow]--- Estimating Interactive Satiation Model (gamma_i,t) ---[/bold yellow]"
    )
    res_model = estimate_interactive_gamma_model(vec_data)
    display_results(res_model)


if __name__ == "__main__":
    main()
