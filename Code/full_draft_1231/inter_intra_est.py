import itertools
import os
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

CATEGORIES = ["other", "berry", "plain"]
cat_map = {c: i for i, c in enumerate(CATEGORIES)}

BASE_DELTA_DISCOUNT = 0.85
LAMBDA_OWN_WEIGHT = 0.70

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

        choice_history = {}  # Stores volume share vectors for past trips

        for t_idx, week in enumerate(weeks):
            continuation = np.zeros(n_categories, dtype=np.float64)
            switch = np.zeros(n_categories, dtype=np.float64)

            if t_idx > 0:
                for s in range(t_idx):
                    weeks_elapsed = pd.Timedelta(week - weeks[s]).days / 7.0
                    discount = delta_discount**weeks_elapsed
                    shares_s = choice_history[s]

                    # Own-flavor stock scaled by volume share bought on trip s
                    continuation += discount * shares_s

                    # Cross-flavor stock for non-chosen/minor flavors
                    if np.sum(shares_s) > 0:
                        for j in range(n_categories):
                            cross_weight = 1.0 - shares_s[j]
                            switch[j] += discount * cross_weight

            C_t = (lambda_own * continuation) + ((1.0 - lambda_own) * switch)

            record = {
                "household_code": hh_id,
                "week_end": pd.Timestamp(week).strftime("%Y-%m-%d"),
            }
            for j in range(n_categories):
                record[f"C_sat_{j}"] = C_t[j]
            purchase_records.append(record)

            # Record volume shares for trip t
            sub = hh_group[hh_group["week_end"] == week]
            q_t = np.zeros(n_categories, dtype=np.float64)
            for row in sub.itertuples():
                if hasattr(row, "choice_idx") and row.choice_idx < n_categories:
                    q_t[int(row.choice_idx)] = float(row.units_bought)

            total_q = np.sum(q_t)
            shares_t = (
                q_t / total_q
                if total_q > 0
                else np.zeros(n_categories, dtype=np.float64)
            )
            choice_history[t_idx] = shares_t.copy()

    df_sat = pd.DataFrame(purchase_records)

    # Standardize across panel for numerical stability
    for j in range(n_categories):
        c_col = f"C_sat_{j}"
        std_val = df_sat[c_col].std()
        if std_val > 0:
            df_sat[c_col] = df_sat[c_col] / std_val

    return df_sat


def generate_basket_permutations(counts):
    """Generates all unique sequences of item draws for a given basket vector [n_other, n_berry, n_plain]."""
    items = []
    for flavor_idx, cnt in enumerate(counts):
        items.extend([flavor_idx] * int(cnt))
    if len(items) == 0:
        return []
    return list(set(itertools.permutations(items)))


def load_and_preprocess():
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

    console.print(
        "\n[bold green]Computing Inter-Trip Lattin Stock (Volume-Share Weighted)...[/bold green]"
    )
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

    counts_dict = {}
    for (hh_id, week), grp in weekly_purchases.groupby(["household_code", "week_end"]):
        c_arr = np.zeros(3, dtype=np.int64)
        for row in grp.itertuples():
            if row.choice_idx < 3:
                c_arr[int(row.choice_idx)] = int(row.units_bought)
        counts_dict[(hh_id, week)] = c_arr

    trip_data = []

    for row_dict in hh_weeks.to_dict("records"):
        store = row_dict["store_code_uc"]
        week = (
            row_dict["week_end_x"] if "week_end_x" in row_dict else row_dict["week_end"]
        )
        hh_id = row_dict["household_code"]

        if (store, week) not in choice_set_matrix:
            continue

        counts = counts_dict.get((hh_id, week), np.zeros(3, dtype=np.int64))
        if np.sum(counts) == 0:
            continue

        inc_val = hh_dict.get(hh_id, 0.0)
        prices = choice_set_matrix[(store, week)][:3, 0]
        resids = choice_set_matrix[(store, week)][:3, 1]

        c_inter = np.array(
            [
                row_dict.get("C_sat_0", 0.0),
                row_dict.get("C_sat_1", 0.0),
                row_dict.get("C_sat_2", 0.0),
            ],
            dtype=np.float64,
        )

        perms = generate_basket_permutations(counts)

        trip_data.append(
            {
                "prices": prices,
                "resids": resids,
                "c_inter": c_inter,
                "income": inc_val,
                "permutations": perms,
            }
        )

    return trip_data


def total_objective_sequential(params, trip_data):
    """
    Sequential Intra-Trip + Volume-Weighted Inter-Trip Satiation Log-Likelihood
    Parameter Order (9 parameters):
    0: Constant
    1..2: Beta_Berry, Beta_Plain (vs Other Base = 0.0)
    3..4: Beta_Berry x Inc, Beta_Plain x Inc
    5: Gamma_Inter (Inter-Trip Lattin Satiation)
    6: Gamma_Intra (Intra-Trip Basket Satiation)
    7: Price (alpha)
    8: Control Func (sigma)
    """
    const = params[0]
    b0 = params[1:3]
    b_inc = params[3:5]
    g_inter = params[5]
    g_intra = params[6]
    alpha = params[7]
    sigma = params[8]

    total_ll = 0.0

    for trip in trip_data:
        prices = trip["prices"]
        resids = trip["resids"]
        c_inter = trip["c_inter"]
        inc = trip["income"]
        perms = trip["permutations"]

        beta_non_other = b0 + b_inc * inc
        beta_vec = np.append(0.0, beta_non_other)

        # Baseline Utility V_base before item draws
        v_base = (
            const
            + beta_vec
            + g_inter * np.log(1.0 + np.maximum(0.0, c_inter))
            + alpha * prices
            + sigma * resids
        )

        perm_lls = []

        for seq in perms:
            c_intra = np.zeros(3, dtype=np.float64)
            seq_log_prob = 0.0

            for choice in seq:
                # Total Satiation Utility Shifter
                v_draw = v_base + g_intra * c_intra
                log_probs = v_draw - logsumexp(v_draw)

                seq_log_prob += log_probs[choice]
                c_intra[choice] += 1.0  # Increment intra-basket counter

            perm_lls.append(seq_log_prob)

        # Average log-likelihood across all sequence permutations for this basket
        trip_ll = logsumexp(perm_lls) - np.log(len(perm_lls))
        total_ll += trip_ll

    return -total_ll if np.isfinite(total_ll) else 1e10


def estimate_sequential_model(trip_data):
    total_params = 9
    x0 = np.zeros(total_params)
    x0[0] = 0.5
    x0[1] = 0.5
    x0[2] = -0.5
    x0[5] = -0.1  # Inter-trip gamma initial guess
    x0[6] = -0.2  # Intra-trip gamma initial guess
    x0[7] = -0.5  # Price alpha guess

    bounds = (
        [(-20.0, 20.0)] * 5
        + [(-10.0, 0.0)]  # Enforce g_inter <= 0
        + [(-10.0, 0.0)]  # Enforce g_intra <= 0
        + [(-30.0, -0.01), (-30.0, 30.0)]  # Price + Control function
    )

    res = minimize(
        total_objective_sequential,
        x0=x0,
        args=(trip_data,),
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

            f1 = total_objective_sequential(x1, trip_data)
            f2 = total_objective_sequential(x2, trip_data)
            f3 = total_objective_sequential(x3, trip_data)
            f4 = total_objective_sequential(x4, trip_data)

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
        title="SEQUENTIAL INTRA + INTER TRIP INSIDE CHOICE MODEL",
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
        "gamma_Inter (Inter-Trip Lattin Satiation)",
        "gamma_Intra (Intra-Trip Basket Satiation)",
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
        "\n[bold yellow]--- Preprocessing Data for Sequential Model ---[/bold yellow]"
    )
    trip_data = load_and_preprocess()

    console.print("\n[bold yellow]--- Estimating Model ---[/bold yellow]")
    res_model = estimate_sequential_model(trip_data)
    display_results(res_model)


if __name__ == "__main__":
    main()
