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
from scipy.special import logsumexp, expit
import statsmodels.formula.api as smf

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

# =================================================
# 2. DATA / INPUT HANDLING
# =================================================


def load_and_preprocess():
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
        .collect()
        .to_pandas()
    )

    merged_df["flavor_str"] = merged_df["flavor"].fillna("").astype(str)
    merged_df["flavor_cd"] = pd.to_numeric(
        merged_df["flavor_cd"], errors="coerce"
    ).fillna(0)

    merged_master = merged_df.copy()
    merged_master["flavor"] = np.select(
        [
            merged_master["flavor_str"].str.contains("berry", case=False, na=False),
            merged_master["flavor_cd"].isin([67676592, 66987057]),
        ],
        [1, 2],
        default=0,
    )

    yogurt_df = merged_master[merged_master["yogurt_purchase"] == 1]
    yogurt_df = yogurt_df[
        yogurt_df["serving_per_container_cd"].isin([65622705, 67181961])
    ]

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
        "price ~ price_iv + size1_amount + C(week_end)",
        data=merged_master,
        missing="drop",
    ).fit()

    merged_master["iv_res"] = iv_res.resid
    merged_master = merged_master.dropna(subset=["iv_res"])

    def map_flavor_category(flavor):
        if pd.isna(flavor):
            return "outside"
        elif flavor == 0:
            return "other"
        elif flavor == 1:
            return "berry"
        elif flavor == 2:
            return "plain"

    def get_modal_flavor(group):
        if group.empty:
            return np.nan
        counts = group["flavor"].value_counts()
        if counts.empty:
            return np.nan
        max_count = counts.max()
        top_modes = counts[counts == max_count].index.to_numpy()
        return rng.choice(top_modes)

    # ---------------------------------------------
    # State Update (theta)
    # ---------------------------------------------

    trip_level = merged_master.copy()
    trip_level["category_chosen"] = trip_level["flavor"].map(map_flavor_category)
    trip_level = trip_level.drop_duplicates(subset=["household_code", "trip_code_uc"])

    modal_choices = (
        trip_level.sort_values(["household_code", "week_end", "trip_code_uc"])
        .groupby(["household_code", "trip_code_uc", "week_end"])
        .apply(get_modal_flavor)
        .reset_index(name="modal_x")
    )

    modal_choices = modal_choices.sort_values(["household_code", "week_end"])
    modal_choices["theta_prev"] = (
        modal_choices.groupby("household_code")["modal_x"].shift(1).fillna(0.0)
    )

    # -------------------------------------------
    # Choice Set Construction
    # -------------------------------------------

    merged_master["category"] = merged_master["flavor"].map(map_flavor_category)

    cat_choice_sets = (
        merged_master.groupby(["store_code_uc", "week_end", "category"])
        .agg(price=("price", "mean"), iv_res=("iv_res", "mean"))
        .reset_index()
    )

    choice_set_matrix = {}
    for (store, week), group in cat_choice_sets.groupby(["store_code_uc", "week_end"]):
        mat = np.zeros((4, 2))
        mean_store_prices = group["price"].mean()
        mat[:3, 0] = mean_store_prices

        for row in group.itertuples():
            if row.category in cat_map and row.category != "outside":
                idx = cat_map[row.category]
                res_val = 0.0 if np.isnan(row.iv_resid) else row.iv_resid
                mat[idx] = [row.price, res_val]
        choice_set_matrix[(store, week)] = mat

    # ------------------------------------------
    # Data Packing
    # ------------------------------------------

    trips_processed = trip_level.merge(
        modal_choices[["household_code", "trip_code_uc", "theta_prev"]],
        on=["household_code", "trip_code_uc"],
        how="left",
    )
    trips_processed["choice_idx"] = (
        trips_processed["category_chosen"].map(cat_map).fillna(3).astype(np.int64)
    )

    hh_packed_data = {}
    for hh_id, group in trips_processed.groupby("household_code"):
        stores = group["store_code_uc"].to_numpy()
        weeks = group["week_end"].to_numpy()
        choices = group["choice_idx"].to_numpy(dtype=np.float64)
        thetas = group["theta_prev"].to_numpy(dtype=np.float64)

        valid_mask = np.array(
            [(s, w) in choice_set_matrix for s, w in zip(stores, weeks)]
        )
        if not valid_mask.any():
            continue

        hh_packed_data[hh_id] = {
            "matrices": [
                choice_set_matrix[(s, w)]
                for s, w in zip(stores[valid_mask], weeks[valid_mask])
            ],
            "choices": choices[valid_mask],
            "thetas": thetas[valid_mask],
        }

    return hh_packed_data


# =================================================
# CORE LOGIC / PROCESSING
# =================================================


def total_objective(params, hh_packed_data):
    const, beta_ber, beta_pl, gamma, alpha, sigma = params

    d_berry = np.array([0.0, 1.0, 0.0])
    d_plain = np.array([0.0, 0.0, 1.0])
    cat_flavors = np.array([0.0, 1.0, 2.0])

    total_ll = 0.0

    for hh_data in hh_packed_data.values():
        matrices = hh_data["matrices"]
        choices = hh_data["choices"]
        thetas = hh_data["thetas"]

        for X_mat, y_idx, theta in zip(matrices, choices, thetas):
            prices = X_mat[:, 0]
            resids = X_mat[:, 1]

            Xi = np.abs(cat_flavors - theta)

            u = np.zeros(4)
            u[:3] = (
                const
                + beta_ber * d_berry
                + beta_pl * d_plain
                + gamma * Xi
                + alpha * prices[:3]
                + sigma * resids[:3]
            )
            u[3] = 0.0

            u_max = np.max(u)
            log_prob = u[y_idx] - logsumexp(u - u_max)

            if not np.isfinite(log_prob):
                log_prob = -700.0

            total_ll += log_prob
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
        args=(hh_packed_data),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-8},
    )

    cov_matrix = res.hess_inv.todense()
    se = np.sqrt(np.diag(cov_matrix))
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
        title="LOV SIMPLE SPEC RESULTS",
        show_header=True,
        header_style="bold_magenta",
    )
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Estimate", justify="right")
    table.add_column("Std. Error", justify="right")
    table.add_column("z-stat", justify="right")
    table.add_column("p-value", justify="right")

    param_names = [
        "beta_0",
        "beta_ber",
        "beta_pl",
        "LOV",
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

        alpha = results["params"][4]
        wtp_results = {
            "Intercept": -1 * (results["params"][0] / alpha),
            "Berry Flavor": -1 * (results["params"][1] / alpha),
            "Plain": -1 * (results["params"] / alpha),
            "LOV": -1 * (results["params"][3] / alpha),
        }
        console.print(
            "\n[bold yellow]WTP Relative to Other Flavors Relative to Outside:[/bold yellow]"
        )
        for param, wtp in wtp_results.items():
            console.print(f"  {param}: ${wtp:.4f}")


def main():
    hh_packed_data = load_and_preprocess()
    results = estimate_model(hh_packed_data)
    display_results(results)


if __name__ == "__main__":
    main()
