"""Panel flavor demand with persistent tastes and stable own-flavor exposure.

Read est_step4.md before interpreting estimates. This is conditional flavor
demand: total yogurt purchases and the visited store are taken as given. It is
not a model of yogurt participation, consumption, store choice, or inventory.
Default heterogeneity is observed household income, with no simulated draws.
Use --heterogeneity random for the previous normal random-coefficient model.
Nothing executes on import; outputs remain within this script's folder.
Run ``python3 est_step4.py --help`` or ``--self-test`` for entry points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/scanner_panel.parquet"
)
MIN_PURCHASE_WEEKS = 2
# Match the midpoint mapping already used in est_step3.py. These are bracket
# representatives, not exact incomes; the open upper bracket is set to $100,000.
INCOME_CODE_MAP = {3: 2500, 4: 6500, 6: 9000, 8: 11000, 10: 13500,
                   11: 17500, 13: 22500, 15: 27500, 16: 32500, 17: 37500,
                   18: 42500, 19: 47500, 21: 55000, 23: 65000,
                   26: 85000, 27: 100000}


def input_path(value: str) -> Path:
    """Allow external data; resolve relative input paths against the script folder."""
    return (ROOT / value).resolve()


def local_path(value: str) -> Path:
    """Resolve symlinks too: a path spelled locally must actually remain local."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must stay inside {ROOT}: {value}")
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Use a CSV or Parquet input.")


def dates(values: pd.Series) -> pd.Series:
    # Nielsen dates can be YYYYMMDD integers. Direct pd.to_datetime(integer)
    # would interpret them as nanoseconds since 1970, silently corrupting gaps.
    return pd.to_datetime(
        values.astype(str).str.replace(r"\.0$", "", regex=True),
        format="mixed",
        errors="raise",
    ).dt.normalize()


def family(flavor: str) -> str:
    """A deliberately coarse comparison to exact-flavor exposure, not a taxonomy."""
    # These are the draft's established category codes, not raw Nielsen codes.
    if flavor.strip() in {"0", "1", "2"}:
        return {"0": "other", "1": "berry", "2": "plain"}[flavor.strip()]
    flavor = flavor.lower()
    if "plain" in flavor or "unflavored" in flavor:
        return "plain"
    if any(
        word in flavor for word in ["berry", "straw", "blue", "rasp", "cherry", "cran"]
    ):
        return "berry"
    return "other"


@dataclass
class Panel:
    # One row per household-store-week. Counts belong ONLY to that store.
    # All stores within the same household-week receive the same pre-week stock;
    # their purchases are summed once to update next week's exposure.
    household: str
    weeks: np.ndarray
    counts: np.ndarray
    prices: np.ndarray
    available: np.ndarray
    controls: np.ndarray
    stock: np.ndarray
    family_stock: np.ndarray
    keep: np.ndarray
    # Prices are completed by the documented mean-price fallback before fitting.
    price_complete: np.ndarray | None = None
    income: float = 0.0
    income_missing: bool = False


@dataclass
class Dataset:
    panels: list[Panel]
    flavors: list[str]
    control_names: list[str]
    family_index: np.ndarray
    delta: float
    burnin: int
    attrition: pd.DataFrame | None = None
    income_table: pd.DataFrame | None = None
    income_metadata: dict | None = None


def stocks(weeks, counts, family_index, delta):
    """Construct PRE-choice stock from servings, never shares or previous states.

    At consecutive purchase weeks a and b, S_b = delta**gap * (S_a + Q_a).
    This convention places new servings in stock at the end of week a, and lets
    them decay during the gap. There is no within-week ordering assumption.
    Gaps without yogurt purchases decay stock but do not create invented choices.
    """
    own = np.zeros_like(counts, dtype=float)
    broad = np.zeros_like(own)
    running = np.zeros(counts.shape[1])
    previous = None
    for week in np.unique(weeks):
        rows = np.flatnonzero(weeks == week)
        if previous is not None:
            gap = float((week - previous) / np.timedelta64(1, "D")) / 7
            running *= delta**gap
        own[rows] = running
        for j in range(counts.shape[1]):
            broad[rows, j] = running[family_index == family_index[j]].sum()
        running += counts[rows].sum(axis=0)
        previous = week
    return own, broad


def prepare(args) -> tuple[Dataset, dict]:
    purchases = read_table(input_path(args.input))
    sample_stages = []

    def sample_stage(label, frame):
        sample_stages.append({"stage": label, "source_rows": len(frame),
                              "households": int(frame.household_code.nunique())})
    required = [
        "household_code",
        "store_code_uc",
        "week_end",
        "quantity",
        "price",
        args.flavor_column,
    ]
    missing = set(required) - set(purchases)
    if missing:
        raise ValueError(f"Missing purchase columns: {sorted(missing)}")
    sample_stage("input", purchases)

    # Apply the established single-person / single-serving / yogurt restrictions.
    # Missing filter columns require explicit confirmation via a CLI flag: a
    # prefiltered dataset is valid, silently assuming one is not.
    filters = {
        "household_size": [1],
        "serving_per_container_cd": [67181961, 65622705],
        "product_module_code_hms": [3612, 3603],
    }
    for column, allowed in filters.items():
        if column in purchases:
            purchases = purchases[
                pd.to_numeric(purchases[column], errors="coerce").isin(allowed)
            ]
        elif not args.prefiltered:
            raise ValueError(
                f"Missing {column}; use --prefiltered only for an already restricted sample."
            )
        sample_stage(f"after_{column}_restriction", purchases)

    price_diagnostics = {}

    def normalize(frame, is_purchase):
        frame = frame.copy()
        for column in ["store_code_uc", "week_end", "price", args.flavor_column]:
            if column not in frame:
                raise ValueError(f"Missing menu column: {column}")
        if frame[["store_code_uc", args.flavor_column]].isna().any().any():
            raise ValueError(
                "Missing store/flavor identifiers; clean them explicitly before estimation."
            )
        frame["week_end"] = dates(frame["week_end"])
        frame["store_code_uc"] = frame["store_code_uc"].astype(str)
        frame["flavor_label"] = (
            frame[args.flavor_column].astype(str).str.strip().str.upper()
        )
        if frame["flavor_label"].eq("").any():
            raise ValueError(
                "Blank flavors must be cleaned, not treated as an outside alternative."
            )
        # Scanner merges may leave missing prices; zero, negative, infinite,
        # and malformed values are also not usable prices. Keep the purchases:
        # dropping their quantities would falsely lower subsequent exposure.
        price = pd.to_numeric(frame["price"], errors="coerce")
        valid_price = np.isfinite(price) & (price > 0)
        source = "purchase" if is_purchase else "menu"
        price_diagnostics[f"{source}_rows_with_unusable_price"] = int((~valid_price).sum())
        frame["price"] = price.where(valid_price, np.nan)
        if (~valid_price).any():
            print(f"{source}: {int((~valid_price).sum()):,}/{len(frame):,} rows have missing/invalid prices; preserving quantities and averaging only valid prices.", flush=True)
        if is_purchase:
            if frame["household_code"].isna().any():
                raise ValueError("Missing household identifiers.")
            frame["household_code"] = frame["household_code"].astype(str)
            q = pd.to_numeric(frame["quantity"], errors="raise").to_numpy(float)
            if (
                not np.isfinite(q).all()
                or (q <= 0).any()
                or not np.allclose(q, np.round(q))
            ):
                raise ValueError(
                    "Quantity must be positive integer single-serving cup counts."
                )
            frame["quantity"] = np.round(q).astype(int)
        return frame

    purchases = normalize(purchases, True)
    if purchases.empty:
        raise ValueError("No purchases after sample restrictions.")
    # With no independent menu, assume every retained category is available at
    # each observed store-week, as in step 2. A category with no sampled purchase
    # is not declared unavailable. Complete its price from observed averages.
    menu_source = (
        normalize(read_table(input_path(args.menu)), False)
        if args.menu
        else purchases.copy()
    )
    # Keep a stable, explicit exact-flavor menu. Do not aggregate rare flavors
    # into OTHER: that would redefine own-flavor exposure as a mixture of tastes.
    flavors = (
        purchases.groupby("flavor_label")["quantity"]
        .sum()
        .nlargest(args.top_flavors)
        .index.tolist()
    )
    if len(flavors) < 2:
        raise ValueError("At least two observed flavors are needed.")
    excluded_cups = int(
        purchases.loc[~purchases.flavor_label.isin(flavors), "quantity"].sum()
    )
    purchases = purchases[purchases.flavor_label.isin(flavors)].copy()
    sample_stage("after_retained_flavor_restriction", purchases)
    menu_source = menu_source[menu_source.flavor_label.isin(flavors)].copy()
    flavor_map = {f: j for j, f in enumerate(flavors)}
    family_index = np.array(
        [{"plain": 0, "berry": 1, "other": 2}[family(f)] for f in flavors]
    )

    # Flavor-level utilities control for the menu's brand/style composition.
    # Categorical codes are dummies, not numeric slopes on arbitrary labels.
    # Menu rows should be one product listing per store/week (not transactions).
    # In the purchase fallback these proportions are transaction-weighted, a
    # measurement limitation; they are not consumer-specific chosen attributes.
    control_names = []
    for column in args.attribute_columns:
        if column not in menu_source:
            raise ValueError(
                f"Requested brand/style attribute absent from menu: {column}"
            )
        values = menu_source[column].fillna("MISSING").astype(str)
        levels = sorted(values.unique())
        for level in levels[1:]:
            name = f"{column}={level}"
            menu_source[name] = values.eq(level).astype(float)
            control_names.append(name)
    key = ["store_code_uc", "week_end", "flavor_label"]
    menu = menu_source.groupby(key, observed=True)[["price"] + control_names].mean()
    store_means = menu.groupby(level=["store_code_uc", "flavor_label"]).mean()
    flavor_means = menu.groupby(level="flavor_label").mean()
    grand_means = menu.mean()
    if not np.isfinite(grand_means["price"]):
        raise ValueError("No finite positive prices in the input menu/purchases; supply usable per-cup prices.")
    baskets = (
        purchases.groupby(["household_code"] + key, observed=True)
        .quantity.sum()
        .reset_index()
    )

    panels = []
    one_alternative_rows = 0
    price_fallbacks = {"store_flavor": 0, "overall_flavor": 0, "overall_all_flavors": 0}
    household_audit = []
    for household, frame in baskets.groupby("household_code", sort=True):
        row_keys = (
            frame[["week_end", "store_code_uc"]]
            .drop_duplicates()
            .sort_values(["week_end", "store_code_uc"])
        )
        n, j = len(row_keys), len(flavors)
        counts = np.zeros((n, j), dtype=int)
        prices = np.zeros((n, j))
        available = np.zeros((n, j), dtype=bool)
        price_complete = np.ones(n, dtype=bool)
        controls = np.zeros((n, j, len(control_names) + 2))
        weeks = row_keys.week_end.to_numpy(dtype="datetime64[ns]")
        for t, row in enumerate(row_keys.itertuples(index=False)):
            try:
                offer = menu.loc[(row.store_code_uc, row.week_end)]
            except KeyError as error:
                raise ValueError(
                    f"No menu for household {household}, {row.store_code_uc}, {row.week_end}"
                ) from error
            if not args.menu:
                # Missing sampled sales do not establish missing shelf availability.
                offer = offer.reindex(flavors)
            for label, values in offer.iterrows():
                k = flavor_map[label]
                available[t, k] = True
                completed = values.copy()
                # Prefer local prices. The final all-flavor fallback is used only
                # if a flavor has no positive observed price anywhere, and is
                # counted separately so this strong imputation stays visible.
                fallbacks = []
                if (row.store_code_uc, label) in store_means.index:
                    fallbacks.append(("store_flavor", store_means.loc[(row.store_code_uc, label)]))
                if label in flavor_means.index:
                    fallbacks.append(("overall_flavor", flavor_means.loc[label]))
                fallbacks.append(("overall_all_flavors", grand_means))
                for source, replacement in fallbacks:
                    if not np.isfinite(completed["price"]) and np.isfinite(replacement["price"]):
                        price_fallbacks[source] += 1
                    completed = completed.fillna(replacement)
                prices[t, k] = completed["price"]
                controls[t, k, : len(control_names)] = completed[control_names].fillna(0).to_numpy(float)
            selected = frame[
                (frame.week_end == row.week_end)
                & (frame.store_code_uc == row.store_code_uc)
            ]
            for purchase in selected.itertuples():
                counts[t, flavor_map[purchase.flavor_label]] += purchase.quantity
            if np.any((counts[t] > 0) & ~available[t]):
                raise ValueError(
                    "A purchased flavor is absent from its store-week menu."
                )
        own, broad = stocks(weeks, counts, family_index, args.delta)
        unique = np.unique(weeks)
        # Require two DISTINCT purchase weeks, not two store visits or baskets.
        # Eligible households retain all rows, including the first week; there
        # is no burn-in and the purchase weeks need not be consecutive.
        keep = np.ones(n, dtype=bool)
        audit = {"household": household, "all_store_week_rows": n,
                 "all_purchase_weeks": len(unique)}
        # If a supplied menu has only one category, keep the row: its conditional
        # log probability is zero, so it adds history but no taste information.
        one_alternative_rows += int(np.sum(available.sum(axis=1) < 2))
        eligible = len(unique) >= MIN_PURCHASE_WEEKS
        audit["retained_rows"] = n if eligible else 0
        audit["retained_weeks"] = len(unique) if eligible else 0
        audit["retained"] = eligible
        household_audit.append(audit)
        if not eligible:
            continue
        # Same-flavor exposure interacted with lagged total cups / elapsed gap
        # lets dose/timing effects differ across alternatives. A common scalar
        # gap or total-inventory regressor cancels in a conditional flavor logit.
        lag_total, gap = history_controls(weeks, counts)
        controls[:, :, -2] = np.log1p(own) * lag_total[:, None]
        controls[:, :, -1] = np.log1p(own) * gap[:, None]
        panels.append(
                Panel(
                    household,
                    weeks,
                    counts,
                    prices,
                    available,
                    controls,
                    own,
                    broad,
                    keep,
                    price_complete,
                )
        )
    if not panels:
        raise ValueError(
            "No households have at least two distinct yogurt-purchase weeks after product/sample filters."
        )
    names = control_names + [
        "own_log_stock_x_log_previous_total",
        "own_log_stock_x_log_gap_weeks",
    ]
    # Timing interactions are optional because they compete directly with gamma.
    # The default baseline estimates one exposure slope without these terms.
    if not args.timing_controls:
        panels = [replace(p, controls=p.controls[:, :, :-2]) for p in panels]
        names = control_names
    attrition = pd.DataFrame(household_audit)
    stage_totals = [{"stage": "before_minimum_purchase_weeks",
                     "store_week_rows": int(attrition.all_store_week_rows.sum()),
                     "households_with_any_rows": len(attrition)},
                    {"stage": "after_minimum_purchase_weeks",
                     "store_week_rows": int(attrition.retained_rows.sum()),
                     "households_with_any_rows": len(panels)}]
    data = Dataset(panels, flavors, names, family_index, args.delta, 0, attrition)
    if getattr(args, "heterogeneity", "random") == "income":
        data = attach_income(data, purchases, args.income_column, args.income_encoding)
    report = {
        "households": len(panels),
        "source_sample_stages": sample_stages,
        "choice_sample_stages": stage_totals,
        "flavors": flavors,
        "reference_flavor": flavors[0],
        "likelihood_rows": sum(int(p.keep.sum()) for p in panels),
        "likelihood_cups": sum(int(p.counts[p.keep].sum()) for p in panels),
        "excluded_nonselected_flavor_cups": excluded_cups,
        "one_alternative_rows_before_household_filter": one_alternative_rows,
        "otherwise_informative_rows_excluded_for_prices": 0,
        "price_fallback_cells": price_fallbacks,
        "price_cleaning": price_diagnostics,
        "price_missing_policy": "valid store-week-flavor mean, then store-flavor mean, overall flavor mean, and overall all-flavor mean; no purchase rows excluded",
        "menu_source": "independent supplied menu"
        if args.menu
        else "all retained categories assumed available; purchase-derived mean prices",
        "price_unit": "assumed dollars per single-serving cup; user must verify",
        "attribute_controls": names,
        "delta": args.delta,
        "burnin_weeks": 0,
        "minimum_purchase_weeks": MIN_PURCHASE_WEEKS,
        "households_excluded_for_purchase_weeks": int((~attrition.retained).sum()),
    }
    return data, report


def attach_income(data, purchases, column, encoding):
    """Attach one fixed income covariate per retained household without dropping any.

    Choose the earliest observed usable income, rather than letting within-panel
    income changes masquerade as short-run taste dynamics. Standardization uses
    one observation per RETAINED household, not one per cup or purchase row.
    Missing/unknown incomes get h=0 plus explicit missing-income flavor/history
    terms. They are not claimed to be observed average-income households.
    """
    if column not in purchases:
        raise ValueError(f"Income specification requires {column}; choose --income-column or --heterogeneity random.")
    source = purchases[purchases.household_code.isin([p.household for p in data.panels])].copy()
    raw = pd.to_numeric(source[column], errors="coerce")
    dollars = raw.map(INCOME_CODE_MAP) if encoding == "codes" else raw
    dollars = dollars.where(np.isfinite(dollars) & (dollars > 0))
    source["income_dollars"] = dollars
    earliest = source.sort_values("week_end").groupby("household_code").income_dollars.first()
    changes = source.groupby("household_code").income_dollars.nunique()
    known = np.log(earliest.dropna())
    if len(known) < 2 or known.nunique() < 2:
        raise ValueError("Income interactions require at least two distinct usable household incomes; check bracket codes/--income-encoding.")
    center, scale = float(known.mean()), float(known.std(ddof=0))
    panels, rows = [], []
    for p in data.panels:
        value = earliest.loc[p.household]
        missing = bool(pd.isna(value))
        h = 0.0 if missing else float((np.log(value) - center) / scale)
        panels.append(replace(p, income=h, income_missing=missing))
        rows.append({"household": p.household, "income_dollars": None if missing else float(value),
                     "income_standardized_log": h, "income_missing": missing})
    metadata = {"column": column, "encoding": encoding, "log_income_mean": center,
                "log_income_sd": scale, "standardization": "equal weights across retained households with usable income",
                "missing_income_households": sum(p.income_missing for p in panels),
                "households_with_changing_usable_income": int((changes > 1).sum()),
                "income_timing": "earliest observed usable income held fixed throughout household panel",
                "missing_policy": "h=0 and separate missing-income preference/fatigue terms",
                "top_bracket_dollars": 100000 if encoding == "codes" else None,
                "unknown_numeric_codes": sorted(float(v) for v in raw.dropna().unique() if v not in INCOME_CODE_MAP)
                if encoding == "codes" else []}
    return replace(data, panels=panels, income_table=pd.DataFrame(rows), income_metadata=metadata)


def history_controls(weeks, counts):
    lag_total = np.zeros(len(weeks))
    gap = np.zeros(len(weeks))
    previous, previous_total = None, 0
    for week in np.unique(weeks):
        rows = weeks == week
        lag_total[rows] = np.log1p(previous_total)
        if previous is not None:
            gap[rows] = np.log1p(float((week - previous) / np.timedelta64(1, "D")) / 7)
        previous_total = counts[rows].sum()
        previous = week
    return lag_total, gap


class Model:
    """Fixed simulated draws, persistent over ALL rows of one household.

    beta_reference = 0 is a location normalization of taste differences. Gamma
    applies to every flavor including the reference; it is not normalized away.
    SDs are optimized AND reported in their natural units, with zero allowed.
    Independent normal taste differences are a parsimonious starting covariance
    structure. They do not assert independence of observed choices over time.
    """

    def __init__(
        self, data, draws=128, seed=219, gamma_mode="common", family_effect=False
    ):
        self.data, self.j = data, len(data.flavors)
        self.mode, self.family_effect = gamma_mode, family_effect
        dimensions = self.j - 1 + int(gamma_mode == "random")
        sampler = qmc.Sobol(dimensions, scramble=True, seed=seed)
        # Round UP to a power of two to preserve Sobol balance.
        self.draws = ndtri(
            np.clip(sampler.random_base2(int(np.ceil(np.log2(draws)))), 1e-9, 1 - 1e-9)
        )
        self.r = len(self.draws)
        self.g = self.j if gamma_mode == "flavor" else 1
        self.names = (
            [f"beta[{f}]" for f in data.flavors[1:]]
            + [f"sd_beta[{f}]" for f in data.flavors[1:]]
            + ([f"gamma[{f}]" for f in data.flavors] if self.g > 1 else ["gamma"])
            + (["sd_gamma"] if gamma_mode == "random" else [])
            + (["gamma_family"] if family_effect else [])
            + ["alpha_price"]
            + data.control_names
        )
        self.gamma_start = 2 * (self.j - 1)
        self.alpha_index = (
            self.gamma_start + self.g + int(gamma_mode == "random") + int(family_effect)
        )
        # Signed mean gamma is intentionally allowed: the data may indicate
        # reinforcement instead of fatigue. A positive mean means fatigue.
        self.bounds = (
            [(-12, 12)] * (self.j - 1)
            + [(0, 4)] * (self.j - 1)
            + [(-8, 8)] * self.g
            + ([(0, 4)] if gamma_mode == "random" else [])
            + ([(-8, 8)] if family_effect else [])
            + [(-10, 0)]
            + [(-8, 8)] * len(data.control_names)
        )

    def unpack(self, x):
        k = self.j - 1
        beta = np.column_stack(
            [np.zeros(self.r), x[:k] + self.draws[:, :k] * x[k : 2 * k]]
        )
        gamma = np.broadcast_to(x[2 * k : 2 * k + self.g], (self.r, self.j)).copy()
        cursor = 2 * k + self.g
        if self.mode == "random":
            gamma += x[cursor] * self.draws[:, -1, None]
            cursor += 1
        broad = x[cursor] if self.family_effect else 0.0
        return beta, gamma, broad, x[self.alpha_index], x[self.alpha_index + 1 :]

    def log_probabilities(self, x, p):
        beta, gamma, broad, alpha, coefficients = self.unpack(x)
        utility = (
            beta[None, :, :]
            + alpha * p.prices[:, None, :]
            - gamma[None, :, :] * np.log1p(p.stock[:, None, :])
            - broad * np.log1p(p.family_stock[:, None, :])
            + (p.controls @ coefficients)[:, None, :]
        )
        utility = np.where(p.available[:, None, :], utility, -np.inf)
        return utility - logsumexp(utility, axis=2, keepdims=True)

    def objective(self, x):
        ll = 0.0
        for p in self.data.panels:
            logp = self.log_probabilities(x, p)[p.keep]
            # Unavailable alternatives have log P = -inf and zero counts.
            # Set their log probabilities to zero BEFORE multiplying, avoiding
            # the undefined 0 * -inf operation without clipping probabilities.
            logp = np.where(p.available[p.keep, None, :], logp, 0.0)
            draw_ll = (p.counts[p.keep, None, :] * logp).sum(axis=(0, 2))
            ll += logsumexp(draw_ll) - np.log(self.r)
        # Multinomial coefficients are constant in parameters. Omitting them
        # does not change estimates or profile differences.
        return -float(ll)

    def starting_vector(self):
        base = np.zeros(len(self.names))
        base[self.j - 1 : 2 * (self.j - 1)] = 0.3
        base[self.gamma_start : self.gamma_start + self.g] = 0.2
        base[self.alpha_index] = -0.5
        return base

    def fit(
        self,
        starts=3,
        initial=None,
        fixed_gamma=None,
        maxiter=400,
        seed=17,
        no_fatigue=False,
    ):
        rng = np.random.default_rng(seed)
        base = self.starting_vector()
        if initial is not None:
            base = initial.copy()
        fixed = {} if fixed_gamma is None else {self.gamma_start: fixed_gamma}
        if no_fatigue:
            # Gamma's mean = 0 alone is NOT a no-fatigue model when gamma has a
            # random SD or a separate family slope. Fix every history slope.
            for k in range(self.gamma_start, self.alpha_index):
                fixed[k] = 0.0
            for c, name in enumerate(self.data.control_names):
                if name.startswith("own_log_stock_x_"):
                    fixed[self.alpha_index + 1 + c] = 0.0
        free = np.array([k for k in range(len(base)) if k not in fixed])
        candidates = []
        for attempt in range(starts):
            x = base.copy()
            if attempt:
                x += rng.normal(0, 0.15, len(x))
            for k, (low, high) in enumerate(self.bounds):
                x[k] = np.clip(x[k], low, high)
            for k, value in fixed.items():
                x[k] = value

            def expand(v):
                full = x.copy()
                full[free] = v
                return full

            result = minimize(
                lambda v: self.objective(expand(v)),
                x[free],
                jac=(lambda v: self.gradient(expand(v))[free]) if hasattr(self, "gradient") else None,
                bounds=[self.bounds[k] for k in free],
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-5},
            )
            result.x = expand(result.x)
            candidates.append(result)
        finite = [r for r in candidates if np.isfinite(r.fun)]
        if not finite:
            raise RuntimeError(
                "All optimization starts returned nonfinite likelihoods."
            )
        best = min(finite, key=lambda r: r.fun)
        best.start_objectives = [float(r.fun) for r in candidates]
        return best


class IncomeModel(Model):
    """Observed heterogeneity with NO quadrature or latent random coefficients.

    beta_ij = b_j + c_j*h_i, gamma_i = g0 + g1*h_i. Reference b/c are
    both zero. Gamma affects every category, including the reference. A single
    scalar income main effect shared by all alternatives would cancel, so income
    enters flavor intercepts and the exposure slope instead.

    Missing income households remain, with separate flavor and fatigue effects.
    The exact design matrix is shared by utility, gradient, rank diagnostics,
    and clustered inference so the interpretation cannot drift between them.
    """
    def __init__(self, data, family_effect=False):
        self.data, self.j = data, len(data.flavors)
        self.mode, self.family_effect, self.g = "income", family_effect, 1
        self.r = 1  # one deterministic utility per household, NOT a simulation draw
        self.draws = np.empty((1, 0))
        self.has_missing = any(p.income_missing for p in data.panels)
        k = self.j - 1
        self.names = ([f"beta[{f}]" for f in data.flavors[1:]]
                      + [f"beta_income[{f}]" for f in data.flavors[1:]]
                      + ([f"beta_missing_income[{f}]" for f in data.flavors[1:]] if self.has_missing else []))
        self.gamma_start = len(self.names)
        self.names += ["gamma", "gamma_income"]
        if self.has_missing:
            self.names += ["gamma_missing_income"]
        if family_effect:
            self.names += ["gamma_family"]
        self.alpha_index = len(self.names)
        self.names += ["alpha_price"] + data.control_names
        self.bounds = ([(-12, 12)] * (2*k + k*int(self.has_missing))
                       + [(-8, 8)] * (self.alpha_index-self.gamma_start)
                       + [(-10, 0)] + [(-8, 8)] * len(data.control_names))
        self._designs = [self.design(p) for p in data.panels]

    def starting_vector(self):
        base = np.zeros(len(self.names))
        base[self.gamma_start] = .1
        base[self.alpha_index] = -.5
        return base

    def design(self, p):
        x = np.zeros((len(p.weeks), self.j, len(self.names)))
        k = self.j - 1
        for j in range(1, self.j):
            x[:, j, j-1] = 1
            x[:, j, k+j-1] = p.income
            if self.has_missing:
                x[:, j, 2*k+j-1] = float(p.income_missing)
        z = np.log1p(p.stock)
        x[:, :, self.gamma_start] = -z
        x[:, :, self.gamma_start+1] = -p.income*z
        cursor = self.gamma_start+2
        if self.has_missing:
            x[:, :, cursor] = -float(p.income_missing)*z
            cursor += 1
        if self.family_effect:
            x[:, :, cursor] = -np.log1p(p.family_stock)
        x[:, :, self.alpha_index] = p.prices
        x[:, :, self.alpha_index+1:] = p.controls
        return x

    def log_probabilities(self, params, p):
        utility = self.design(p) @ params
        utility = np.where(p.available, utility, -np.inf)
        logp = utility-logsumexp(utility, axis=1, keepdims=True)
        return logp[:, None, :]

    def components(self, params, information=False):
        ll, scores = 0.0, []
        hessian = np.zeros((len(params), len(params)))
        for p, full_x in zip(self.data.panels, self._designs):
            x, y, available = full_x[p.keep], p.counts[p.keep], p.available[p.keep]
            utility = np.where(available, x @ params, -np.inf)
            logp = utility-logsumexp(utility, axis=1, keepdims=True)
            probabilities = np.exp(logp)
            ll += float((y*np.where(available, logp, 0)).sum())
            total = y.sum(axis=1)
            scores.append(np.einsum("tj,tjk->k", y-total[:, None]*probabilities, x))
            if information:
                # Exact curvature: sum N_t * Cov_P(X_tj). Unlike a finite-
                # difference Hessian this cannot invent curvature from rounding.
                means = np.einsum("tj,tjk->tk", probabilities, x)
                hessian += np.einsum("t,tj,tjk,tjl->kl", total, probabilities, x, x)
                hessian -= np.einsum("t,tk,tl->kl", total, means, means)
        return -ll, np.array(scores), hessian

    def objective(self, params):
        return self.components(params)[0]

    def gradient(self, params):
        return -self.components(params)[1].sum(axis=0)

    def rank_diagnostic(self):
        # Utility differences, not raw utility columns, identify conditional
        # demand. With partial supplied menus choose an available pivot per row.
        rows = []
        for p, x in zip(self.data.panels, self._designs):
            for t in np.flatnonzero(p.keep):
                offered = np.flatnonzero(p.available[t])
                rows.extend(x[t, offered[1:]]-x[t, offered[0]])
        matrix = np.array(rows).reshape(-1, len(self.names))
        norms = np.linalg.norm(matrix, axis=0)
        scaled = matrix/np.where(norms > 0, norms, 1)
        singular = np.linalg.svd(scaled, compute_uv=False)
        rank = int(np.linalg.matrix_rank(scaled)) if len(matrix) else 0
        return {"parameters": len(self.names), "relative_design_rows": len(matrix),
                "rank": rank, "full_column_rank": rank == len(self.names),
                "column_scaled_singular_values": singular.tolist(),
                "zero_columns": [name for name, norm in zip(self.names, norms) if norm == 0],
                "interpretation": "empirical linear-log-odds rank under maintained model; not proof of exogeneity"}

    def clustered_inference(self, params):
        _, scores, hessian = self.components(params, information=True)
        groups = len(scores)
        diagnostics = {"method": "household-cluster sandwich with G/(G-1) correction",
                       "clusters": groups, "hessian_rank": int(np.linalg.matrix_rank(hessian)),
                       "hessian_condition_number": float(np.linalg.cond(hessian))}
        se = np.full(len(params), np.nan)
        if groups > 1 and diagnostics["hessian_rank"] == len(params) and diagnostics["hessian_condition_number"] < 1e12:
            bread = np.linalg.inv(hessian)
            covariance = bread @ (scores.T @ scores) @ bread * groups/(groups-1)
            diagonal = np.diag(covariance)
            se = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
            # Ordinary interior standard errors are not valid at an active bound.
            active = [np.isclose(v, low, atol=1e-5) or np.isclose(v, high, atol=1e-5)
                      for v, (low, high) in zip(params, self.bounds)]
            se[active] = np.nan
        diagnostics["standard_errors_available"] = bool(np.isfinite(se).any())
        return se, diagnostics


def make_model(data, args):
    if args.heterogeneity == "income":
        return IncomeModel(data, args.family_effect)
    return Model(data, args.draws, args.seed, args.gamma_mode, args.family_effect)


def identification(data):
    """Descriptive within-household variation, not a causal identification test.

    Centering z itself can hide variation that cancels across alternatives. We
    also center z_j-z_reference within household on rows offering BOTH flavors.
    This is the variation that moves relative utility in conditional demand.
    """
    rows = []
    for j, flavor_label in enumerate(data.flavors):
        raw, within, relative, switchers = [], [], [], 0
        for p in data.panels:
            mask = p.keep & p.available[:, j]
            z = np.log1p(p.stock[mask, j])
            if len(z):
                raw.extend(z)
                within.extend(z - z.mean())
            pair = p.keep & p.available[:, j] & p.available[:, 0]
            difference = np.log1p(p.stock[pair, j]) - np.log1p(p.stock[pair, 0])
            if len(difference):
                relative.extend(difference - difference.mean())
            purchases = p.counts[p.keep].sum(axis=0)
            switchers += int(purchases[j] > 0 and np.count_nonzero(purchases) > 1)
        rows.append(
            {
                "flavor": flavor_label,
                "available_rows": len(raw),
                "mean_log_stock": float(np.mean(raw)) if raw else None,
                "overall_log_stock_sd": float(np.std(raw)) if raw else None,
                "within_household_log_stock_sd": float(np.std(within))
                if within
                else None,
                "within_household_relative_stock_sd": float(np.std(relative))
                if relative
                else None,
                "households_buying_this_and_another_flavor": switchers,
            }
        )
    return pd.DataFrame(rows)


def profile(model, fit, grid, starts, maxiter):
    if model.g != 1:
        raise ValueError(
            "One-dimensional gamma profiling requires common or random gamma mode."
        )
    rows = []
    for value in grid:
        result = model.fit(starts, fit.x, float(value), maxiter)
        print(
            f"Profile gamma={value:.3g}, negative LL={result.fun:.6f}, converged={result.success}",
            flush=True,
        )
        row = {
            "gamma_fixed": value,
            "negative_log_likelihood": result.fun,
            "converged": bool(result.success),
        }
        row.update(dict(zip(model.names, result.x)))
        rows.append(row)
    table = pd.DataFrame(rows)
    # Include the unrestricted optimum in the baseline. If a constrained fit
    # beats it, flag a local optimum rather than claiming a confidence interval.
    baseline = min(float(fit.fun), float(table.negative_log_likelihood.min()))
    table["twice_log_likelihood_loss"] = 2 * (table.negative_log_likelihood - baseline)
    table["profile_beats_unrestricted"] = table.negative_log_likelihood < fit.fun - 1e-5
    return table


def simulate(model, truth, seed):
    """Generate an actual evolving history, not choices against frozen stocks.

    Condition on observed menus, dates, and basket totals. Income mode keeps the
    observed income and has no latent type draw; random mode draws a continuous
    household type independently of estimation's quadrature nodes. Simulate
    burn-in choices too, construct new stocks, then apply the original keep mask.
    This tests conditional parameter recovery, not the IV exclusion restriction
    or a model of the endogenous purchase timing that is held fixed here.
    """
    rng = np.random.default_rng(seed)
    panels = []
    for p in model.data.panels:
        if isinstance(model, IncomeModel):
            generator = model  # no latent type to draw; income remains observed
        else:
            generator = Model(model.data, 2, seed, model.mode, model.family_effect)
            generator.draws = rng.normal(size=(1, model.draws.shape[1]))
            generator.r = 1
        generated = np.zeros_like(p.counts)
        running, previous = np.zeros(model.j), None
        own, broad = np.zeros_like(p.stock), np.zeros_like(p.family_stock)
        controls = p.controls.copy()
        prev_total = 0
        for week in np.unique(p.weeks):
            rows = np.flatnonzero(p.weeks == week)
            gap = (
                0
                if previous is None
                else float((week - previous) / np.timedelta64(1, "D")) / 7
            )
            running *= model.data.delta**gap
            own[rows] = running
            for j in range(model.j):
                broad[rows, j] = running[
                    model.data.family_index == model.data.family_index[j]
                ].sum()
            for t in rows:
                if p.price_complete is not None and not p.price_complete[t]:
                    # There is no specified choice distribution without menu
                    # prices. Condition on this observed basket, preserving its
                    # exposure contribution, rather than simulate at placeholders.
                    generated[t] = p.counts[t]
                    continue
                for c, name in enumerate(model.data.control_names):
                    if name == "own_log_stock_x_log_previous_total":
                        controls[t, :, c] = np.log1p(running) * np.log1p(prev_total)
                    elif name == "own_log_stock_x_log_gap_weeks":
                        controls[t, :, c] = np.log1p(running) * np.log1p(gap)
                current = replace(
                    p,
                    counts=generated,
                    stock=own,
                    family_stock=broad,
                    controls=controls,
                )
                prob = np.exp(generator.log_probabilities(truth, current)[t, 0])
                generated[t] = rng.multinomial(int(p.counts[t].sum()), prob)
            prev_total = int(generated[rows].sum())
            running += generated[rows].sum(axis=0)
            previous = week
        panels.append(
            replace(
                p, counts=generated, stock=own, family_stock=broad, controls=controls
            )
        )
    return replace(model.data, panels=panels)


def self_test():
    """Focused invariants: calendar decay, stores, panel integration, recovery."""
    weeks = np.array(["2024-01-01", "2024-01-01", "2024-01-15"], dtype="datetime64[ns]")
    counts = np.array([[2, 0], [3, 1], [0, 2]])
    own, broad = stocks(weeks, counts, np.array([0, 1]), 0.5)
    np.testing.assert_allclose(own, [[0, 0], [0, 0], [1.25, 0.25]])
    np.testing.assert_allclose(own, broad)
    panel = Panel(
        "test",
        weeks,
        counts,
        np.array([[1, 2], [1, 2], [2, 1.0]]),
        np.ones_like(counts, dtype=bool),
        np.zeros((3, 2, 0)),
        own,
        broad,
        np.ones(3, dtype=bool),
    )
    data = Dataset([panel], ["PLAIN", "STRAWBERRY"], [], np.array([0, 1]), 0.5, 0)
    model = Model(data, 16)
    x = np.array([0.2, 0.6, 0.4, -0.5])
    logp = model.log_probabilities(x, panel)
    explicit = -logsumexp(
        [
            sum(counts[t, j] * logp[t, r, j] for t in range(3) for j in range(2))
            for r in range(model.r)
        ]
    ) + np.log(model.r)
    np.testing.assert_allclose(model.objective(x), explicit)
    zero = x.copy()
    zero[1] = 0
    fixed_ll = -(counts * model.log_probabilities(zero, panel)[:, 0]).sum()
    np.testing.assert_allclose(model.objective(zero), fixed_ll)
    unavailable = replace(
        panel,
        available=np.array([[True, False], [True, True], [True, True]]),
        counts=np.array([[2, 0], [3, 1], [0, 2]]),
    )
    assert np.isfinite(Model(replace(data, panels=[unavailable]), 16).objective(x))
    # Meaningful synthetic recovery uses many households, varying menus/prices,
    # multiple servings, calendar gaps, and recursively simulated exposure.
    rng = np.random.default_rng(83)
    panels = []
    for h in range(50):
        w = np.datetime64("2024-01-01", "ns") + np.cumsum(
            rng.integers(1, 3, 24)
        ) * np.timedelta64(7, "D")
        q = np.column_stack([np.full(24, 4), np.zeros(24, dtype=int)])
        s, f = stocks(w, q, np.array([0, 1]), 0.8)
        panels.append(
            Panel(
                str(h),
                w,
                q,
                rng.uniform(0.5, 3, (24, 2)),
                np.ones((24, 2), bool),
                np.zeros((24, 2, 0)),
                s,
                f,
                np.arange(24) >= 3,
            )
        )
    design = Dataset(panels, data.flavors, [], np.array([0, 1]), 0.8, 3)
    design_model = Model(design, 64)
    truth = np.array([0.5, 0.7, 0.6, -0.8])
    generated = simulate(design_model, truth, 92)
    for p in generated.panels:
        s, f = stocks(p.weeks, p.counts, generated.family_index, generated.delta)
        np.testing.assert_allclose(p.stock, s)
        np.testing.assert_allclose(p.family_stock, f)
    result = Model(generated, 64).fit(starts=2, maxiter=200)
    assert result.success, result.message
    assert np.max(np.abs(result.x - truth)) < 0.4, (result.x, truth)
    print(
        "Self-test passed: stocks, multi-store aggregation, panel likelihood, availability, recursive simulation, recovery."
    )
    print("Synthetic truth:", truth, "Recovered:", np.round(result.x, 4))
    # Observed income heterogeneity: check the exact log-odds identity and
    # analytical gradient, then recover BOTH preference and fatigue interactions
    # from newly simulated histories. No quadrature nodes enter this exercise.
    income_panels = [replace(p, income=float(h)) for p, h in
                     zip(design.panels, np.linspace(-1.5, 1.5, len(design.panels)))]
    income_data = replace(design, panels=income_panels)
    income_model = IncomeModel(income_data)
    income_truth = np.array([.5, .4, .6, .2, -.8])
    generated_income = simulate(income_model, income_truth, 96)
    fitted_income_model = IncomeModel(generated_income)
    p = generated_income.panels[0]
    logp = fitted_income_model.log_probabilities(income_truth, p)[:, 0]
    expected_odds = (income_truth[0]+income_truth[1]*p.income
                     +income_truth[-1]*(p.prices[:, 1]-p.prices[:, 0])
                     -(income_truth[2]+income_truth[3]*p.income)
                     *(np.log1p(p.stock[:, 1])-np.log1p(p.stock[:, 0])))
    np.testing.assert_allclose(logp[:, 1]-logp[:, 0], expected_odds)
    numeric = []
    for k in range(len(income_truth)):
        plus, minus = income_truth.copy(), income_truth.copy()
        plus[k] += 1e-5; minus[k] -= 1e-5
        numeric.append((fitted_income_model.objective(plus)-fitted_income_model.objective(minus))/2e-5)
    np.testing.assert_allclose(fitted_income_model.gradient(income_truth), numeric, atol=1e-5)
    assert fitted_income_model.rank_diagnostic()["full_column_rank"]
    income_result = fitted_income_model.fit(starts=2, maxiter=200)
    assert income_result.success, income_result.message
    assert np.max(np.abs(income_result.x-income_truth)) < .3, (income_result.x, income_truth)
    se, inference = fitted_income_model.clustered_inference(income_result.x)
    assert inference["standard_errors_available"] and np.isfinite(se).all()
    no_fatigue = fitted_income_model.fit(initial=income_result.x, starts=1, no_fatigue=True)
    np.testing.assert_array_equal(no_fatigue.x[2:4], [0, 0])
    print("Income checks passed: log odds, gradient, rank, clustered inference, nested restrictions, recursive recovery.")
    print("Income truth:", income_truth, "Recovered:", np.round(income_result.x, 4))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_DATA_PATH,
        help=f"Purchase CSV/Parquet (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--menu",
        help="Independently observed store-week product menu CSV/Parquet",
    )
    parser.add_argument("--output", help="Default: step4_income_output (income) or step4_output (random)")
    parser.add_argument("--heterogeneity", choices=["income", "random"], default="income",
                        help="Income interactions without draws (default), or legacy random coefficients")
    parser.add_argument("--income-column", default="household_income")
    parser.add_argument("--income-encoding", choices=["codes", "dollars"], default="codes",
                        help="Map Nielsen brackets to dollars, or use a supplied dollars column")
    parser.add_argument(
        "--flavor-column", default="flavor", help="Flavor category or individual-flavor column"
    )
    parser.add_argument(
        "--attribute-columns",
        nargs="*",
        default=[],
        help="Menu brand/style columns, e.g. brand_cd yogurt_style",
    )
    parser.add_argument("--prefiltered", action="store_true")
    parser.add_argument("--top-flavors", type=int, default=6)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.8,
        help="Weekly exposure retention; sensitivity parameter",
    )
    parser.add_argument("--timing-controls", action="store_true")
    parser.add_argument(
        "--gamma-mode", choices=["common", "flavor", "random"], default="common"
    )
    parser.add_argument("--family-effect", action="store_true")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument(
        "--profile",
        type=float,
        nargs="*",
        help="Fixed gamma values; refit every nuisance parameter",
    )
    parser.add_argument("--recovery-reps", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true",
                        help="Save sample/identification diagnostics without estimating")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not (0 < args.delta < 1):
        parser.error("Require 0 < delta < 1.")
    if (
        min(args.top_flavors, args.draws, args.starts, args.maxiter) < 1
        or args.top_flavors < 2
        or args.recovery_reps < 0
    ):
        parser.error("Invalid flavor/draw/start/iteration/recovery counts.")
    if args.profile is not None and (
        args.gamma_mode == "flavor" or any(abs(v) > 8 for v in args.profile)
    ):
        parser.error("Profile needs common/random gamma and values within [-8, 8].")
    if args.heterogeneity == "income" and args.gamma_mode != "common":
        parser.error("Income mode uses gamma + gamma_income; --gamma-mode flavor/random applies only to --heterogeneity random.")
    output = local_path(args.output or ("step4_income_output" if args.heterogeneity == "income" else "step4_output"))
    data, report = prepare(args)
    model = make_model(data, args)
    output.mkdir(parents=True, exist_ok=True)
    identification(data).to_csv(output / "identification.csv", index=False)
    if data.attrition is not None:
        data.attrition.to_csv(output / "household_attrition.csv", index=False)
    pd.DataFrame(report["source_sample_stages"]).to_csv(output / "source_sample_stages.csv", index=False)
    pd.DataFrame(report["choice_sample_stages"]).to_csv(output / "choice_sample_stages.csv", index=False)
    if isinstance(model, IncomeModel):
        report["income"] = data.income_metadata
        report["income_design"] = model.rank_diagnostic()
        data.income_table.to_csv(output / "household_income.csv", index=False)
        (output / "income_design_rank.json").write_text(json.dumps(report["income_design"], indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.audit_only:
        (output / "sample_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Sample audit saved inside {output}; no estimation performed.")
        return
    print(
        f"Fitting {len(model.names)} parameters " + ("with income interactions and no draws..." if isinstance(model, IncomeModel) else f"with {model.r} persistent draws..."),
        flush=True,
    )
    fit = model.fit(args.starts, maxiter=args.maxiter)
    estimates = pd.DataFrame(
        {
            "parameter": model.names,
            "estimate": fit.x,
            "lower_bound": [b[0] for b in model.bounds],
            "upper_bound": [b[1] for b in model.bounds],
        }
    )
    estimates["at_bound"] = np.isclose(
        estimates.estimate, estimates.lower_bound, atol=1e-5
    ) | np.isclose(estimates.estimate, estimates.upper_bound, atol=1e-5)
    if isinstance(model, IncomeModel):
        se, inference = model.clustered_inference(fit.x)
        if not fit.success:
            se[:] = np.nan
            inference["standard_errors_available"] = False
            inference["reason"] = "optimizer did not converge"
        estimates["se_household_cluster"] = se
        estimates["ci95_low"] = estimates.estimate - 1.96*se
        estimates["ci95_high"] = estimates.estimate + 1.96*se
        report["inference"] = inference
    estimates.to_csv(output / "estimates.csv", index=False)
    # Income mode uses exact curvature and clustered score variability. Legacy
    # random mode still omits Hessian-based standard errors; profiles/recovery
    # remain separate diagnostics, not substitutes for clustered inference.
    report.update(
        {
            "negative_log_likelihood": float(fit.fun),
            "converged": bool(fit.success),
            "optimizer_message": str(fit.message),
            "start_objectives": fit.start_objectives,
            "heterogeneity": args.heterogeneity,
            "gamma_mode": "income_interaction" if isinstance(model, IncomeModel) else args.gamma_mode,
            "family_effect": args.family_effect,
            "actual_draws": 0 if isinstance(model, IncomeModel) else model.r,
            "seed": args.seed,
            "standard_errors": "household-cluster sandwich; unavailable at bounds or with singular/ill-conditioned curvature" if isinstance(model, IncomeModel) else "not reported; inspect profile and recovery",
            "price_endogeneity": "no IV/control function; estimates conditional on supplied prices",
            "initial_conditions": "zero stock at first observed retained-flavor purchase; first week retained; no burn-in",
        }
    )
    # This nested comparison fixes all history coefficients, including optional
    # dose/gap interactions. Leaving those free would retain fatigue effects.
    naive = model.fit(args.starts, fit.x, maxiter=args.maxiter, no_fatigue=True)
    pd.DataFrame({"parameter": model.names, "estimate": naive.x}).to_csv(
        output / "no_fatigue_estimates.csv", index=False
    )
    report["no_fatigue_negative_log_likelihood"] = float(naive.fun)
    report["no_fatigue_converged"] = bool(naive.success)
    report["no_fatigue_beats_full_model"] = bool(naive.fun < fit.fun - 1e-5)
    report["arguments"] = vars(args)
    if args.profile:
        profile(model, fit, args.profile, args.starts, args.maxiter).to_csv(
            output / "gamma_profile.csv", index=False
        )
    recovery = []
    for repetition in range(args.recovery_reps):
        generated = simulate(model, fit.x, args.seed + 1000 + repetition)
        estimated = make_model(generated, args).fit(
            args.starts,
            initial=fit.x,
            maxiter=args.maxiter,
            seed=args.seed + repetition,
        )
        for name, truth, recovered in zip(model.names, fit.x, estimated.x):
            recovery.append(
                {
                    "replication": repetition,
                    "parameter": name,
                    "truth": truth,
                    "estimate": recovered,
                    "error": recovered - truth,
                    "converged": bool(estimated.success),
                }
            )
        print(
            f"Recovery replication {repetition + 1}/{args.recovery_reps}, converged={estimated.success}",
            flush=True,
        )
    if recovery:
        pd.DataFrame(recovery).to_csv(output / "recovery.csv", index=False)
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(estimates.to_string(index=False))
    print(f"Results saved inside {output}")


if __name__ == "__main__":
    main()
