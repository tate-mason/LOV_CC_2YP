#!/usr/bin/env python3
"""Split-history pricing for each focal flavor, separate from demand estimation.

User-supplied paths only. No estimation is performed. --self-test uses only
generated synthetic inputs. See README.md for definitions.
"""
from __future__ import annotations

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next((p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file()), None)
if _policy_root is not None:
    _policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_data_path
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from pprint import pformat
import tempfile
import numpy as np
import pandas as pd

# === Parameters: edit these values === #

# Set paths only on the server. None prevents an accidental empirical run.
household_input = None
fit_dir = None
output = Path("/scratch/dtm63837/Kilts_Panel/results") / "split_pricing"
focal_flavor = "all"                  # all, other, berry, or plain
discounts = np.linspace(0., 1., 101)  # Candidate focal-product coupons.
demand_delta = .92                    # Same default as theoretical supply.
batch_size = 24                       # Progress interval in the household loop.
type_treatment = "posterior-mean"     # Or posterior-integrated.
save_state_policies = False
run_self_tests = False

# === 1. Persistent switching, habit, and price types === #

FLAVORS = ["other", "berry", "plain"]

def expit(x):
    x = np.asarray(x)
    small = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+small), small/(1+small))

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x)
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)

def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def persistent_draws(number, seed):
    if number < 2 or number % 2:
        raise ValueError("Draw count must be even and >=2.")
    half = np.random.default_rng(seed).standard_normal((number//2, 3))
    return np.vstack([half, -half])

def posterior(frame, parameters, draws, control=True):
    """Household-integrated multinomial likelihood; types persist across weeks."""
    labels = np.sort(frame.household_code.unique())
    hh = pd.Categorical(frame.household_code, categories=labels).codes
    prices = frame[["price_"+f for f in FLAVORS]].to_numpy(float)
    available = np.isfinite(prices) & (prices > 0)
    counts = frame[FLAVORS+["outside"]].to_numpy(float)
    novelty = frame[["novelty_"+f for f in FLAVORS]].to_numpy(float)
    stock = frame[["stock_"+f for f in FLAVORS]].to_numpy(float)
    cf = frame[["cf_"+f for f in FLAVORS]].to_numpy(float) if control else np.zeros_like(prices)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or not np.isfinite(novelty).all() or not np.isfinite(stock).all() or not np.isfinite(cf).all()
            or ((counts[:, :3] > 0) & ~available).any()):
        raise ValueError("Invalid counts, history, controls, or purchased-alternative coverage.")
    beta = parameters["inside_constant"] + np.array([0., parameters["beta_berry"], parameters["beta_plain"]])
    alpha = np.exp(parameters["mu_log_minus_alpha"] + parameters["sd_log_minus_alpha"]*draws[:, 0])
    switching = parameters["mean_novelty"] + parameters["sd_novelty"]*draws[:, 1]
    habit = parameters["mean_own_stock"] + parameters["sd_own_stock"]*draws[:, 2]
    loglike = np.empty((len(labels), len(draws)))
    for d, (a, g, b) in enumerate(zip(alpha, switching, habit)):
        u = beta-a*np.nan_to_num(prices)+g*novelty+b*stock+parameters.get("price_control", 0.)*cf
        u = np.column_stack([np.where(available, u, -np.inf), np.zeros(len(frame))])
        logp = u-logsumexp(u, axis=1, keepdims=True)
        ll = (counts*np.where(counts > 0, logp, 0.)).sum(axis=1)
        loglike[:, d] = np.bincount(hh, weights=ll, minlength=len(labels))
    weights = np.exp(loglike-logsumexp(loglike, axis=1, keepdims=True))
    nll = -float((logsumexp(loglike, axis=1)-np.log(len(draws))).sum())
    means = weights @ switching
    # Exactly floor(N/4) per tail. Household ID breaks ties reproducibly.
    order = np.argsort(means, kind="stable")
    groups = np.full(len(labels), "middle", dtype=object)
    n = len(labels)//4
    if n < 1:
        raise ValueError("Need at least four households to form quartiles.")
    groups[order[:n]], groups[order[-n:]] = "low", "high"
    population_q = np.quantile(switching, [.25, .75])
    table = pd.DataFrame(dict(household_code=labels, group=groups, switching_mean=means,
                             switching_sd=np.sqrt(np.maximum(weights @ switching**2-means**2, 0)),
                             alpha_mean=weights @ alpha, posterior_ess=1/(weights**2).sum(axis=1),
                             probability_switching_below_draw_q25=weights @ (switching <= population_q[0]),
                             probability_switching_above_draw_q75=weights @ (switching >= population_q[1])))
    table["habit_mean"] = weights @ habit
    table["habit_sd"] = np.sqrt(np.maximum(weights @ habit**2 - table.habit_mean.to_numpy()**2, 0))
    return table, weights, alpha, np.column_stack([switching, habit]), beta, nll

# === 2. Baskets and the observed initial history === #

def compositions(capacity):
    """All multinomial baskets of three inside flavors and outside."""
    return np.array([(a,b,c,capacity-a-b-c) for a in range(capacity+1)
                     for b in range(capacity-a+1) for c in range(capacity-a-b+1)], dtype=int)

def forecast_context(frame, households):
    """First observed choice establishes h0; price the next two consecutive weeks.

    Carry the initial observation's menu, posted prices and controls forward.
    Type estimation still uses the complete observed sample.
    """
    first = frame.sort_values(["household_code", "week"]).groupby("household_code", sort=True).head(1)
    first = first.set_index("household_code").loc[households.household_code]
    counts = first[FLAVORS].to_numpy(float)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    # Saved stocks precede the observed choice; add its basket exactly once.
    stocks = first[["stock_"+f for f in FLAVORS]].to_numpy(float)+shares
    novelty = shares.sum(axis=1, keepdims=True)-shares
    prices = first[["price_"+f for f in FLAVORS]].to_numpy(float)
    available = np.isfinite(prices) & (prices > 0)
    eligible = available[:, 1]
    households["berry_eligible"] = eligible
    initial_weeks = first.week.to_numpy(dtype="datetime64[ns]")
    households["initial_observed_week"] = initial_weeks
    households["pricing_week1"] = initial_weeks+np.timedelta64(7, "D")
    households["pricing_week2"] = initial_weeks+np.timedelta64(14, "D")
    cf_cols = ["cf_"+f for f in FLAVORS]
    cf = first[cf_cols].to_numpy(float) if all(c in first for c in cf_cols) else np.zeros_like(prices)
    if not np.isfinite(stocks).all() or not np.isfinite(novelty).all():
        raise ValueError("Nonfinite forecast history.")
    return dict(stocks=stocks, novelty=novelty, prices=np.where(available, prices, 0.),
                available=available, cf=cf, eligible=eligible)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


# === 3. Backward induction, one household at a time === #

@server_only
def targeted_coupons(context, households, weights, alpha, switching, beta, discounts,
                     delta=1., capacity=7, control_coefficient=0.,
                     batch_size=24, type_treatment="posterior-mean", state_output=None, benchmark_output=None):
    """Choose household-specific p1(h0) and terminal p2(h1) by backward induction.

    Work through ONE household at a time. This makes type-specific pricing
    explicit and avoids the old extra household dimension in every array.
    For this household, arrays run over possible types, baskets, and coupons.
    batch_size only controls how frequently progress is reported.
    """
    baskets = compositions(capacity)
    number_states = len(baskets)
    number_coupons = len(discounts)
    state_numbers = np.arange(number_states)

    # A basket records [other, berry, plain, outside]. History uses inside shares.
    inside_counts = baskets[:, :3]
    inside_total = inside_counts.sum(axis=1, keepdims=True)
    basket_shares = np.divide(
        inside_counts, inside_total,
        out=np.zeros_like(inside_counts, dtype=float), where=inside_total > 0)
    basket_novelty = basket_shares.sum(axis=1, keepdims=True) - basket_shares
    log_multinomial_coefficient = (
        math.lgamma(capacity + 1)
        - np.vectorize(math.lgamma)(baskets + 1).sum(axis=1))

    eligible_households = np.flatnonzero(context["eligible"])
    targeted = households.group.isin(["low", "high"]).to_numpy()
    rows = []
    maximum_mass_error = 0.
    wrote_states = False
    common_values = np.zeros((number_coupons, number_coupons))

    for number_done, household in enumerate(eligible_households, start=1):
        prices = context["prices"][household]
        available = context["available"][household]
        stock = context["stocks"][household]
        novelty = context["novelty"][household]
        control = context["cf"][household]

        # Default: the firm knows estimated price, switching, and habit coefficients.
        # Sensitivity: retain the full probability distribution of possible types.
        if type_treatment == "posterior-mean":
            price_sensitivity = np.array([weights[household] @ alpha])
            history_coefficient = (weights[household] @ switching)[None, :]
            type_weights = np.ones(1)
        else:
            price_sensitivity = alpha
            history_coefficient = switching
            type_weights = weights[household]

        # Columns of history_coefficient are b_N (switching) and b_S (habit).
        b_n = history_coefficient[:, 0]
        b_s = history_coefficient[:, 1]
        history1 = b_n[:, None]*novelty + b_s[:, None]*stock
        history2 = (b_n[:, None, None]*basket_novelty
                    + b_s[:, None, None]*(stock + basket_shares))
        utility1 = (beta - price_sensitivity[:, None]*prices
                    + history1 + control_coefficient*control)
        utility2 = (beta - price_sensitivity[:, None, None]*prices
                    + history2 + control_coefficient*control)
        utility1 = np.where(available, utility1, -np.inf)
        utility2 = np.where(available, utility2, -np.inf)

        # STEP A: terminal outcomes for every candidate coupon and every h1.
        # Metrics are [revenue, berry units, coupon expenditure].
        terminal = []
        for discount in discounts:
            utility = utility2.copy()
            utility[:, :, 1] += price_sensitivity[:, None] * prices[1] * discount
            outside = np.zeros((*utility.shape[:-1], 1))
            all_utilities = np.concatenate([utility, outside], axis=-1)
            log_denominator = logsumexp(all_utilities, axis=-1)
            berry_quantity = capacity * np.exp(utility[:, :, 1] - log_denominator)
            offered_price = prices[1] * (1 - discount)
            terminal.append(np.stack([
                berry_quantity * offered_price,
                berry_quantity,
                berry_quantity * prices[1] * discount], axis=-1))
        terminal = np.asarray(terminal)

        first = np.zeros((number_coupons, 3))
        second = np.zeros((number_coupons, 3))
        paths = []

        # STEP B: try each p1(h0), calculate its state probabilities, and then
        # select the best p2 for each state. Optimize BEFORE averaging revenue.
        for action, discount in enumerate(discounts):
            utility = utility1.copy()
            utility[:, 1] += price_sensitivity * prices[1] * discount
            all_utilities = np.column_stack([utility, np.zeros(len(type_weights))])
            log_probability = all_utilities - logsumexp(all_utilities, axis=1, keepdims=True)
            terms = np.zeros((len(type_weights), number_states, 4))
            np.multiply(baskets[None, :, :], log_probability[:, None, :],
                        out=terms, where=baskets[None, :, :] > 0)
            basket_probability = np.exp(log_multinomial_coefficient + terms.sum(axis=-1))
            error = float(np.abs(basket_probability.sum(axis=1) - 1).max())
            maximum_mass_error = max(maximum_mass_error, error)

            # Joint(type,h1) = initial type weight x Pr(h1 | type,p1).
            joint = type_weights[:, None] * basket_probability
            state_probability = joint.sum(axis=0)
            berry_quantity = capacity * (type_weights * np.exp(log_probability[:, 1])).sum()
            first[action] = [berry_quantity * prices[1] * (1 - discount),
                             berry_quantity,
                             berry_quantity * prices[1] * discount]

            # In the uncertainty sensitivity, integrate possible types before
            # choosing a terminal coupon: the firm cannot observe the true draw.
            # Result dimensions: terminal coupon x basket h1 x metric.
            weighted_terminal = (terminal * joint[None, :, :, None]).sum(axis=1)
            if targeted[household]:
                common_values[action] += first[action, 0] + delta * weighted_terminal[:, :, 0].sum(axis=1)
            elif action == 0:
                common_values += first[0, 0] + delta * weighted_terminal[0, :, 0].sum()
            best_terminal_action = weighted_terminal[:, :, 0].argmax(axis=0)
            selected = weighted_terminal[best_terminal_action, state_numbers]
            second[action] = selected.sum(axis=0)
            if action == 0:
                no_coupon_second = weighted_terminal[0].sum(axis=0)
            paths.append((best_terminal_action, state_probability))

        # STEP C: choose the first coupon by total expected discounted revenue.
        dynamic_action = int((first[:, 0] + delta * second[:, 0]).argmax())
        if not targeted[household]:
            dynamic_action = 0

        # Keep posted prices as a benchmark. Only dynamic pricing is optimized.
        for policy, action in [("no_coupon", 0), ("personalized_dynamic", dynamic_action)]:
            use_coupons = targeted[household] and policy == "personalized_dynamic"
            initial = first[action]
            best_terminal_action, state_probability = paths[action]
            if use_coupons:
                terminal_metrics = second[action]
                terminal_coupon = discounts[best_terminal_action]
            else:
                terminal_metrics = no_coupon_second
                terminal_coupon = np.zeros(number_states)

            p1 = prices[1] * (1 - discounts[action])
            p2 = prices[1] * (1 - terminal_coupon)
            mean_p2 = float(state_probability @ p2)
            ratio = np.divide(p1, p2, out=np.full(number_states, np.nan), where=p2 > 0)
            positive_mass = state_probability > 1e-14
            expected_ratio = np.nan
            if np.isfinite(ratio[positive_mass]).all():
                expected_ratio = float(state_probability[positive_mass] @ ratio[positive_mass])

            row = dict(
                household_code=households.iloc[household].household_code,
                group=households.iloc[household].group, policy=policy,
                discount1=discounts[action],
                expected_discount2=float(state_probability @ terminal_coupon),
                p1=p1, expected_p2=mean_p2,
                ratio_of_expected_prices=p1 / mean_p2 if mean_p2 > 0 else np.nan,
                expected_price_ratio=expected_ratio,
                revenue1=initial[0], revenue2=terminal_metrics[0],
                discounted_revenue=initial[0] + delta * terminal_metrics[0],
                berry_units1=initial[1], berry_units2=terminal_metrics[1],
                coupon_expenditure=initial[2] + delta * terminal_metrics[2])
            for branch, label in [(baskets[:, 1] == 0, "no_berry"), (baskets[:, 1] > 0, "berry")]:
                branch_mass = state_probability[branch].sum()
                row["probability_" + label] = branch_mass
                row["mean_p2_given_" + label] = np.nan
                if branch_mass > 0:
                    row["mean_p2_given_" + label] = float(
                        state_probability[branch] @ p2[branch] / branch_mass)
            rows.append(row)

            if state_output is not None and use_coupons:
                detail = pd.DataFrame(dict(
                    household_code=row["household_code"], group=row["group"], policy=policy,
                    state=state_numbers, rival1_units=baskets[:, 0], focal_units=baskets[:, 1],
                    rival2_units=baskets[:, 2], outside=baskets[:, 3], probability=state_probability,
                    discount1=discounts[action], discount2=terminal_coupon, p1=p1, p2=p2,
                    price_ratio=ratio))
                detail.drop(columns=["household_code"]).to_csv(state_output, mode="a" if wrote_states else "w",
                              header=not wrote_states, index=False, compression="gzip")
                wrote_states = True

        if number_done % (batch_size * 40) == 0 or number_done == len(eligible_households):
            print(f"Targeted pricing: integrated {number_done}/{len(eligible_households)} eligible households", flush=True)

    if maximum_mass_error > 1e-9:
        raise AssertionError(f"Basket probability mass error: {maximum_mass_error}")
    table = pd.DataFrame(rows)
    baseline = table.loc[table.policy == "no_coupon"].set_index("household_code").discounted_revenue
    table["baseline_revenue"] = table.household_code.map(baseline)
    table["revenue_change"] = table.discounted_revenue - table.baseline_revenue
    table["revenue_change_pct"] = 100 * table.revenue_change / table.baseline_revenue
    comparison = table.pivot(index="household_code", columns="policy", values="discounted_revenue")
    if (comparison.personalized_dynamic + 1e-8 < comparison.no_coupon).any():
        raise AssertionError("Dynamic pricing does not dominate the feasible no-coupon benchmark.")
    if benchmark_output is not None:
        a1, a2 = np.unravel_index(common_values.argmax(), common_values.shape)
        totals = table.groupby("policy").discounted_revenue.sum()
        common = float(common_values[a1, a2])
        if common + 1e-8 < totals["no_coupon"] or common > totals["personalized_dynamic"] + 1e-8:
            raise AssertionError("Expected no coupon <= common coupons <= personalized dynamic.")
        for policy, value, d1, d2 in [
                ("no_coupon", totals["no_coupon"], 0., 0.),
                ("optimized_common_coupons", common, discounts[a1], discounts[a2]),
                ("personalized_dynamic", totals["personalized_dynamic"], np.nan, np.nan)]:
            benchmark_output.append(dict(policy=policy, discounted_revenue=value,
                discount1=d1, discount2=d2,
                revenue_change_pct=100*(value/totals["no_coupon"]-1)))
    return table, maximum_mass_error




# === 4. Validate a saved split fit and solve one focal case === #

# Same fixed-belief comparison as the theory, using fitted low/medium/high types.
TYPE_GROUPS = ["low", "middle", "high"]


def gamma_type_values(households):
    """Estimated novelty means for the existing low/medium/high prediction groups."""
    values = households.groupby("group").switching_mean.mean().reindex(TYPE_GROUPS).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Need fitted low, medium and high types to price type uncertainty")
    return values

GAMMA_PRIORS = [
    ("Equal types", [1/3, 1/3, 1/3]),
    ("Low type more likely", [.5, .25, .25]),
    ("Medium type more likely", [.25, .5, .25]),
    ("High type more likely", [.25, .25, .5]),
]


def fixed_gamma_policy(mass, first, terminal, discounts, prior, delta):
    """Finite-grid counterpart of theory.evaluate: fixed prior in EVERY h1.

    mass: first action,type,basket; first: first action,type;
    terminal: type,basket,terminal action. No likelihood enters terminal weights.
    """
    prior = np.asarray(prior, float)
    if not np.isclose(prior.sum(), 1.) or (prior < 0).any():
        raise ValueError("Invalid gamma prior")
    if not np.allclose(mass.sum(axis=-1), 1., atol=1e-9):
        raise AssertionError("Basket mass failed")
    # Unknown type: terminal policy depends on h1 but keeps the original prior.
    mixture_terminal = np.einsum("t,tsa->sa", prior, terminal)
    second_action = mixture_terminal.argmax(axis=-1)
    terminal_value = mixture_terminal.max(axis=-1)
    mixture_mass = np.einsum("t,ats->as", prior, mass)
    first_value = first @ prior
    objectives = first_value + delta * (mixture_mass @ terminal_value)
    action = int(objectives.argmax())
    unknown_value = float(objectives[action])
    # Known type: optimize both offers separately for each persistent gamma.
    known_second = terminal.argmax(axis=-1)
    known_terminal = terminal.max(axis=-1)
    known_values = first + delta*np.einsum("ats,ts->at", mass, known_terminal)
    known_first = known_values.argmax(axis=0)
    types = np.arange(len(prior))
    known_by_type = known_values[known_first, types]
    known_value = float(prior @ known_by_type)
    known_mass = mass[known_first, types]
    known_d1 = float(prior @ discounts[known_first])
    known_d2_by_type = (known_mass * discounts[known_second]).sum(axis=-1)
    unknown_d2 = float(mixture_mass[action] @ discounts[second_action])
    # Audit economic performance separately under a truly persistent gamma.
    # Fixed-belief objectives mix types independently at each state; they need
    # not equal realized ex ante revenue under persistent heterogeneity.
    selected_true = np.take_along_axis(terminal, second_action[None, :, None], axis=-1)[..., 0]
    persistent_value = float(first_value[action] + delta * np.sum(
        prior[:, None]*mass[action]*selected_true))
    if known_value + 1e-8 < persistent_value:
        raise AssertionError("Full-information bound on persistent-type revenue failed")
    return dict(known_value=known_value, unknown_value=unknown_value,
                persistent_value=persistent_value, known_by_type=known_by_type,
                known_first=known_first, known_d2_by_type=known_d2_by_type,
                known_d1=known_d1, known_d2=float(prior @ known_d2_by_type),
                unknown_d1=float(discounts[action]), unknown_d2=unknown_d2,
                second_action=second_action)


@server_only
def information_frictions(context, households, weights, alpha, switching, beta,
                          discounts, delta, capacity, control_coefficient, output):
    """Known versus fixed-belief gamma, with observed empirical initial history."""
    gamma_values = gamma_type_values(households)
    baskets = compositions(capacity)
    inside = baskets[:, :3]
    totals = inside.sum(axis=1, keepdims=True)
    shares = np.divide(inside, totals, out=np.zeros_like(inside, dtype=float), where=totals > 0)
    novelty2 = shares.sum(axis=1, keepdims=True)-shares
    coeff = math.lgamma(capacity+1)-np.vectorize(math.lgamma)(baskets+1).sum(axis=1)
    rows, state_rows = [], []
    eligible = np.flatnonzero(context["eligible"])
    for number, hh in enumerate(eligible, 1):
        a = float(weights[hh] @ alpha)
        _, habit = weights[hh] @ switching
        prices, available = context["prices"][hh], context["available"][hh]
        stock, novelty = context["stocks"][hh], context["novelty"][hh]
        initial_share = novelty.sum()/2-novelty[1]
        h0 = "j" if initial_share > 1e-12 else "0"
        fixed = beta-a*prices+control_coefficient*context["cf"][hh]
        u1 = fixed+gamma_values[:, None]*novelty+habit*stock
        u2 = (fixed[None, None, :]+gamma_values[:, None, None]*novelty2[None, :, :]
              +habit*(stock[None, None, :]+shares[None, :, :]))
        u1, u2 = np.where(available, u1, -np.inf), np.where(available, u2, -np.inf)
        def probabilities(u, discount):
            u = u.copy()
            u[..., 1] += a*prices[1]*discount
            full = np.concatenate([u, np.zeros((*u.shape[:-1], 1))], axis=-1)
            return full-logsumexp(full, axis=-1, keepdims=True)
        terminal = np.stack([capacity*np.exp(probabilities(u2, d)[..., 1])*prices[1]*(1-d)
                             for d in discounts], axis=-1)
        masses, first = [], []
        for discount in discounts:
            logp = probabilities(u1, discount)
            terms = np.zeros((3, len(baskets), 4))
            np.multiply(baskets[None, :, :], logp[:, None, :], out=terms,
                        where=baskets[None, :, :] > 0)
            masses.append(np.exp(coeff[None, :]+terms.sum(axis=-1)))
            first.append(capacity*np.exp(logp[:, 1])*prices[1]*(1-discount))
        mass, first = np.asarray(masses), np.asarray(first)
        for scenario, prior in GAMMA_PRIORS:
            result = fixed_gamma_policy(mass, first, terminal, discounts, prior, delta)
            for case, value, d1, d2, realized in [
                ("known", result["known_value"], result["known_d1"], result["known_d2"], result["known_value"]),
                ("unknown", result["unknown_value"], result["unknown_d1"], result["unknown_d2"], result["persistent_value"])]:
                rows.append(dict(scenario=scenario, h0=h0, case=case, group=households.iloc[hh].group,
                    profit=value, persistent_type_revenue=realized,
                    expected_discount1=d1, expected_discount2=d2,
                    expected_p1=prices[1]*(1-d1), expected_p2=prices[1]*(1-d2),
                    value_of_information=result["known_value"]-result["unknown_value"]))
            for k, gamma in enumerate(gamma_values):
                state_rows.append(dict(scenario=scenario, h0=h0, type=["low", "medium", "high"][k], gamma=gamma, probability=prior[k],
                    belief_after_focal=prior[k], belief_after_no_focal=prior[k],
                    known_profit=result["known_by_type"][k],
                    known_p1=prices[1]*(1-discounts[result["known_first"][k]]),
                    known_expected_p2=prices[1]*(1-result["known_d2_by_type"][k])))
        if number % 100 == 0 or number == len(eligible):
            print(f"Fixed-belief gamma pricing: {number}/{len(eligible)} eligible households", flush=True)
    table = pd.DataFrame(rows)
    def aggregate(keys):
        result = table.groupby(keys)[["profit", "persistent_type_revenue", "value_of_information"]].sum()
        result = result.join(table.groupby(keys)[["expected_p1", "expected_p2", "expected_discount1", "expected_discount2"]].mean())
        result["households"] = table.groupby(keys).size()
        return result
    summary = aggregate(["scenario", "h0", "case"])
    summary.to_csv(output / "information_friction_aggregate.csv")
    lines = [r"\begin{tabular}{llrrr}\hline",
             r"Initial state & Prior & Known profit & Fixed-belief profit & Difference\\\hline"]
    for scenario, _ in GAMMA_PRIORS:
        for h0 in ["j", "0"]:
            if (scenario, h0, "known") not in summary.index:
                continue
            known_row = summary.loc[(scenario, h0, "known")]
            unknown_row = summary.loc[(scenario, h0, "unknown")]
            lines.append(f"{h0} & {scenario} & {known_row.profit:.4f} & "
                         f"{unknown_row.profit:.4f} & {known_row.value_of_information:.4f} " + r"\\")
    lines.append(r"\hline\end{tabular}")
    (output / "information_friction_results.tex").write_text("\n".join(lines)+"\n")
    aggregate(["scenario", "h0", "case", "group"]).to_csv(output / "information_friction_by_group.csv")
    pd.DataFrame(state_rows).groupby(["scenario", "h0", "type", "gamma"]).agg(
        probability=("probability", "first"), belief_after_focal=("belief_after_focal", "first"),
        belief_after_no_focal=("belief_after_no_focal", "first"),
        mean_known_profit=("known_profit", "mean"), mean_known_p1=("known_p1", "mean"),
        mean_known_expected_p2=("known_expected_p2", "mean")
    ).to_csv(output / "information_friction_states.csv")
    settings = dict(types=["low", "medium", "high"], gamma=gamma_values.tolist(), priors=GAMMA_PRIORS, delta=delta, cost=0.,
        beliefs="fixed in every realized basket; no Bayesian updating", initial_history="observed",
        theory_reference="final_draft/writeup/1108_writeup/theory/gamma_uncertainty.py",
        gamma_interpretation="mean fitted novelty coefficient within each predicted type group",
        alpha_habit="household posterior means", profit="fixed-belief objective for unknown case",
        persistent_type_revenue="evaluated under the same persistent type across both periods")
    (output / "information_friction_settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    plt = plotting()
    scenarios = [name for name, _ in GAMMA_PRIORS]
    labels = ["Equal", "Low-heavy", "Medium-heavy", "High-heavy"]
    for metric, stem, ylabel in [("profit", "profits", "Expected discounted profit"),
                                 ("expected_p1", "first_prices", "Mean first-period offered price"),
                                 ("expected_p2", "terminal_prices", "Mean terminal offered price"),
                                 ("value_of_information", "information_value", "Known minus fixed-belief profit")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        x = np.arange(4)
        for ax, h0 in zip(axes, ["j", "0"]):
            for case, offset, label in [("known", -.18, "Known type"), ("unknown", .18, "Uncertain type; fixed beliefs")]:
                if metric == "value_of_information" and case == "unknown":
                    continue
                try:
                    values = summary.xs((h0, case), level=("h0", "case")).reindex(scenarios)[metric]
                except KeyError:
                    values = np.full(4, np.nan)
                ax.bar(x+(0 if metric == "value_of_information" else offset), values,
                       width=.6 if metric == "value_of_information" else .36, label=label)
            ax.set_xticks(x, labels, rotation=20)
            ax.set_title("Initially "+("inside" if h0 == "j" else "outside"))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(ylabel)
        if metric != "value_of_information":
            axes[0].legend(fontsize=8)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"information_friction_{stem}.{extension}", dpi=180)
        plt.close(fig)


@server_only
def run_households(args):
    require_data_path(args.fit_dir)
    # This branch is entered only when the person running the script supplies
    # BOTH paths. Validation supplies generated temporary fixtures only.
    fit = json.loads((args.fit_dir / "fit.json").read_text())
    if fit["model"] != "heterogeneous_history" or not fit.get("converged", False):
        raise ValueError("This exercise requires a converged split heterogeneous-history fit.")
    estimates = pd.read_csv(args.fit_dir / "point_estimates.csv")
    if "at_bound" in estimates and estimates.at_bound.astype(str).str.lower().eq("true").any():
        raise ValueError("Fit has boundary parameters; review before running counterfactuals.")
    parameters = dict(zip(estimates.parameter, estimates.estimate))
    settings = fit["settings"]
    capacity = settings["capacity"]
    if not isinstance(capacity, int) or not 1 <= capacity <= 12:
        raise ValueError("Exact basket enumeration supports integer capacity 1..12.")
    frame = pd.read_csv(args.household_input, dtype={"household_code": str}, parse_dates=["week"])
    required = ["household_code", "week", "outside", *FLAVORS]
    required += [prefix+f for prefix in ["price_", "stock_", "novelty_"] for f in FLAVORS]
    control = not settings.get("no_price_control", False)
    if control:
        required += ["cf_"+f for f in FLAVORS]
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f"Household input missing columns: {sorted(missing)}")
    if frame.household_code.isna().any() or frame.week.isna().any() or frame.duplicated(["household_code", "week"]).any():
        raise ValueError("Need unique household/week rows with nonmissing IDs/dates.")
    if len(frame) != fit["weeks"] or frame.household_code.nunique() != fit["households"]:
        raise ValueError("Input household/week counts do not match fit metadata.")
    draws = persistent_draws(settings["draws"], settings["seed"])
    households, weights, alpha, switching, beta, nll = posterior(frame, parameters, draws, control)
    # Exact likelihood reproduction guards against silently pairing files from different fits.
    if not np.isclose(nll, fit["nll"], rtol=1e-9, atol=1e-4):
        raise ValueError(f"Likelihood mismatch: reconstructed {nll}, fit {fit['nll']}.")
    focal = args.focal_flavor
    if focal != "berry":
        mapping = {"berry": focal, focal: "berry"}
        original = frame.copy()
        original_beta = beta.copy()
        for destination, source in mapping.items():
            for prefix in ["", "price_", "stock_", "novelty_", "cf_"]:
                if prefix+source in original:
                    frame[prefix+destination] = original[prefix+source]
            beta[FLAVORS.index(destination)] = original_beta[FLAVORS.index(source)]
    context = forecast_context(frame, households)
    counts = [int(((households.group == g) & households.berry_eligible).sum()) for g in ["low", "middle", "high"]]
    if any(n == 0 for n in counts):
        raise ValueError("Each switching segment must have focal-eligible forecast households.")
    benchmarks = []
    table, mass_error = targeted_coupons(
        context, households, weights, alpha, switching, beta, args.discounts,
        args.demand_delta, capacity,
        parameters.get("price_control", 0.) if control else 0., args.batch_size,
        args.type_treatment,
        args.output / "targeted_state_policies.csv.gz" if args.save_state_policies else None,
        benchmark_output=benchmarks)
    benchmark_table = pd.DataFrame(benchmarks)
    benchmark_table.to_csv(args.output / "pricing_policy_benchmarks.csv", index=False)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["No coupons", "Optimized common\ncoupons", "Personalized\ndynamic coupons"],
           benchmark_table.revenue_change_pct, color=["#999999", "#3575AD", "#D17925"])
    ax.set(ylabel="Revenue gain relative to no coupons (%)", title=f"{focal.title()}: pricing policy benchmarks")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Same eligible households; middle group receives no coupons. Common offers vary by period, not household or history.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ["pdf", "png"]:
        fig.savefig(args.output / f"pricing_policy_benchmarks.{extension}", dpi=180)
    plt.close(fig)
    information_frictions(context, households, weights, alpha, switching, beta,
                          args.discounts, args.demand_delta, capacity,
                          parameters.get("price_control", 0.) if control else 0., args.output)
    households.rename(columns={"berry_eligible": "focal_eligible"}, inplace=True)
    table.rename(columns={c: c.replace("berry", "focal") for c in table if "berry" in c}, inplace=True)
    table["focal_flavor"] = focal
    households.drop(columns=["household_code"]).to_csv(args.output / "household_posterior_summary.csv", index=False)
    table.drop(columns=["household_code"]).to_csv(args.output / "targeted_household_results.csv", index=False)
    totals = ["revenue1", "revenue2", "discounted_revenue", "baseline_revenue", "revenue_change",
              "focal_units1", "focal_units2", "coupon_expenditure"]
    aggregate = table.groupby("policy", sort=False)[totals].sum()
    aggregate["revenue_change_pct"] = 100*aggregate.revenue_change/aggregate.baseline_revenue
    aggregate.to_csv(args.output / "targeted_revenue_aggregate.csv")
    group = table.groupby(["policy", "group"], sort=False)[totals].sum()
    group["revenue_change_pct"] = 100*group.revenue_change/group.baseline_revenue
    means = table.groupby(["policy", "group"])[["p1", "expected_p2", "discount1", "expected_discount2",
                                                   "ratio_of_expected_prices", "expected_price_ratio"]].mean()
    group = group.join(means)
    group["eligible_households"] = table.groupby(["policy", "group"]).size()
    group.to_csv(args.output / "targeted_revenue_by_switching_group.csv")
    # Aggregate terminal offers conditional on first-period focal purchase.
    # Weight by branch probability, rather than averaging household conditional means.
    branch_rows = []
    dynamic = table.loc[table.policy == "personalized_dynamic"]
    for group_name, segment in dynamic.groupby("group"):
        for branch in ["focal", "no_focal"]:
            mass = segment["probability_" + branch]
            branch_price = segment["mean_p2_given_" + branch]
            probability_sum = mass.sum()
            branch_rows.append(dict(group=group_name, branch=branch,
                probability=probability_sum/len(segment),
                mean_terminal_price=(mass*branch_price).sum()/probability_sum if probability_sum else np.nan))
    branches = pd.DataFrame(branch_rows)
    paired = dynamic.assign(terminal_price_gap=(dynamic.mean_p2_given_focal - dynamic.mean_p2_given_no_focal))
    paired.groupby("group").agg(
        mean_within_household_price_gap=("terminal_price_gap", "mean"),
        households_with_both_branches=("terminal_price_gap", "count")
    ).to_csv(args.output / "terminal_offer_within_household_gap.csv")
    branches.to_csv(args.output / "terminal_offers_by_purchase.csv", index=False)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(3)
    for branch, offset, label in [("no_focal", -.18, "After no focal purchase"),
                                  ("focal", .18, "After focal purchase")]:
        values = branches.loc[branches.branch == branch].set_index("group").reindex(["low", "middle", "high"])
        ax.bar(x+offset, values.mean_terminal_price, width=.36, label=label)
    ax.set(xticks=x, xticklabels=["Low switching", "Middle switching", "High switching"],
           ylabel="Mean terminal price ($)", title=f"{focal.title()}: terminal offers conditional on purchase")
    ax.legend()
    fig.text(.5, .015, "Branch-probability-weighted means under personalized dynamic pricing; middle receives no coupons.\nConditional comparisons can reflect household composition as well as history effects.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for extension in ["pdf", "png"]:
        fig.savefig(args.output / f"terminal_offers_by_purchase.{extension}", dpi=180)
    plt.close(fig)

    households.groupby("group").agg(households=("household_code", "size"),
                                    focal_eligible=("focal_eligible", "sum"),
                                    switching_mean=("switching_mean", "mean"),
                                    switching_mean_min=("switching_mean", "min"),
                                    switching_mean_max=("switching_mean", "max"),
                                    mean_posterior_sd=("switching_sd", "mean"),
                                    habit_mean=("habit_mean", "mean"),
                                    mean_habit_posterior_sd=("habit_sd", "mean"),
                                    mean_alpha=("alpha_mean", "mean"),
                                    median_posterior_ess=("posterior_ess", "median")).to_csv(args.output / "household_group_summary.csv")
    make_targeting_figures(households, table, aggregate, args.output, delta=args.demand_delta,
                           focal_flavor=focal)
    return dict(focal_flavor=focal, state_coordinates={"focal_units": focal,
                "rival1_units": "berry" if focal == "other" else "other",
                "rival2_units": "berry" if focal == "plain" else "plain"}, households=len(households), eligible_by_group=dict(zip(["low", "middle", "high"], counts)),
                reconstructed_nll=nll, basket_mass_error=mass_error,
                input_sha256=digest(args.household_input), fit_sha256=digest(args.fit_dir / "fit.json"),
                parameter_sha256=digest(args.fit_dir / "point_estimates.csv"),
                posterior_draws=len(draws), posterior_seed=settings["seed"],
                capacity=capacity, history_model="b_N*N + b_S*S", price_control=control,
                initial_history="first_observed_basket", pricing_periods=2,
                baskets=len(compositions(capacity)),
                discount_grid=args.discounts.tolist(), demand_delta=args.demand_delta,
                type_treatment=args.type_treatment,
                gamma_information=dict(gamma_values=gamma_type_values(households).tolist(), priors=GAMMA_PRIORS,
                    beliefs="fixed; no Bayesian updating", alpha_habit="known posterior means",
                    initial_history="observed"),
                aggregate=aggregate.reset_index().to_dict("records"))


# === 5. Figures === #

def make_targeting_figures(households, table, aggregate, output, delta=.95, focal_flavor="berry"):
    """Plot household types and individually optimized berry discounts."""
    plt = plotting()
    groups = ["low", "middle", "high"]
    labels = ["Low switching\n(bottom 25%)", "Middle switching\n(middle 50%)", "High switching\n(top 25%)"]
    colors = ["#3575AD", "#999999", "#D17925"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.histogram_bin_edges(households.switching_mean, bins=30)
    for g, label, color in zip(groups, labels, colors):
        axes[0].hist(households.loc[households.group == g, "switching_mean"],
                     bins=bins, alpha=.65, label=label.replace("\n", " "), color=color)
    axes[0].set(title="Estimated consumer types", xlabel=r"Estimated household type (posterior mean $b_N$)",
                ylabel="Number of households")
    axes[0].legend(fontsize=9)
    revenue = table.loc[table.policy == "personalized_dynamic"].groupby("group")[["revenue1", "revenue2"]].sum().reindex(groups)
    x = np.arange(len(groups))
    # Average over state-contingent terminal prices first, discount, then total.
    total_expected_revenue = revenue.revenue1 + delta * revenue.revenue2
    axes[1].bar(x, total_expected_revenue, width=.5, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set(title="Total Expected Revenue",
                xlabel="Consumer type group", ylabel="Total Expected Revenue")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    axes[1].axhline(0, color="gray", linewidth=.6)
    fig.text(.5, .015,
             f"First-period revenue + {delta:g} × expected terminal revenue. "
             f"Low/high types receive personalized {focal_flavor} offers; middle types receive no coupons.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"household_coupon_results.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    means = table.loc[table.policy == "personalized_dynamic"].groupby("group")[["discount1", "expected_discount2"]].mean().reindex(groups)
    ax.bar(x-.18, 100*means.discount1, width=.36, label=r"$p_1(h_0)$ offer", color=colors[0])
    ax.bar(x+.18, 100*means.expected_discount2, width=.36,
           label=r"Terminal $p_2(h_1)$ offer (expected over histories)", color=colors[2])
    ax.set_xticks(x, labels)
    ax.set(xlabel="Consumer type group", ylim=(0, 105),
           ylabel=f"Mean optimal {focal_flavor} discount (% of posted price)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Dynamic pricing: optimal discounts by consumer type")
    fig.text(.5, .015, f"Household-weighted means among {focal_flavor}-eligible households. Middle group receives no coupons by policy design.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"discounts_by_consumer_type.{extension}", dpi=180)
    plt.close(fig)

    dynamic = table.loc[table.policy == "personalized_dynamic"].merge(
        households[["household_code", "switching_mean", "alpha_mean"]], on="household_code", validate="one_to_one")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for column, (variable, xlabel) in enumerate([
            ("switching_mean", r"Estimated household type (posterior mean $b_N$)"),
            ("alpha_mean", r"Estimated price sensitivity (posterior mean $a$)")]):
        for row, (discount, title) in enumerate([
                ("discount1", r"Optimal discount at $p_1(h_0)$"),
                ("expected_discount2", r"Expected terminal discount at $p_2(h_1)$")]):
            ax = axes[row, column]
            for g, label, color in zip(groups, labels, colors):
                subset = dynamic.loc[dynamic.group == g]
                ax.scatter(subset[variable], 100*subset[discount], s=12, alpha=.5,
                           color=color, label=label.replace("\n", " "))
            ax.set(xlabel=xlabel, ylabel=f"{focal_flavor.title()} discount (% of posted price)", title=title, ylim=(-2, 102))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Dynamic pricing: household discounts and estimated consumer characteristics")
    fig.text(.5, .015, f"Each point is one {focal_flavor}-eligible household. Terminal offers average over the household's realized h1 histories.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"discounts_by_household_type.{extension}", dpi=180)
    plt.close(fig)


@server_only
def self_test():
    """Server-only independent scalar enumeration and limiting-case checks."""
    rng = np.random.default_rng(42)
    discounts = np.array([0., .25, .5])
    mass = rng.dirichlet(np.ones(4), size=(3, 3))
    first = rng.uniform(.1, 2., size=(3, 3))
    terminal = rng.uniform(.1, 2., size=(3, 4, 3))
    priors = [prior for _, prior in GAMMA_PRIORS]+list(np.eye(3))
    for prior in priors:
        prior = np.asarray(prior)
        r = fixed_gamma_policy(mass, first, terminal, discounts, prior, .92)
        # Independent scalar loops: each terminal state retains the same prior.
        v = [max(sum(prior[t]*terminal[t, s, a] for t in range(3))
                 for a in range(3)) for s in range(4)]
        unknown = max(sum(prior[t]*first[a, t] for t in range(3))+.92*sum(
            sum(prior[t]*mass[a, t, s] for t in range(3))*v[s] for s in range(4))
                      for a in range(3))
        known = sum(prior[t]*max(first[a, t]+.92*sum(
            mass[a, t, s]*max(terminal[t, s, b] for b in range(3)) for s in range(4))
                                for a in range(3)) for t in range(3))
        assert np.isclose(r["unknown_value"], unknown)
        assert np.isclose(r["known_value"], known)
        if np.count_nonzero(prior) == 1:
            assert np.isclose(known, unknown)
            assert np.isclose(unknown, r["persistent_value"])
        changed = fixed_gamma_policy(mass[::-1], first[::-1], terminal, discounts, prior, .92)
        assert np.array_equal(r["second_action"], changed["second_action"])
    equal_mass = np.repeat(mass[:, :1], 3, axis=1)
    equal_first = np.repeat(first[:, :1], 3, axis=1)
    equal_terminal = np.repeat(terminal[:1], 3, axis=0)
    r = fixed_gamma_policy(equal_mass, equal_first, equal_terminal, discounts, [.5, .25, .25], .92)
    assert np.isclose(r["known_value"], r["unknown_value"])
    # End-to-end small synthetic export, including both observed h0 categories.
    households = pd.DataFrame(dict(group=["low", "middle", "high"], switching_mean=[-.7, .2, 1.1]))
    context = dict(eligible=np.ones(3, bool), prices=np.tile([2., 2.5, 3.], (3, 1)),
                   available=np.ones((3, 3), bool), stocks=np.array([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]]),
                   novelty=np.array([[1., 0., 1.], [0., 1., 1.], [1., 0., 1.]]), cf=np.zeros((3, 3)))
    with tempfile.TemporaryDirectory(dir="/scratch/dtm63837/Kilts_Panel") as directory:
        output = Path(directory)
        information_frictions(context, households, np.ones((3, 1)), np.array([1.]),
                              np.array([[0., .4]]), np.array([1., 1.2, .8]),
                              discounts, .92, 1, 0., output)
        summary = pd.read_csv(output / "information_friction_aggregate.csv")
        assert len(summary) == 16
        assert set(summary.case) == {"known", "unknown"}
        states = pd.read_csv(output / "information_friction_states.csv")
        assert len(states) == 24
        assert np.allclose(states.probability, states.belief_after_focal)
        assert np.allclose(states.probability, states.belief_after_no_focal)
        for stem in ["profits", "first_prices", "terminal_prices", "information_value"]:
            for extension in ["pdf", "png"]:
                assert (output / f"information_friction_{stem}.{extension}").is_file()
    print("Server checks passed: scalar enumeration, all four priors, degenerate priors, identical types, fixed terminal beliefs, persistent-type revenue bound, both initial histories and aggregate exports.")


# === 6. Optional command-line overrides === #

def read_command_line(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-input", type=Path, help="Explicit household input selected by you; no default.")
    parser.add_argument("--fit-dir", type=Path, help="Explicit existing fitted-results directory; no default.")
    parser.add_argument("--output", type=Path, default=Path("/scratch/dtm63837/Kilts_Panel/results") / "split_empirical_outputs")
    parser.add_argument("--discounts", type=float, nargs="+", default=np.linspace(0., 1., 101),
                        help="Candidate discounts; default is every 1%% from 0%% through 100%%.")
    parser.add_argument("--demand-delta", type=float, default=.92)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--type-treatment", choices=["posterior-mean", "posterior-integrated"], default="posterior-mean",
                        help="Default treats estimated household types as known, matching base supply; optional posterior uncertainty sensitivity.")
    parser.add_argument("--save-state-policies", action="store_true", help="Export every optimized household/basket terminal coupon to compressed CSV.")
    parser.add_argument("--focal-flavor", choices=["all", *FLAVORS], default="all")
    parser.add_argument("--self-test", action="store_true", help="Generated synthetic inputs only.")
    parser.set_defaults(**vars(settings))
    return parser.parse_args()


# === 7. Run the three focal-product exercises, step by step === #

@server_only
def run_pricing(args):
    if args.self_test:
        self_test()
        return
    if args.household_input is None or args.fit_dir is None:
        raise ValueError("Provide --household-input AND --fit-dir; there are no default empirical input paths.")
    args.discounts = np.unique(np.asarray(args.discounts, dtype=float))
    if (not np.isfinite(args.discounts).all() or args.discounts[0] != 0. or args.discounts[-1] > 1.
            or not 0 <= args.demand_delta <= 1 or args.batch_size < 1):
        raise ValueError("Discounts must include zero and lie in [0,1]; delta in [0,1], batch size >=1.")
    args.output.mkdir(parents=True, exist_ok=True)
    output_root = args.output
    selected = FLAVORS if args.focal_flavor == "all" else [args.focal_flavor]
    reports, comparisons = {}, []
    for focal in selected:
        args.focal_flavor = focal
        args.output = output_root / focal
        args.output.mkdir(parents=True, exist_ok=True)
        reports[focal] = run_households(args)
        for row in reports[focal]["aggregate"]:
            comparisons.append(dict(focal_flavor=focal, **row))
    args.output = output_root
    args.focal_flavor = "all" if len(selected) == 3 else selected[0]
    pd.DataFrame(comparisons).to_csv(output_root / "focal_flavor_comparison.csv", index=False)
    report = reports
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(args).items()},
                    results=report, script_sha256=digest(Path(__file__).resolve()),
                    versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output / "run_notes.txt").write_text(pformat(manifest, sort_dicts=False, width=90)+"\n")
    print(f"Standalone empirical targeting completed: {args.output}")


if __name__ == "__main__":
    settings = SimpleNamespace(household_input=household_input, fit_dir=fit_dir,
        output=output, focal_flavor=focal_flavor, discounts=discounts,
        demand_delta=demand_delta, batch_size=batch_size, type_treatment=type_treatment,
        save_state_policies=save_state_policies, self_test=run_self_tests)
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_pricing(settings)
