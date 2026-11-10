#!/usr/bin/env python3
"""Split-history pricing for each focal flavor, separate from demand estimation.

User-supplied paths only. No estimation is performed. --self-test uses only
generated synthetic inputs. See SPLIT_TARGETING.md for definitions.
"""
from __future__ import annotations

# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

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


@server_only
def targeted_coupons(context, households, weights, alpha, switching, beta, discounts,
                     delta=1., capacity=7, control_coefficient=0.,
                     batch_size=24, type_treatment="posterior-mean", state_output=None):
    """Observed h0, then p1(h0), realized h1, and terminal p2(h1).

    The observed initialization and two priced choices form three choice dates.
    Individual backward induction; switching quartiles are reporting groups only.

    Baseline: posted prices in both weeks. Dynamic: choose first coupon accounting for optimized
    state-contingent terminal revenue. The seller observes the full basket.
    """
    baskets = compositions(capacity)
    inside = baskets[:, :3]
    totals = inside.sum(axis=1, keepdims=True)
    shares = np.divide(inside, totals, out=np.zeros_like(inside, dtype=float), where=totals > 0)
    coeff = math.lgamma(capacity+1)-np.vectorize(math.lgamma)(baskets+1).sum(axis=1)
    k, nstates = len(discounts), len(baskets)
    rows, maximum_mass_error, wrote_states = [], 0., False
    indices = np.flatnonzero(context["eligible"])
    # Only the two tails receive targeting; middle households remain at posted prices.
    targeted = households.group.isin(["low", "high"]).to_numpy()
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        h = len(ids)
        prices, availability = context["prices"][ids], context["available"][ids]
        stock, novelty, cf = context["stocks"][ids], context["novelty"][ids], context["cf"][ids]
        if type_treatment == "posterior-mean":
            # Base supply information assumption: an estimated fixed type is known.
            a, g = (weights[ids] @ alpha)[:, None], (weights[ids] @ switching)[:, None]
            w = np.ones((h, 1))
        else:
            a = np.broadcast_to(alpha, (h, len(alpha)))
            g = np.broadcast_to(switching, (h, *switching.shape))
            w = weights[ids]
        # g[...,0] is switching/novelty; g[...,1] is signed own-stock habit.
        history1 = g[..., 0, None]*novelty[:, None, :] + g[..., 1, None]*stock[:, None, :]
        novelty2 = shares.sum(axis=1, keepdims=True)-shares
        stock2 = stock[:, None, :]+shares[None, :, :]
        history2 = (g[..., 0, None, None]*novelty2[None, None, :, :]
                    +g[..., 1, None, None]*stock2[:, None, :, :])
        u1 = beta[None, None, :]-a[..., None]*prices[:, None, :]+history1+control_coefficient*cf[:, None, :]
        u2 = beta[None, None, None, :]-a[:, :, None, None]*prices[:, None, None, :]+history2+control_coefficient*cf[:, None, None, :]
        u1 = np.where(availability[:, None, :], u1, -np.inf)
        u2 = np.where(availability[:, None, None, :], u2, -np.inf)
        terminal = []
        for discount in discounts:
            u = u2.copy()
            u[..., 1] += a[:, :, None]*prices[:, None, None, 1]*discount
            denominator = logsumexp(np.concatenate([u, np.zeros((*u.shape[:-1], 1))], axis=-1), axis=-1)
            quantity = capacity*np.exp(u[..., 1]-denominator)
            terminal.append(np.stack([quantity*prices[:, None, None, 1]*(1-discount), quantity,
                                      quantity*prices[:, None, None, 1]*discount], axis=-1))
        # Metrics: net revenue, berry units, coupon expenditure.
        first = np.zeros((h, k, 3))
        second = np.zeros((h, k, 3))
        stage0_second = None
        paths = []
        for i, discount in enumerate(discounts):
            u = u1.copy()
            u[..., 1] += a*prices[:, None, 1]*discount
            full = np.concatenate([u, np.zeros((*u.shape[:-1], 1))], axis=-1)
            logp = full-logsumexp(full, axis=-1, keepdims=True)
            terms = np.zeros((h, a.shape[1], nstates, 4))
            np.multiply(baskets[None, None, :, :], logp[:, :, None, :], out=terms,
                        where=baskets[None, None, :, :] > 0)
            probability = np.exp(coeff[None, None, :]+terms.sum(axis=-1))
            maximum_mass_error = max(maximum_mass_error, float(np.abs(probability.sum(axis=-1)-1).max()))
            joint = w[:, :, None]*probability
            state_probability = joint.sum(axis=1)
            quantity = capacity*(w*np.exp(logp[..., 1])).sum(axis=1)
            first[:, i] = np.stack([quantity*prices[:, 1]*(1-discount), quantity,
                                    quantity*prices[:, 1]*discount], axis=-1)
            # Integrate types BEFORE choosing a terminal coupon: no foresight of
            # an unobserved true draw in the posterior-integrated sensitivity.
            metrics = np.stack([np.einsum("hds,hdsm->hsm", joint, candidate)
                                for candidate in terminal], axis=2)  # household,state,action,metric
            best = metrics[..., 0].argmax(axis=2)
            selected = np.take_along_axis(metrics, best[:, :, None, None], axis=2)[:, :, 0, :]
            # E_h1[p2(h1)*E(q2|h1,p2,type)] under this candidate p1(h0).
            # Optimize p2 separately in each state BEFORE summing its revenue.
            second[:, i] = selected.sum(axis=1)
            if i == 0:
                stage0_second = metrics[:, :, 0, :].sum(axis=1)
            # Storing these compact arrays allows exporting only chosen first actions.
            paths.append((best, selected, state_probability))
        zero_choice = np.zeros(h, dtype=int)
        dynamic_choice = (first[..., 0]+delta*second[..., 0]).argmax(axis=1)
        dynamic_choice = np.where(targeted[ids], dynamic_choice, 0)
        for name, choice in [("no_coupon", zero_choice), ("personalized_dynamic", dynamic_choice)]:
            for local, hh in enumerate(ids):
                action = int(choice[local])
                initial = first[local, action]
                is_target = targeted[hh] and name != "no_coupon"
                terminal_metrics = second[local, action] if is_target else stage0_second[local]
                best, selected, probability = paths[action]
                state_coupon = discounts[best[local]] if is_target else np.zeros(nstates)
                mass = probability[local]
                p1 = prices[local, 1]*(1-discounts[action])
                p2 = prices[local, 1]*(1-state_coupon)
                mean_p2 = float(mass @ p2)
                ratio = np.divide(p1, p2, out=np.full(nstates, np.nan), where=p2 > 0)
                positive_mass = mass > 1e-14
                expected_ratio = float(mass[positive_mass] @ ratio[positive_mass]) if np.isfinite(ratio[positive_mass]).all() else np.nan
                row = dict(household_code=households.iloc[hh].household_code,
                           group=households.iloc[hh].group, policy=name,
                           discount1=discounts[action], expected_discount2=float(mass @ state_coupon),
                           p1=p1, expected_p2=mean_p2,
                           ratio_of_expected_prices=p1/mean_p2 if mean_p2 > 0 else np.nan,
                           expected_price_ratio=expected_ratio,
                           revenue1=initial[0], revenue2=terminal_metrics[0],
                           discounted_revenue=initial[0]+delta*terminal_metrics[0],
                           berry_units1=initial[1], berry_units2=terminal_metrics[1],
                           coupon_expenditure=initial[2]+delta*terminal_metrics[2])
                for branch, label in [(baskets[:, 1] == 0, "no_berry"), (baskets[:, 1] > 0, "berry")]:
                    branch_mass = mass[branch].sum()
                    row[f"probability_{label}"] = branch_mass
                    row[f"mean_p2_given_{label}"] = float(mass[branch] @ p2[branch]/branch_mass) if branch_mass > 0 else np.nan
                rows.append(row)
                if state_output is not None and is_target:
                    detail = pd.DataFrame(dict(household_code=row["household_code"], group=row["group"], policy=name,
                                               state=np.arange(nstates), rival1_units=baskets[:, 0], focal_units=baskets[:, 1],
                                               rival2_units=baskets[:, 2], outside=baskets[:, 3], probability=mass,
                                               discount1=discounts[action], discount2=state_coupon, p1=p1, p2=p2, price_ratio=ratio))
                    detail.drop(columns=["household_code"]).to_csv(state_output, mode="a" if wrote_states else "w", header=not wrote_states,
                                  index=False, compression="gzip")
                    wrote_states = True
        if start % (batch_size*40) == 0 or start+batch_size >= len(indices):
            print(f"Targeted pricing: integrated {min(start+batch_size,len(indices))}/{len(indices)} eligible households", flush=True)
    if maximum_mass_error > 1e-9:
        raise AssertionError(f"Basket probability mass error: {maximum_mass_error}")
    table = pd.DataFrame(rows)
    baseline = table[table.policy == "no_coupon"].set_index("household_code").discounted_revenue
    table["baseline_revenue"] = table.household_code.map(baseline)
    table["revenue_change"] = table.discounted_revenue-table.baseline_revenue
    table["revenue_change_pct"] = 100*table.revenue_change/table.baseline_revenue
    wide = table.pivot(index="household_code", columns="policy", values="discounted_revenue")
    if (wide.personalized_dynamic+1e-8 < wide.no_coupon).any():
        raise AssertionError("Household dynamic policy does not dominate the feasible no-coupon baseline.")
    return table, maximum_mass_error

@server_only
def run_households(args):
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
    table, mass_error = targeted_coupons(
        context, households, weights, alpha, switching, beta, args.discounts,
        args.demand_delta, capacity,
        parameters.get("price_control", 0.) if control else 0., args.batch_size,
        args.type_treatment,
        args.output / "targeted_state_policies.csv.gz" if args.save_state_policies else None)
    households.rename(columns={"berry_eligible": "focal_eligible"}, inplace=True)
    table.rename(columns={c: c.replace("berry", "focal") for c in table if "berry" in c}, inplace=True)
    table["focal_flavor"] = focal
    households.drop(columns=["household_code"]).to_csv(args.output / "household_posterior_summary.csv", index=False)
    table.drop(columns=["household_code"]).to_csv(args.output / "targeted_household_results.csv", index=False)
    totals = ["revenue1", "revenue2", "discounted_revenue", "baseline_revenue", "revenue_change",
              "focal_units1", "focal_units2", "coupon_expenditure"]
    aggregate = table.groupby("policy", sort=False)[totals].sum()
    aggregate["revenue_change_pct"] = 100*aggregate.revenue_change/aggregate.baseline_revenue
    aggregate = aggregate.drop(index="no_coupon", errors="ignore")
    aggregate = aggregate.rename(index={"personalized_dynamic": "Personalized pricing", "optimized_common_coupons": "Common pricing"})
    aggregate.to_csv(args.output / "targeted_revenue_aggregate.csv")
    group = table.groupby(["policy", "group"], sort=False)[totals].sum()
    group["revenue_change_pct"] = 100*group.revenue_change/group.baseline_revenue
    means = table.groupby(["policy", "group"])[["p1", "expected_p2", "discount1", "expected_discount2",
                                                   "ratio_of_expected_prices", "expected_price_ratio"]].mean()
    group = group.join(means)
    group["eligible_households"] = table.groupby(["policy", "group"]).size()
    group = group.drop(index="no_coupon", level="policy", errors="ignore")
    group = group.rename(index={"personalized_dynamic": "Personalized pricing", "optimized_common_coupons": "Common pricing"})
    group.to_csv(args.output / "targeted_revenue_by_switching_group.csv")
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
                aggregate=aggregate.reset_index().to_dict("records"))


def make_targeting_figures(households, table, aggregate, output, delta=1.0, focal_flavor="berry"):
    import sys
    from pathlib import Path
    presentation_root = next(p for p in Path(__file__).resolve().parents if (p / "pricing_presentation.py").is_file())
    sys.path.insert(0, str(presentation_root))
    from pricing_presentation import empirical_figures
    return empirical_figures(table, output, focal_flavor, delta)


def self_test():
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_split_targeting.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError("Split targeting checks failed")


@server_only
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-input", type=Path, help="Explicit household input selected by you; no default.")
    parser.add_argument("--fit-dir", type=Path, help="Explicit existing fitted-results directory; no default.")
    parser.add_argument("--output", type=Path, default=Path("/scratch/dtm63837/Kilts_Panel/results") / "split_empirical_outputs")
    parser.add_argument("--discounts", type=float, nargs="+", default=np.linspace(0., 1., 101),
                        help="Candidate discounts; default is every 1%% from 0%% through 100%%.")
    parser.add_argument("--demand-delta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--type-treatment", choices=["posterior-mean", "posterior-integrated"], default="posterior-mean",
                        help="Default treats estimated household types as known, matching base supply; optional posterior uncertainty sensitivity.")
    parser.add_argument("--save-state-policies", action="store_true", help="Export every optimized household/basket terminal coupon to compressed CSV.")
    parser.add_argument("--focal-flavor", choices=["all", *FLAVORS], default="all")
    parser.add_argument("--self-test", action="store_true", help="Generated synthetic inputs only.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.household_input is None or args.fit_dir is None:
        parser.error("Provide --household-input AND --fit-dir; there are no default empirical input paths.")
    args.discounts = np.unique(np.asarray(args.discounts, dtype=float))
    if (not np.isfinite(args.discounts).all() or args.discounts[0] != 0. or args.discounts[-1] > 1.
            or not 0 <= args.demand_delta <= 1 or args.batch_size < 1):
        parser.error("Discounts must include zero and lie in [0,1]; delta in [0,1], batch size >=1.")
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
    print(f"Standalone empirical targeting completed: {args.output}")


if __name__ == "__main__":
    main()
