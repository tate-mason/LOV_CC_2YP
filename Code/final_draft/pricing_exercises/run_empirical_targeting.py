#!/usr/bin/env python3
"""Standalone empirical berry targeting, separate from demand estimation.

User-supplied paths only. No estimation is performed. --self-test uses only
generated synthetic inputs. See EMPIRICAL_TARGETING.md for definitions.
"""
from __future__ import annotations
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
    half = np.random.default_rng(seed).standard_normal((number//2, 2))
    return np.vstack([half, -half])

def posterior(frame, parameters, draws, control=True):
    """Household-integrated multinomial likelihood; types persist across weeks."""
    labels = np.sort(frame.household_code.unique())
    hh = pd.Categorical(frame.household_code, categories=labels).codes
    prices = frame[["price_"+f for f in FLAVORS]].to_numpy(float)
    available = np.isfinite(prices) & (prices > 0)
    counts = frame[FLAVORS+["outside"]].to_numpy(float)
    xi = frame[["xi_"+f for f in FLAVORS]].to_numpy(float)
    cf = frame[["cf_"+f for f in FLAVORS]].to_numpy(float) if control else np.zeros_like(prices)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or not np.isfinite(xi).all() or not np.isfinite(cf).all()
            or ((counts[:, :3] > 0) & ~available).any()):
        raise ValueError("Invalid counts, history, controls, or purchased-alternative coverage.")
    beta = parameters["inside_constant"] + np.array([0., parameters["beta_berry"], parameters["beta_plain"]])
    alpha = np.exp(parameters["mu_log_minus_alpha"] + parameters["sd_log_minus_alpha"]*draws[:, 0])
    gamma = parameters["mean_gamma"] + parameters["sd_gamma"]*draws[:, 1]
    loglike = np.empty((len(labels), len(draws)))
    for d, (a, g) in enumerate(zip(alpha, gamma)):
        u = beta-a*np.nan_to_num(prices)+g*xi+parameters.get("price_control", 0.)*cf
        u = np.column_stack([np.where(available, u, -np.inf), np.zeros(len(frame))])
        logp = u-logsumexp(u, axis=1, keepdims=True)
        ll = (counts*np.where(counts > 0, logp, 0.)).sum(axis=1)
        loglike[:, d] = np.bincount(hh, weights=ll, minlength=len(labels))
    weights = np.exp(loglike-logsumexp(loglike, axis=1, keepdims=True))
    nll = -float((logsumexp(loglike, axis=1)-np.log(len(draws))).sum())
    means = weights @ gamma
    # Exactly floor(N/4) per tail. Household ID breaks ties reproducibly.
    order = np.argsort(means, kind="stable")
    groups = np.full(len(labels), "middle", dtype=object)
    n = len(labels)//4
    if n < 1:
        raise ValueError("Need at least four households to form quartiles.")
    groups[order[:n]], groups[order[-n:]] = "low", "high"
    population_q = np.quantile(gamma, [.25, .75])
    table = pd.DataFrame(dict(household_code=labels, group=groups, gamma_mean=means,
                             gamma_sd=np.sqrt(np.maximum(weights @ gamma**2-means**2, 0)),
                             alpha_mean=weights @ alpha, posterior_ess=1/(weights**2).sum(axis=1),
                             probability_gamma_below_draw_q25=weights @ (gamma <= population_q[0]),
                             probability_gamma_above_draw_q75=weights @ (gamma >= population_q[1])))
    return table, weights, alpha, gamma, beta, nll

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


def targeted_coupons(context, households, weights, alpha, gamma, beta, discounts,
                     delta=1., capacity=7, habit_weight=.7, control_coefficient=0.,
                     batch_size=24, type_treatment="posterior-mean", state_output=None):
    """Observed h0, then p1(h0), realized h1, and terminal p2(h1).

    The observed initialization and two priced choices form three choice dates.
    Individual backward induction; gamma quartiles are reporting groups only.

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
            a, g = (weights[ids] @ alpha)[:, None], (weights[ids] @ gamma)[:, None]
            w = np.ones((h, 1))
        else:
            a = np.broadcast_to(alpha, (h, len(alpha)))
            g = np.broadcast_to(gamma, (h, len(gamma)))
            w = weights[ids]
        xi1 = (1-habit_weight)*novelty-habit_weight*stock
        xi2 = (1-habit_weight)*(shares.sum(axis=1, keepdims=True)-shares)[None, :, :] - habit_weight*(stock[:, None, :]+shares[None, :, :])
        u1 = beta[None, None, :]-a[..., None]*prices[:, None, :]+g[..., None]*xi1[:, None, :]+control_coefficient*cf[:, None, :]
        u2 = beta[None, None, None, :]-a[:, :, None, None]*prices[:, None, None, :]+g[:, :, None, None]*xi2[:, None, :, :]+control_coefficient*cf[:, None, None, :]
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
                                               state=np.arange(nstates), other=baskets[:, 0], berry=baskets[:, 1],
                                               plain=baskets[:, 2], outside=baskets[:, 3], probability=mass,
                                               discount1=discounts[action], discount2=state_coupon, p1=p1, p2=p2, price_ratio=ratio))
                    detail.to_csv(state_output, mode="a" if wrote_states else "w", header=not wrote_states,
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

def run_households(args):
    # This branch is entered only when the person running the script supplies
    # BOTH paths. Validation supplies generated temporary fixtures only.
    fit = json.loads((args.fit_dir / "fit.json").read_text())
    if fit["model"] != "heterogeneous" or not fit.get("converged", False):
        raise ValueError("This exercise requires a converged single-normal heterogeneous-gamma fit.")
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
    required += [prefix+f for prefix in ["price_", "stock_", "novelty_", "xi_"] for f in FLAVORS]
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
    weight = settings.get("habit_weight", .7)
    for f in FLAVORS:
        expected = (1-weight)*frame["novelty_"+f]-weight*frame["stock_"+f]
        if not np.allclose(expected, frame["xi_"+f], atol=1e-8):
            raise ValueError("Saved Xi does not match the fit's history definition.")
    draws = persistent_draws(settings["draws"], settings["seed"])
    households, weights, alpha, gamma, beta, nll = posterior(frame, parameters, draws, control)
    # Exact likelihood reproduction guards against silently pairing files from different fits.
    if not np.isclose(nll, fit["nll"], rtol=1e-9, atol=1e-4):
        raise ValueError(f"Likelihood mismatch: reconstructed {nll}, fit {fit['nll']}.")
    context = forecast_context(frame, households)
    counts = [int(((households.group == g) & households.berry_eligible).sum()) for g in ["low", "middle", "high"]]
    if any(n == 0 for n in counts):
        raise ValueError("Each gamma segment must have berry-eligible forecast households.")
    table, mass_error = targeted_coupons(
        context, households, weights, alpha, gamma, beta, args.discounts,
        args.demand_delta, capacity, weight,
        parameters.get("price_control", 0.) if control else 0., args.batch_size,
        args.type_treatment,
        args.output / "targeted_state_policies.csv.gz" if args.save_state_policies else None)
    households.to_csv(args.output / "household_posterior_summary.csv", index=False)
    table.to_csv(args.output / "targeted_household_results.csv", index=False)
    totals = ["revenue1", "revenue2", "discounted_revenue", "baseline_revenue", "revenue_change",
              "berry_units1", "berry_units2", "coupon_expenditure"]
    aggregate = table.groupby("policy", sort=False)[totals].sum()
    aggregate["revenue_change_pct"] = 100*aggregate.revenue_change/aggregate.baseline_revenue
    aggregate.to_csv(args.output / "targeted_revenue_aggregate.csv")
    group = table.groupby(["policy", "group"], sort=False)[totals].sum()
    group["revenue_change_pct"] = 100*group.revenue_change/group.baseline_revenue
    means = table.groupby(["policy", "group"])[["p1", "expected_p2", "discount1", "expected_discount2",
                                                   "ratio_of_expected_prices", "expected_price_ratio"]].mean()
    group = group.join(means)
    group["eligible_households"] = table.groupby(["policy", "group"]).size()
    group.to_csv(args.output / "targeted_revenue_by_gamma_group.csv")
    households.groupby("group").agg(households=("household_code", "size"),
                                    berry_eligible=("berry_eligible", "sum"),
                                    gamma_mean=("gamma_mean", "mean"),
                                    gamma_mean_min=("gamma_mean", "min"),
                                    gamma_mean_max=("gamma_mean", "max"),
                                    mean_posterior_sd=("gamma_sd", "mean"),
                                    mean_alpha=("alpha_mean", "mean"),
                                    median_posterior_ess=("posterior_ess", "median")).to_csv(args.output / "household_group_summary.csv")
    make_targeting_figures(households, table, aggregate, args.output, delta=args.demand_delta)
    return dict(households=len(households), eligible_by_group=dict(zip(["low", "middle", "high"], counts)),
                reconstructed_nll=nll, basket_mass_error=mass_error,
                input_sha256=digest(args.household_input), fit_sha256=digest(args.fit_dir / "fit.json"),
                parameter_sha256=digest(args.fit_dir / "point_estimates.csv"),
                posterior_draws=len(draws), posterior_seed=settings["seed"],
                capacity=capacity, habit_weight=weight, price_control=control,
                initial_history="first_observed_basket", pricing_periods=2,
                baskets=len(compositions(capacity)),
                discount_grid=args.discounts.tolist(), demand_delta=args.demand_delta,
                type_treatment=args.type_treatment,
                aggregate=aggregate.reset_index().to_dict("records"))


def make_targeting_figures(households, table, aggregate, output, delta=.95):
    """Plot household types and individually optimized berry discounts."""
    plt = plotting()
    groups = ["low", "middle", "high"]
    labels = ["Low gamma\n(bottom 25%)", "Middle gamma\n(middle 50%)", "High gamma\n(top 25%)"]
    colors = ["#3575AD", "#999999", "#D17925"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.histogram_bin_edges(households.gamma_mean, bins=30)
    for g, label, color in zip(groups, labels, colors):
        axes[0].hist(households.loc[households.group == g, "gamma_mean"],
                     bins=bins, alpha=.65, label=label.replace("\n", " "), color=color)
    axes[0].set(title="Estimated consumer types", xlabel=r"Estimated household type (posterior mean $\gamma$)",
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
             "Low/high types receive personalized berry offers; middle types receive no coupons.",
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
           ylabel="Mean optimal berry discount (% of posted price)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Dynamic pricing: optimal discounts by consumer type")
    fig.text(.5, .015, "Household-weighted means among berry-eligible households. Middle group receives no coupons by policy design.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"discounts_by_consumer_type.{extension}", dpi=180)
    plt.close(fig)

    dynamic = table.loc[table.policy == "personalized_dynamic"].merge(
        households[["household_code", "gamma_mean", "alpha_mean"]], on="household_code", validate="one_to_one")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for column, (variable, xlabel) in enumerate([
            ("gamma_mean", r"Estimated household type (posterior mean $\gamma$)"),
            ("alpha_mean", r"Estimated price sensitivity (posterior mean $a$)")]):
        for row, (discount, title) in enumerate([
                ("discount1", r"Optimal discount at $p_1(h_0)$"),
                ("expected_discount2", r"Expected terminal discount at $p_2(h_1)$")]):
            ax = axes[row, column]
            for g, label, color in zip(groups, labels, colors):
                subset = dynamic.loc[dynamic.group == g]
                ax.scatter(subset[variable], 100*subset[discount], s=12, alpha=.5,
                           color=color, label=label.replace("\n", " "))
            ax.set(xlabel=xlabel, ylabel="Berry discount (% of posted price)", title=title, ylim=(-2, 102))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Dynamic pricing: household discounts and estimated consumer characteristics")
    fig.text(.5, .015, "Each point is one berry-eligible household. Terminal offers average over the household's realized h1 histories.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"discounts_by_household_type.{extension}", dpi=180)
    plt.close(fig)


def self_test():
    """Generated synthetic households only; no user-supplied input is accessed."""
    rng = np.random.default_rng(41)
    parameters = dict(inside_constant=.8, beta_berry=-.2, beta_plain=-.1,
                      mu_log_minus_alpha=-.3, sd_log_minus_alpha=.4,
                      mean_gamma=-.1, sd_gamma=.2)
    rows = []
    for household in range(12):
        stock, previous = np.zeros(3), np.zeros(3)
        for t in range(3):
            counts = rng.multinomial(2, [.25, .3, .2, .25])
            novelty = previous.sum()-previous
            row = dict(household_code=f"synthetic_{household:02}",
                       week=pd.Timestamp("2020-01-04")+pd.Timedelta(weeks=t), outside=counts[3])
            for j, flavor in enumerate(FLAVORS):
                row.update({flavor: counts[j], "price_"+flavor: 1.+.1*j,
                            "stock_"+flavor: stock[j], "novelty_"+flavor: novelty[j],
                            "xi_"+flavor: .3*novelty[j]-.7*stock[j]})
            rows.append(row)
            previous = counts[:3]/max(counts[:3].sum(), 1)
            stock += previous
    frame = pd.DataFrame(rows)
    draws = persistent_draws(8, 19)
    households, weights, alpha, gamma, beta, nll = posterior(frame, parameters, draws, False)
    assert np.allclose(weights.sum(axis=1), 1.)
    scalar_ll = []
    for a, g in zip(alpha, gamma):
        value = 0.
        for row in rows[:3]:
            u = [beta[j]-a*row["price_"+f]+g*row["xi_"+f] for j, f in enumerate(FLAVORS)]+[0.]
            value += np.array([row[f] for f in FLAVORS]+[row["outside"]]) @ (np.array(u)-logsumexp(u))
        scalar_ll.append(value)
    assert np.allclose(weights[0], np.exp(scalar_ll-logsumexp(scalar_ll)))
    context = forecast_context(frame, households)
    discounts = np.array([0., .25, .5])
    for treatment in ["posterior-mean", "posterior-integrated"]:
        table, error = targeted_coupons(context, households, weights, alpha, gamma, beta, discounts,
                                        capacity=2, batch_size=4, type_treatment=treatment)
        assert error < 1e-12 and len(table) == 24
        # Accounting and treating the middle segment as a holdout.
        assert np.allclose(table.discounted_revenue+table.coupon_expenditure,
                           1.1*(table.berry_units1+table.berry_units2))
        assert np.allclose(table.loc[table.group == "middle", "revenue_change"], 0.)
    no_history, _ = targeted_coupons(context, households, weights, np.ones_like(alpha),
                                     np.zeros_like(gamma), beta, discounts, capacity=2)
    assert np.allclose(no_history.revenue1, no_history.revenue2)
    assert np.allclose(no_history.discount1, no_history.expected_discount2)
    # Individual actions must be invariant to renaming the low/high reporting groups.
    swapped = households.copy()
    swapped["group"] = swapped.group.replace({"low": "high", "high": "low"})
    changed, _ = targeted_coupons(context, swapped, weights, alpha, gamma, beta, discounts, capacity=2)
    original, _ = targeted_coupons(context, households, weights, alpha, gamma, beta, discounts, capacity=2)
    assert np.allclose(changed.discount1, original.discount1)
    assert np.allclose(changed.discounted_revenue, original.discounted_revenue)
    assert len(compositions(7)) == 120
    restricted = {k: v.copy() for k,v in context.items()}
    restricted["available"][:, 2] = False
    restricted["prices"][:, 2] = 0.
    _, error = targeted_coupons(restricted, households, weights, alpha, gamma, beta,
                                discounts[:1], capacity=7, batch_size=4)
    assert error < 1e-12
    with tempfile.TemporaryDirectory(prefix="targeting_synthetic_") as directory:
        root = Path(directory)
        synthetic_input = root / "synthetic_records.csv"
        frame.to_csv(synthetic_input, index=False)
        pd.DataFrame({"parameter": list(parameters), "estimate": list(parameters.values()), "at_bound": False}).to_csv(root / "point_estimates.csv", index=False)
        fit = dict(model="heterogeneous", converged=True, nll=nll, weeks=len(frame), households=12,
                   settings=dict(capacity=2, draws=8, seed=19, no_price_control=True, habit_weight=.7))
        (root / "fit.json").write_text(json.dumps(fit))
        args = argparse.Namespace(fit_dir=root, household_input=synthetic_input, output=root,
                                  discounts=discounts, demand_delta=1., batch_size=4,
                                  type_treatment="posterior-mean", save_state_policies=True)
        report = run_households(args)
        assert report["households"] == 12
        states = pd.read_csv(root / "targeted_state_policies.csv.gz")
        assert np.allclose(states.groupby(["household_code", "policy"]).probability.sum(), 1.)
        # Independent exhaustive two-period calculation for the first targeted
        # household: all first actions and all terminal actions at each basket.
        household = int(np.flatnonzero(households.group.eq("low").to_numpy())[0])
        a, g = weights[household] @ alpha, weights[household] @ gamma
        basket_rows = compositions(2)
        values = []
        for d1 in discounts:
            prices = context["prices"][household].copy()
            prices[1] *= 1-d1
            xi = .3*context["novelty"][household]-.7*context["stocks"][household]
            u = np.r_[beta-a*prices+g*xi, 0.]
            prob = np.exp(u-logsumexp(u))
            value = 2*prob[1]*prices[1]
            continuation = 0.
            for basket in basket_rows:
                mass = math.factorial(2)/math.prod(math.factorial(int(n)) for n in basket)*np.prod(prob**basket)
                x = basket[:3]/max(basket[:3].sum(), 1)
                xi2 = .3*(x.sum()-x)-.7*(context["stocks"][household]+x)
                returns = []
                for d2 in discounts:
                    second_prices = context["prices"][household].copy()
                    second_prices[1] *= 1-d2
                    u2 = np.r_[beta-a*second_prices+g*xi2, 0.]
                    prob2 = np.exp(u2-logsumexp(u2))
                    returns.append(2*prob2[1]*second_prices[1])
                continuation += mass*max(returns)
            values.append(value+continuation)
        actual = pd.read_csv(root / "targeted_household_results.csv")
        chosen = actual[(actual.household_code == households.iloc[household].household_code) & actual.policy.eq("personalized_dynamic")].iloc[0]
        assert np.isclose(chosen.discounted_revenue, max(values))
    print("Synthetic targeting checks passed: posterior likelihood, per-household optimization, full-basket backward induction, reporting-group invariance, policy dominance, missing menus, exact basket mass, exports.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-input", type=Path, help="Explicit household input selected by you; no default.")
    parser.add_argument("--fit-dir", type=Path, help="Explicit existing fitted-results directory; no default.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "empirical_outputs")
    parser.add_argument("--discounts", type=float, nargs="+", default=np.linspace(0., 1., 101),
                        help="Candidate discounts; default is every 1%% from 0%% through 100%%.")
    parser.add_argument("--demand-delta", type=float, default=.95)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--type-treatment", choices=["posterior-mean", "posterior-integrated"], default="posterior-mean",
                        help="Default treats estimated household types as known, matching base supply; optional posterior uncertainty sensitivity.")
    parser.add_argument("--save-state-policies", action="store_true", help="Export every optimized household/basket terminal coupon to compressed CSV.")
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
    report = run_households(args)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(args).items()},
                    results=report, script_sha256=digest(Path(__file__).resolve()),
                    versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Standalone empirical targeting completed: {args.output}")


if __name__ == "__main__":
    main()
