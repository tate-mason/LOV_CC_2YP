"""Empirical berry pricing. Edit Parameters, then run python3 demand/pricing.py.

The first observed choice becomes t=0 and establishes h0. The firm then chooses
p1(h0), observes a basket h1, and chooses terminal p2(h1). Only dynamic pricing
is optimized. No coupons is retained solely as a revenue benchmark.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys
import tempfile
from pprint import pformat

import numpy as np
import pandas as pd

# Use a writable plotting cache, including on a shared research server.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "final_simple_matplotlib"))

import math
import hashlib



# === Parameters: edit these values === #

# Select your actual saved weekly inputs and fitted-results folder.
# None deliberately prevents an accidental empirical run with guessed paths.
household_input = None              # Example: Path("/your/results/weekly_inputs.csv")
fit_dir = None                      # Example: Path("/your/results")
output = Path(__file__).resolve().parents[1] / "results" / "empirical_pricing"
discounts = np.linspace(0.0, 1.0, 101) # 0%, 1%, ..., 100% berry coupons.
demand_delta = 0.95                  # Revenue discount factor; independent of history weight.
batch_size = 24                     # Progress interval; each household is solved separately.
type_treatment = "posterior-mean"    # Or posterior-integrated as a sensitivity.
save_state_policies = False          # True saves every conditional terminal coupon.
run_self_tests = False                    # See checks/run_checks.py for checks without CSV access.
FLAVORS = ["other", "berry", "plain"]




# === 1. Stable probabilities and persistent household types === #



def expit(x):
    x = np.asarray(x)
    small = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+small), small/(1+small))

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x)
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)

def persistent_draws(number, seed):
    if number < 2 or number % 2:
        raise ValueError("Draw count must be even and >=2.")
    half = np.random.default_rng(seed).standard_normal((number//2, 2))
    return np.vstack([half, -half])

# Use each household's observed history to weight the possible types. Quartiles classify
# households for reporting and targeting.
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



# === 2. Possible baskets and the first observed choice as h0 === #



# List every basket of capacity purchase opportunities, including outside. With capacity=7
# there are 120 possible baskets.
def compositions(capacity):
    """All multinomial baskets of three inside flavors and outside."""
    return np.array([(a,b,c,capacity-a-b-c) for a in range(capacity+1)
                     for b in range(capacity-a+1) for c in range(capacity-a-b+1)], dtype=int)

# Treat the first observed basket as t=0, form h0, and price the next two weeks. Later
# baskets affect type estimation but do not enter h0.
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



# === 3. Backward induction: terminal p2(h1), then initial p1(h0) === #



# Solve the terminal coupon separately for every household and basket h1. Weight those
# revenues by the probabilities induced by each candidate first coupon.
def targeted_coupons(context, households, weights, alpha, gamma, beta, discounts,
                     delta=1., capacity=7, habit_weight=.7, control_coefficient=0.,
                     batch_size=24, type_treatment="posterior-mean", state_output=None):
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

    for number_done, household in enumerate(eligible_households, start=1):
        prices = context["prices"][household]
        available = context["available"][household]
        stock = context["stocks"][household]
        novelty = context["novelty"][household]
        control = context["cf"][household]

        # Default: the firm knows the household's estimated (alpha, gamma).
        # Sensitivity: retain the full probability distribution of possible types.
        if type_treatment == "posterior-mean":
            price_sensitivity = np.array([weights[household] @ alpha])
            history_coefficient = np.array([weights[household] @ gamma])
            type_weights = np.ones(1)
        else:
            price_sensitivity = alpha
            history_coefficient = gamma
            type_weights = weights[household]

        xi1 = (1 - habit_weight) * novelty - habit_weight * stock
        xi2 = ((1 - habit_weight) * basket_novelty
               - habit_weight * (stock + basket_shares))

        # u1: possible type x flavor. u2: possible type x basket h1 x flavor.
        utility1 = (beta - price_sensitivity[:, None] * prices
                    + history_coefficient[:, None] * xi1
                    + control_coefficient * control)
        utility2 = (beta - price_sensitivity[:, None, None] * prices
                    + history_coefficient[:, None, None] * xi2
                    + control_coefficient * control)
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
                    state=state_numbers, other=baskets[:, 0], berry=baskets[:, 1],
                    plain=baskets[:, 2], outside=baskets[:, 3], probability=state_probability,
                    discount1=discounts[action], discount2=terminal_coupon, p1=p1, p2=p2,
                    price_ratio=ratio))
                detail.to_csv(state_output, mode="a" if wrote_states else "w",
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
    return table, maximum_mass_error



# === 4. Read the explicitly selected fit, validate it, and save results === #



def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Read only the paths you selected. Reconstruct the saved likelihood before using its
# coefficients for counterfactual pricing.
def run_households(settings):
    # This branch is entered only when the person running the script supplies
    # BOTH paths. Validation supplies generated temporary fixtures only.
    fit = json.loads((settings.fit_dir / "fit.json").read_text())
    if fit["model"] != "heterogeneous" or not fit.get("converged", False):
        raise ValueError("This exercise requires a converged single-normal heterogeneous-gamma fit.")
    estimates = pd.read_csv(settings.fit_dir / "point_estimates.csv")
    if "at_bound" in estimates and estimates.at_bound.astype(str).str.lower().eq("true").any():
        raise ValueError("Fit has boundary parameters; review before running counterfactuals.")
    parameters = dict(zip(estimates.parameter, estimates.estimate))
    settings = fit["settings"]
    capacity = settings["capacity"]
    if not isinstance(capacity, int) or not 1 <= capacity <= 12:
        raise ValueError("Exact basket enumeration supports integer capacity 1..12.")
    frame = pd.read_csv(settings.household_input, dtype={"household_code": str}, parse_dates=["week"])
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
        context, households, weights, alpha, gamma, beta, settings.discounts,
        settings.demand_delta, capacity, weight,
        parameters.get("price_control", 0.) if control else 0., settings.batch_size,
        settings.type_treatment,
        settings.output / "targeted_state_policies.csv.gz" if settings.save_state_policies else None)
    households.to_csv(settings.output / "household_posterior_summary.csv", index=False)
    table.to_csv(settings.output / "targeted_household_results.csv", index=False)
    totals = ["revenue1", "revenue2", "discounted_revenue", "baseline_revenue", "revenue_change",
              "berry_units1", "berry_units2", "coupon_expenditure"]
    aggregate = table.groupby("policy", sort=False)[totals].sum()
    aggregate["revenue_change_pct"] = 100*aggregate.revenue_change/aggregate.baseline_revenue
    aggregate.to_csv(settings.output / "targeted_revenue_aggregate.csv")
    group = table.groupby(["policy", "group"], sort=False)[totals].sum()
    group["revenue_change_pct"] = 100*group.revenue_change/group.baseline_revenue
    means = table.groupby(["policy", "group"])[["p1", "expected_p2", "discount1", "expected_discount2",
                                                   "ratio_of_expected_prices", "expected_price_ratio"]].mean()
    group = group.join(means)
    group["eligible_households"] = table.groupby(["policy", "group"]).size()
    group.to_csv(settings.output / "targeted_revenue_by_gamma_group.csv")
    households.groupby("group").agg(households=("household_code", "size"),
                                    berry_eligible=("berry_eligible", "sum"),
                                    gamma_mean=("gamma_mean", "mean"),
                                    gamma_mean_min=("gamma_mean", "min"),
                                    gamma_mean_max=("gamma_mean", "max"),
                                    mean_posterior_sd=("gamma_sd", "mean"),
                                    mean_alpha=("alpha_mean", "mean"),
                                    median_posterior_ess=("posterior_ess", "median")).to_csv(settings.output / "household_group_summary.csv")
    make_targeting_figures(households, table, aggregate, settings.output, delta=settings.demand_delta)
    return dict(households=len(households), eligible_by_group=dict(zip(["low", "middle", "high"], counts)),
                reconstructed_nll=nll, basket_mass_error=mass_error,
                input_sha256=digest(settings.household_input), fit_sha256=digest(settings.fit_dir / "fit.json"),
                parameter_sha256=digest(settings.fit_dir / "point_estimates.csv"),
                posterior_draws=len(draws), posterior_seed=settings["seed"],
                capacity=capacity, habit_weight=weight, price_control=control,
                initial_history="first_observed_basket", pricing_periods=2,
                baskets=len(compositions(capacity)),
                discount_grid=settings.discounts.tolist(), demand_delta=settings.demand_delta,
                type_treatment=settings.type_treatment,
                aggregate=aggregate.reset_index().to_dict("records"))



# === 5. Figures with the same type colors throughout === #



def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

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



# === 6. Optional complete export check using generated inputs === #



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
        settings = argparse.Namespace(fit_dir=root, household_input=synthetic_input, output=root,
                                  discounts=discounts, demand_delta=1., batch_size=4,
                                  type_treatment="posterior-mean", save_state_policies=True)
        report = run_households(settings)
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



# === 7. Plain-text run notes === #



def write_run_notes(output, information):
    """Write the same run information in plain text; you never edit a JSON file."""
    output.mkdir(parents=True, exist_ok=True)
    text = "RUN INFORMATION\n\n"
    text += "This is an output record, not a settings file.\n"
    text += "Change the Python parameters at the top of the script to change the run.\n\n"
    text += pformat(information, sort_dicts=False, width=90)
    (output / "run_notes.txt").write_text(text + "\n")




# === 8. Optional command-line overrides === #



def read_command_line(settings):
    """Optional server flags. Normal use: edit the parameters at the top."""
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
    parser.set_defaults(**vars(settings))
    return parser.parse_args()




# === 9. Run the pricing exercise, step by step === #



def run_pricing(settings):
    """The actual research steps, in the order they run."""
    if settings.self_test:
        self_test()
        return
    if settings.household_input is None or settings.fit_dir is None:
        raise ValueError("Set household_input and fit_dir in the Parameters section, or supply --household-input and --fit-dir.")
    settings.discounts = np.unique(np.asarray(settings.discounts, dtype=float))
    if (not np.isfinite(settings.discounts).all() or settings.discounts[0] != 0. or settings.discounts[-1] > 1.
            or not 0 <= settings.demand_delta <= 1 or settings.batch_size < 1):
        raise ValueError("Discounts must include zero and lie in [0,1]; delta in [0,1], batch size >=1.")
    settings.output.mkdir(parents=True, exist_ok=True)
    report = run_households(settings)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(settings).items()},
                    results=report, script_sha256=digest(Path(__file__).resolve()),
                    versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    (settings.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_run_notes(settings.output, manifest)
    print(f"Standalone empirical targeting completed: {settings.output}")

if __name__ == "__main__":
    # Collect the ordinary Python variables above into one settings record.
    # settings.discounts means exactly the variable discounts from the Parameters section.
    settings = SimpleNamespace(
        household_input=household_input,
        fit_dir=fit_dir,
        output=output,
        discounts=discounts,
        demand_delta=demand_delta,
        batch_size=batch_size,
        type_treatment=type_treatment,
        save_state_policies=save_state_policies,
        self_test=run_self_tests,
    )
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_pricing(settings)
