"""Two-period monopoly with myopic logit consumers and variety preferences.

Run python3 toy_model.py --help. Prices are net transaction prices. A period-one
outside choice exits permanently. Consumers have no prior purchase, so variety
utility begins in period two. Known type masses are fixed. Profits are per
initial consumer. 'Uniform' means common across types, not across products.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

beta_0 = np.array([2.2, 2.0])
gamma_H, gamma_L = 2.0, 0.1
alpha, delta = -1.5, 0.95
costs = np.array([1.2, 1.0])
J = len(costs)
PRICE_MIN, PRICE_MAX = 1.0, 8.0


def logit_probs(V):
    full = np.r_[0.0, V]
    e = np.exp(full - full.max())
    prob = e / e.sum()
    return prob[1:], float(prob[0])


def val_state(p, prev_j, gamma_r):
    return beta_0 + gamma_r * (np.arange(J) != prev_j) + alpha * np.asarray(p)


def shares_jac(p, gamma, state=None):
    """Logit shares and price derivatives; no variety bonus before a purchase."""
    utility = beta_0 + alpha * np.asarray(p) if state is None else val_state(p, state, gamma)
    s, _ = logit_probs(utility)
    return s, alpha * (np.diag(s) - np.outer(s, s))


def maximize(fun):
    """Bounded multistart optimization with analytic gradients and status checks."""
    candidates = []
    for markup in (0.3, 1.2, 3.0):
        res = minimize(
            lambda p: -fun(p)[0],
            costs + markup,
            jac=lambda p: -fun(p)[1],
            method="L-BFGS-B",
            bounds=[(PRICE_MIN, PRICE_MAX)] * J,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 400},
        )
        if res.success:
            candidates.append(res)
    if not candidates:
        raise RuntimeError("All pricing optimization starts failed")
    return min(candidates, key=lambda res: res.fun).x


@dataclass
class Policy:
    p1: np.ndarray
    p2: np.ndarray  # history j, destination k
    profit: float
    type_profits: np.ndarray


def evaluate_policy(p1, p2, gammas):
    """Expected discounted profit conditional on each type, before the first purchase."""
    values = []
    for gamma in gammas:
        s1, _ = shares_jac(p1, gamma)
        w = np.array(
            [shares_jac(p2[j], gamma, j)[0] @ (p2[j] - costs) for j in range(J)]
        )
        values.append(s1 @ (p1 - costs + delta * w))
    return np.array(values)


def solve_common(gammas, weights):
    """Commit to all prices using known, fixed population type masses.

    History indexes a preannounced menu; no beliefs are updated after purchases.
    """
    gammas, weights = np.asarray(gammas), np.asarray(weights, dtype=float)
    if (
        len(gammas) != len(weights)
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Nonnegative type weights must sum to one")

    def objective(flat):
        p1, p2 = flat[:J], flat[J:].reshape(J, J)
        value, grad1, grad2 = 0.0, np.zeros(J), np.zeros((J, J))
        for gamma, weight in zip(gammas, weights):
            s1, jac1 = shares_jac(p1, gamma)
            w = np.zeros(J)
            for h in range(J):
                s2, jac2 = shares_jac(p2[h], gamma, h)
                margin = p2[h] - costs
                w[h] = s2 @ margin
                grad2[h] += delta * weight * s1[h] * (s2 + jac2.T @ margin)
            effective = p1 - costs + delta * w
            value += weight * (s1 @ effective)
            grad1 += weight * (s1 + jac1.T @ effective)
        return value, np.r_[grad1, grad2.ravel()]

    candidates = []
    for markup in (0.3, 1.2, 3.0):
        start = np.tile(costs + markup, J + 1)
        result = minimize(
            lambda x: -objective(x)[0],
            start,
            jac=lambda x: -objective(x)[1],
            method="L-BFGS-B",
            bounds=[(PRICE_MIN, PRICE_MAX)] * len(start),
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 400},
        )
        if result.success:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("All common-menu optimization starts failed")
    best = min(candidates, key=lambda result: result.fun)
    p1, p2 = best.x[:J], best.x[J:].reshape(J, J)
    by_type = evaluate_policy(p1, p2, gammas)
    return Policy(p1, p2, weights @ by_type, by_type)


def solve_perfect_info(l1):
    """Legacy tuple interface; each type is solved even at zero population weight."""
    policies = [solve_common([g], [1.0]) for g in (gamma_H, gamma_L)]
    return (
        np.array([p.p1 for p in policies]),
        np.array([p.p2 for p in policies]),
        l1 * policies[0].profit + (1 - l1) * policies[1].profit,
    )


def solve_imperfect_info(l1):
    p = solve_common([gamma_H, gamma_L], [l1, 1 - l1])
    return p.p1, p.p2, p.profit


# Compatibility names for the no-prior-purchase model used by the discount analysis.
solve_newborn = solve_common
evaluate_newborn = evaluate_policy


def newborn_discount_analysis(output):
    """Choose between fixed high-type regular and low-type discounted menus.

    The discount is a period-two offer; the first-period price is held fixed.
    q is Pr(L), not the probability of randomly sending an offer. At q=1/2
    discount recipients are equally likely to be H or L. Sending probability
    itself yields an affine objective and has no interior optimum.
    """
    types = [gamma_H, gamma_L]
    regular = solve_newborn([gamma_H], [1.0])
    discounted = solve_newborn([gamma_L], [1.0])
    # Compare terminal discounts only: high types receive introductory investment
    # in period one, so their first price need not exceed low-type prices.
    discounted.p1 = regular.p1.copy()
    if np.any(discounted.p1 > regular.p1 + 1e-7) or np.any(
        discounted.p2 > regular.p2 + 1e-7
    ):
        raise RuntimeError(
            "Low-type menu is not a pure discount from the high-type menu"
        )
    regular_values = evaluate_newborn(regular.p1, regular.p2, types)
    discount_values = evaluate_newborn(discounted.p1, discounted.p2, types)
    gains = discount_values - regular_values
    slope = gains[1] - gains[0]
    root = -gains[0] / slope if abs(slope) > 1e-12 else np.nan
    threshold = root if 0 <= root <= 1 else np.nan
    rows = []
    for q in np.arange(1, 100) / 100:
        common = solve_newborn(types, [1 - q, q])
        regular_profit = (1 - q) * regular_values[0] + q * regular_values[1]
        discount_profit = (1 - q) * discount_values[0] + q * discount_values[1]
        rows.append(
            dict(
                probability_L=q,
                probability_H=1 - q,
                regular_profit=regular_profit,
                discount_profit=discount_profit,
                discount_gain=discount_profit - regular_profit,
                offer_discount=discount_profit > regular_profit + 1e-9,
                best_fixed_menu_profit=max(regular_profit, discount_profit),
                reoptimized_common_profit=common.profit,
                common_p1_good1=common.p1[0],
                common_p1_good2=common.p1[1],
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "newborn_discount_probability.csv", index=False)
    price_rows = []
    common50 = solve_newborn(types, [0.5, 0.5])
    for name, policy in [
        ("regular_H_menu", regular),
        ("discount_L_menu", discounted),
        ("newborn_common_50_50", common50),
    ]:
        for period, history, price in [(1, "no prior product", policy.p1)] + [
            (2, f"good{j + 1}", policy.p2[j]) for j in range(J)
        ]:
            price_rows.append(
                dict(
                    menu=name,
                    period=period,
                    history=history,
                    price_good1=price[0],
                    price_good2=price[1],
                )
            )
    pd.DataFrame(price_rows).to_csv(output / "newborn_prices.csv", index=False)
    report = (
        "# Newborn discount decision\n\n"
        "The newborn has no previous product. Their type is unobserved and "
        "Pr(L)=q, Pr(H)=1-q. The first purchase has no switching bonus and therefore "
        "does not reveal type. Type remains unobserved in period two.\n\n"
        "The regular menu is the two-period policy optimized for a known H type. "
        "The discount menu replaces terminal prices with the known-L optimum while keeping "
        "the regular first-period price. All its prices "
        "are verified to be weakly below regular prices. Both menus are fixed while q varies. "
        "Period-two offers depend on the first product purchased. The firm knows both "
        "types exogenously for purposes of constructing these benchmark menus.\n\n"
        f"Profit change from discounting an H newborn: {gains[0]:.8f}\n\n"
        f"Profit change from discounting an L newborn: {gains[1]:.8f}\n\n"
        f"Break-even Pr(L): {threshold:.8f}\n\n"
        f"Break-even Pr(H): {1 - threshold:.8f}\n\n"
        f"50/50 regular profit: {regular_values.mean():.8f}\n\n"
        f"50/50 discount profit: {discount_values.mean():.8f}\n\n"
        f"50/50 reoptimized common profit: {common50.profit:.8f}\n\n"
        "Expected gain = (1-q)*Delta_H + q*Delta_L. The exact cutoff is "
        "-Delta_H/(Delta_L-Delta_H). Offer the fixed discount when the expected "
        "gain is positive; at zero the firm is indifferent. No fixed adoption cost "
        "or classifier error is added.\n\n"
        "The CSV sweeps q from 0.01 through 0.99 and also solves freely reoptimized "
        "common prices at every q. The fixed-menu cutoff does not imply a discontinuity "
        "in those freely optimized prices. It depends on the specified regular and "
        "discount menus.\n\n"
        "If the probability being varied instead means the random offer probability a, "
        "expected profit is regular_profit + a*(discount_profit-regular_profit). "
        "Then the optimum is a=0 or a=1 (or any a at indifference), rather than an "
        "interior threshold in a.\n"
    )
    (output / "newborn_results.md").write_text(report)
    print(report)
    return frame, threshold


def fixed_menu_regions(type_values):
    """Optimal regions among fixed menus; columns are profits for true H and L.

    q is the probability of H. Pairwise crossings are boundaries only when
    they change the upper envelope, so dominated crossings are excluded.
    """
    values = np.asarray(type_values, dtype=float)
    intercepts = values[:, 1]
    slopes = values[:, 0] - intercepts
    breaks = [0.0, 1.0]
    for a in range(len(values)):
        for b in range(a):
            difference = slopes[a] - slopes[b]
            if abs(difference) > 1e-12:
                q = (intercepts[b] - intercepts[a]) / difference
                if 0 < q < 1:
                    breaks.append(float(q))
    breaks = sorted(set(breaks))
    regions = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        winner = int(np.argmax(intercepts + (left + right) / 2 * slopes))
        if regions and regions[-1][2] == winner:
            regions[-1] = (regions[-1][0], right, winner)
        else:
            regions.append((left, right, winner))
    return regions


def fixed_menu_analysis(output, personal, uniform, gamma_grid):
    """Compare H, L, and fixed 50/50 menus without reoptimization."""
    import matplotlib.pyplot as plt

    labels = ["high", "low", "uniform_50_50"]
    policies = [*personal, uniform]
    values = np.array([
        evaluate_policy(p.p1, p.p2, [gamma_H, gamma_L]) for p in policies
    ])
    regions = fixed_menu_regions(values)
    pd.DataFrame([
        dict(menu=name, profit_true_H=v[0], profit_true_L=v[1])
        for name, v in zip(labels, values)
    ]).to_csv(output / "fixed_menu_type_profits.csv", index=False)
    pd.DataFrame([
        dict(probability_H_min=left, probability_H_max=right, best_menu=labels[i])
        for left, right, i in regions
    ]).to_csv(output / "fixed_menu_regions.csv", index=False)
    q = np.unique(np.r_[np.linspace(0, 1, 1001), [r[0] for r in regions]])
    profits = q[:, None] * values[:, 0] + (1 - q[:, None]) * values[:, 1]
    frame = pd.DataFrame(profits, columns=[f"profit_{name}" for name in labels])
    frame.insert(0, "probability_H", q)
    frame["best_menu"] = np.array(labels)[profits.argmax(axis=1)]
    frame.to_csv(output / "fixed_menu_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, name in enumerate(labels):
        axes[0].plot(q, profits[:, i], label=name)
    axes[1].plot(q, profits[:, 0] - profits[:, 2], label="H menu minus uniform")
    axes[1].plot(q, profits[:, 1] - profits[:, 2], label="L menu minus uniform")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for left, _, _ in regions[1:]:
        for ax in axes:
            ax.axvline(left, color="gray", linestyle=":", label=f"Transition {left:.5f}")
    for ax, ylabel in zip(axes, ["Expected discounted profit", "Profit gain over uniform"]):
        ax.set(xlabel="Probability customer is H", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fixed_menu_transitions.png", dpi=180)
    fig.savefig(output / "fixed_menu_transitions.pdf")
    plt.close(fig)

    gamma_values = np.array([
        evaluate_policy(p.p1, p.p2, gamma_grid) for p in policies
    ]).T
    gamma_frame = pd.DataFrame(
        gamma_values, columns=[f"profit_{name}" for name in labels]
    )
    gamma_frame.insert(0, "gamma", gamma_grid)
    gamma_frame.to_csv(output / "fixed_menu_profit_by_gamma.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    display_names = ["High-type menu", "Low-type menu", "Uniform 50/50 menu"]
    for i, name in enumerate(display_names):
        axes[0].plot(q, profits[:, i], label=name)
        axes[2].plot(gamma_grid, gamma_values[:, i], label=name)
    axes[1].plot(q, profits[:, 0] - profits[:, 2], label="High menu minus uniform")
    axes[1].plot(q, profits[:, 1] - profits[:, 2], label="Low menu minus uniform")
    axes[1].plot(q, profits.max(axis=1) - profits[:, 2],
                 color="black", linestyle="--", label="Best menu minus uniform")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for left, _, _ in regions[1:]:
        for ax in axes[:2]:
            ax.axvline(left, color="gray", linestyle=":")
    for gamma, name in [(gamma_L, "L"), (gamma_H, "H")]:
        axes[2].axvline(gamma, color="gray", linestyle=":", label=f"{name}: gamma={gamma:g}")
    axes[0].set(xlabel="Probability customer is H", ylabel="Profit per initial consumer")
    axes[1].set(xlabel="Probability customer is H", ylabel="Profit gain over uniform")
    axes[2].set(xlabel="Variety preference gamma", ylabel="Conditional expected profit")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fixed_menu_profits.png", dpi=180)
    fig.savefig(output / "fixed_menu_profits.pdf")
    plt.close(fig)

    lines = [
        "# Three fixed pricing menus\n",
        "H and L menus maximize profit for known H and known L respectively. "
        "The uniform menu maximizes profit for a 50/50 mix. Each menu allows "
        "product, period, and terminal purchase-history variation. All three "
        "menus remain fixed as q = Pr(H) varies. Consumers have no prior purchase; "
        "variety utility begins in period two.\n",
        "Expected profit for menu m is q*profit_m(H) + (1-q)*profit_m(L). "
        "For two menus a and b, let D_H=profit_a(H)-profit_b(H), and similarly "
        "D_L. Their crossing is q=-D_L/(D_H-D_L). Only crossings on the "
        "upper envelope change the optimal menu.\n",
    ]
    for left, right, i in regions:
        lines.append(f"- {left:.8f} < Pr(H) < {right:.8f}: {labels[i]} menu.")
    lines += [
        "\nAdjacent menus tie at each internal boundary; endpoint choices follow "
        "the adjacent region. There is no targeting cost. With type observed, "
        "choose H at q=1 and L at q=0. With unrestricted reoptimization at every "
        "q, the optimal common menu weakly dominates these three candidates; "
        "these thresholds describe a restricted choice among fixed menus.\n",
        "Both known-type terminal optima have equal product margins at each "
        "history: p_rk2(h)-c_k = -1/(alpha*s_r02(h)) for interior prices. "
        "Thus a higher-margin destination good is not implied by this model. "
        "The H menu has higher terminal prices than the L menu in this calibration; "
        "both are monopoly optima for their respective types. A switch from the "
        "uniform menu can raise some prices and lower others.\n",
        "See prices_50_50.csv for the full menus and fixed_menu_type_profits.csv "
        "for the cross-type profit evaluations.\n",
    ]
    report = "\n".join(lines)
    (output / "fixed_menu_results.md").write_text(report)
    print(report)


def run_analysis(output, points=21, gamma_max=3.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    types = [gamma_H, gamma_L]
    personal = [solve_common([g], [1.0]) for g in types]
    uniform = solve_common(types, [0.5, 0.5])
    mix_cache = {}

    def mixture(lam):
        if lam not in mix_cache:
            common = solve_common(types, [lam, 1 - lam])
            perfect = lam * personal[0].profit + (1 - lam) * personal[1].profit
            mix_cache[lam] = common, perfect, perfect - common.profit
        return mix_cache[lam]

    rows = []
    grid = np.unique(np.r_[np.linspace(0, 1, points), 0.5])
    for lam in grid:
        p, perfect, gain = mixture(float(lam))
        row = dict(
            high_share=lam,
            common_profit=p.profit,
            targeted_profit=perfect,
            targeting_gain=gain,
            common_profit_H=p.type_profits[0],
            common_profit_L=p.type_profits[1],
        )
        for j in range(J):
            row[f"p1_good{j + 1}"] = p.p1[j]
        rows.append(row)
    mix = pd.DataFrame(rows)
    mix.to_csv(output / "population_mix.csv", index=False)

    gamma_rows = []
    for g in np.unique(np.r_[np.linspace(0, gamma_max, points), types]):
        p = solve_common([g], [1.0])
        common_value = evaluate_policy(uniform.p1, uniform.p2, [g])[0]
        row = dict(
            gamma=g,
            targeted_profit=p.profit,
            uniform_50_50_profit=common_value,
            gain_for_type=p.profit - common_value,
        )
        for j in range(J):
            row[f"targeted_p1_good{j + 1}"] = p.p1[j]
        gamma_rows.append(row)
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(output / "profit_by_gamma.csv", index=False)
    policy_rows = []
    for label, p in [
        ("uniform_50_50", uniform),
        ("targeted_H", personal[0]),
        ("targeted_L", personal[1]),
    ]:
        for period, history, prices in [(1, "no prior product", p.p1)] + [
            (2, f"good{j + 1}", p.p2[j]) for j in range(J)
        ]:
            policy_rows.append(
                dict(
                    policy=label,
                    period=period,
                    history=history,
                    **{f"price_good{k + 1}": prices[k] for k in range(J)},
                )
            )
    pd.DataFrame(policy_rows).to_csv(output / "prices_50_50.csv", index=False)

    newborn, newborn_cutoff = newborn_discount_analysis(output)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed_menu_analysis(output, personal, uniform, gamma_df.gamma.to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(mix.high_share, mix.common_profit, label="Common across types")
    axes[0].plot(mix.high_share, mix.targeted_profit, label="Observed type")
    axes[0].set(
        xlabel="Population share of high type", ylabel="Profit per initial consumer"
    )
    axes[1].plot(mix.high_share, mix.targeting_gain, label="Gross gain from targeting")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Population share of high type", ylabel="Profit gain")
    axes[2].plot(
        gamma_df.gamma, gamma_df.targeted_profit, label="Prices optimized for gamma"
    )
    axes[2].plot(
        gamma_df.gamma, gamma_df.uniform_50_50_profit, label="Fixed 50/50 common policy"
    )
    axes[2].set(xlabel="Variety preference gamma", ylabel="Conditional expected profit")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "profits.png", dpi=180)
    fig.savefig(output / "profits.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(newborn.probability_L, newborn.regular_profit, label="Regular H menu")
    axes[0].plot(
        newborn.probability_L, newborn.discount_profit, label="Discount L menu"
    )
    axes[0].plot(
        newborn.probability_L,
        newborn.reoptimized_common_profit,
        label="Reoptimized common menu",
    )
    axes[0].set(xlabel="Probability newborn is low type", ylabel="Profit per newborn")
    axes[1].plot(
        newborn.probability_L, newborn.discount_gain, label="Gain from fixed discount"
    )
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(
        xlabel="Probability newborn is low type",
        ylabel="Discount profit minus regular profit",
    )
    if np.isfinite(newborn_cutoff):
        for ax in axes:
            ax.axvline(
                newborn_cutoff,
                color="gray",
                linestyle=":",
                label=f"Cutoff {newborn_cutoff:.3f}",
            )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "newborn_discount_cutoff.png", dpi=180)
    fig.savefig(output / "newborn_discount_cutoff.pdf")
    plt.close(fig)
    report = (
        "# Toy model numerical results\n\n"
        "Uniform means common across types; prices vary by product and period-one purchase history.\n"
        "All prices are committed in advance using fixed, known type masses; no learning or belief updating.\n"
        "Consumers are myopic. Period-one outside choices exit. Consumers have no prior purchase.\n"
        "Profits are per initial consumer, with N normalized to one.\n\n"
        f"50/50 common profit: {uniform.profit:.8f}\n\n"
        f"50/50 targeted profit: {mixture(0.5)[1]:.8f}\n\n"
        f"50/50 targeting gain: {mixture(0.5)[2]:.8f}\n\n"
        "The fixed terminal-discount decision for these same consumers "
        "is reported separately in newborn_results.md. No fixed adoption cost "
        "or noisy classifier is assumed.\n\n"
        "profit_by_gamma.csv reports conditional expected profits, not a probability density: "
        "a density over profits requires specifying a distribution over gamma. The population experiment "
        "uses two point masses, H and L.\n\n"
        "Optimization uses three starts and checks solver status; this is numerical evidence, "
        "not a proof of global optimality. See prices_50_50.csv for all prices.\n\n"
        "The three fixed H, L, and 50/50 menu transitions are reported in "
        "fixed_menu_results.md and fixed_menu_transitions.png.\n"
    )
    (output / "results.md").write_text(report)
    print(report)
    print(pd.DataFrame(policy_rows).to_string(index=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="updates/toy_model/results")
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--gamma-max", type=float, default=3.0)
    args = parser.parse_args()
    if args.points < 3 or args.gamma_max < 0:
        parser.error("Need points >= 3, gamma-max >= 0")
    run_analysis(args.output_dir, args.points, args.gamma_max)
