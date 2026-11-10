#!/usr/bin/env python3
"""Observed initial history, hidden fixed gamma, Bayesian learning from choices."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_supply_exercises import expit, scalar_maximum, supply_solution, plotting

GAMMAS = np.array([-1.25, 0., 1.25])
PRIORS = {
    "favor_negative": np.array([.5, .25, .25]),
    "favor_zero": np.array([.25, .5, .25]),
    "favor_positive": np.array([.25, .25, .5]),
    "equal": np.full(3, 1/3),
}
def solve(h0, weights, beta=2., alpha=-1., delta=1.0, cost=0., points=81):
    """One policy for a known h0 and a distribution over unobserved gamma."""
    weights = np.asarray(weights, dtype=float)
    if (h0 not in ["j", "0"] or weights.shape != (3,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)
            or not np.isfinite([beta, alpha, delta, cost]).all() or alpha >= 0
            or not 0 <= delta <= 1):
        raise ValueError("Need known h0, three normalized type probabilities, alpha<0, delta in [0,1].")
    initial_intercepts = beta + (-1 if h0 == "j" else 1)*GAMMAS
    scale = (20+abs(beta)+max(abs(GAMMAS))+abs(alpha*cost))/abs(alpha)

    def maximize(function, terminal=False):
        lower = cost if terminal else cost-2*scale
        upper = cost+2*scale
        price, value = scalar_maximum(function, (lower, upper), points)
        if min(price-lower, upper-price) < 1e-6*scale:
            raise RuntimeError("Optimum near numerical bound; enlarge search domain.")
        return price, value

    def terminal_policy(first):
        # The firm observes h1, not gamma. Bayes' rule weights each possible type.
        inside = expit(initial_intercepts+alpha*first)
        outside = expit(-initial_intercepts-alpha*first)
        policies = []
        for h1, likelihood in [("j", inside), ("0", outside)]:
            joint = weights*likelihood
            probability = float(joint.sum())
            posterior = joint/probability
            intercepts = beta+(-1 if h1 == "j" else 1)*GAMMAS

            def expected_terminal_profit(price):
                shares = expit(intercepts+alpha*np.asarray(price)[..., None])
                return (np.asarray(price)-cost)*(shares @ posterior)

            price, value = maximize(expected_terminal_profit, terminal=True)
            share = expit(intercepts+alpha*price)
            foc = float(posterior @ (share+alpha*(price-cost)*share*(1-share)))
            policies.append(dict(h1=h1, p2=price, value=value,
                                 probability=probability, posterior=posterior, foc=foc))
        return policies

    def expected_profit(first):
        # Optimize the expectation over gamma, rather than prices for each gamma.
        purchase_probability = float(weights @ expit(initial_intercepts+alpha*first))
        continuation = sum(p["probability"]*p["value"] for p in terminal_policy(first))
        return (first-cost)*purchase_probability+delta*continuation

    p1, optimal_profit = maximize(expected_profit)
    terminal = terminal_policy(p1)
    rows = []
    for i, gamma in enumerate(GAMMAS):
        s1 = float(expit(initial_intercepts[i]+alpha*p1))
        for branch in terminal:
            h1, p2 = branch["h1"], branch["p2"]
            probability = s1 if h1 == "j" else float(expit(-initial_intercepts[i]-alpha*p1))
            s2 = float(expit(beta+(-1 if h1 == "j" else 1)*gamma+alpha*p2))
            row = dict(case="unknown_type", h0=h0, gamma=gamma, h1=h1,
                       prior_type=weights[i], transition_probability=probability,
                       path_probability=weights[i]*probability, p1=p1, p2=p2,
                       second_purchase_probability=s2,
                       conditional_profit=(p1-cost)*(h1 == "j")+delta*(p2-cost)*s2)
            for k, name in enumerate(["negative", "zero", "positive"]):
                row[f"posterior_{name}"] = branch["posterior"][k]
            rows.append(row)

    probabilities = np.array([r["path_probability"] for r in rows])
    reconstructed = float(probabilities @ [r["conditional_profit"] for r in rows])
    assert np.isclose(probabilities.sum(), 1.)
    assert np.isclose(reconstructed, optimal_profit, atol=1e-8)
    assert np.allclose(sum(p["probability"]*p["posterior"] for p in terminal), weights)
    for branch in terminal:
        joint = [r["path_probability"] for r in rows if r["h1"] == branch["h1"]]
        assert np.allclose(branch["posterior"]*branch["probability"], joint, atol=1e-10)

    # Envelope derivative accounts for how p1 affects transitions and learning.
    shares = expit(initial_intercepts+alpha*p1)
    derivative = alpha*shares*(1-shares)
    future = [expit(beta+q*GAMMAS+alpha*p["p2"])*(p["p2"]-cost)
              for q, p in zip([-1, 1], terminal)]
    foc1 = float(weights @ (shares+derivative*(p1-cost+delta*(future[0]-future[1]))))
    residual = max(abs(foc1), *(abs(p["foc"]) for p in terminal))
    if residual > 1e-6:
        raise AssertionError(f"FOC residual {residual}")
    summary = dict(case="unknown_type", h0=h0, p1=p1,
                   p2_after_inside=terminal[0]["p2"], p2_after_outside=terminal[1]["p2"],
                   optimal_profit=optimal_profit, first_purchase_probability=float(weights @ shares),
                   prior_negative=weights[0], prior_zero=weights[1], prior_positive=weights[2])
    return rows, [summary], residual


def figures(paths, summary, output, delta=1.):
    """One readable page per prior: distribution, policy, and expected profit."""
    plt = plotting()
    colors = ["#3366a6", "#78909c", "#d48435"]
    titles = {"favor_negative": "Pr(gamma = −1.25) = 0.50",
              "favor_zero": "Pr(gamma = 0) = 0.50",
              "favor_positive": "Pr(gamma = +1.25) = 0.50",
              "equal": "Equal probability on all three gamma values"}
    for scenario, weights in PRIORS.items():
        data = summary[summary.scenario == scenario].set_index("h0").loc[["0", "j"]]
        with plt.rc_context({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13}):
            fig = plt.figure(figsize=(13, 7), layout="constrained")
            grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.7])
            prior_ax = fig.add_subplot(grid[:, 0])
            price_ax = fig.add_subplot(grid[0, 1])
            profit_ax = fig.add_subplot(grid[1, 1])
            bars = prior_ax.bar([r"$L$", r"$0$", r"$H$"], weights, color="#8267a0", width=.6)
            prior_ax.bar_label(bars, labels=[f"{p:.2f}" for p in weights], padding=6)
            prior_ax.set(xlabel=r"$\Pr(\gamma=\gamma_r)=\mu_r$", ylabel=r"$\Pr(\gamma=\gamma_r)=\mu_r$",
                         title="Uncertainty faced by the firm", ylim=(0, .65))
            x = np.arange(2)
            width = .23
            for offset, key, label, color in zip([-1, 0, 1],
                    ["p1", "p2_after_inside", "p2_after_outside"],
                    ["Period 1", "Period 2: bought in period 1", "Period 2: did not buy in period 1"], colors):
                bars = price_ax.bar(x+offset*width, data[key], width, color=color, label=label)
                price_ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
            price_ax.set(title="Optimal prices under gamma uncertainty", ylabel="Price")
            price_ax.set_ylim(min(0, float(data[["p1", "p2_after_inside", "p2_after_outside"]].min().min())*1.2),
                              float(data[["p1", "p2_after_inside", "p2_after_outside"]].max().max())*1.6)
            price_ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), fontsize=9, frameon=False)
            bars = profit_ax.bar(x, data.optimal_profit, color="#467d5d", width=.5)
            profit_ax.bar_label(bars, fmt="%.3f", padding=4)
            profit_ax.set(title="Maximized expected two-period profit", ylabel="Expected two-period profit",
                          ylim=(min(0, data.optimal_profit.min()*1.2), data.optimal_profit.max()*1.2))
            for ax in [price_ax, profit_ax]:
                ax.set_xticks(x, ["Known h0: outside", "Known h0: inside"])
                ax.spines[["top", "right"]].set_visible(False)
            prior_ax.spines[["top", "right"]].set_visible(False)
            dominant = ["L", "0", "H"][int(np.argmax(weights))]
            prior_title = r"$\mu_r=0.33$" if scenario == "equal" else rf"$\mu_{{{dominant}}}=0.5$"
            fig.suptitle(prior_title + f" | δ = {delta:g}", fontsize=18)
            for extension in ["png", "pdf"]:
                fig.savefig(output / f"optimal_policy_{scenario}.{extension}", dpi=180)
            plt.close(fig)


def self_test():
    for h0 in ["0", "j"]:
        for i, gamma in enumerate(GAMMAS):
            _, summary, _ = solve(h0, np.eye(3)[i])
            reference = supply_solution(gamma, float(h0 == "j"))
            actual = summary[0]
            assert np.isclose(actual["optimal_profit"], reference["known_value"], atol=1e-7)
            assert np.isclose(actual["p1"], reference["known_p1_inside" if h0 == "j" else "known_p1_outside"], atol=1e-6)
        for weights in PRIORS.values():
            rows, summary, _ = solve(h0, weights)
            # All true types face exactly the same p1 and same p2 at a given h1.
            assert np.ptp([r["p1"] for r in rows]) == 0
            for h1 in ["0", "j"]:
                assert np.ptp([r["p2"] for r in rows if r["h1"] == h1]) == 0
            # Revealing gamma can only increase the optimized expected profit.
            known = sum(w*supply_solution(g, float(h0 == "j"))["known_value"] for w, g in zip(weights, GAMMAS))
            assert known >= summary[0]["optimal_profit"]-1e-7
    print("Passed: common policy across gamma, degenerate-type benchmark, Bayesian beliefs, probability mass, profit reconstruction and FOCs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--beta", type=float, default=2.)
    parser.add_argument("--alpha", type=float, default=-1.)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--cost", type=float, default=0.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    paths, summaries, residuals = [], [], []
    for scenario, weights in PRIORS.items():
        for h0 in ["0", "j"]:
            rows, summary, residual = solve(h0, weights, args.beta, args.alpha, args.delta, args.cost)
            # Independently repeat with a denser price search.
            _, dense, _ = solve(h0, weights, args.beta, args.alpha, args.delta, args.cost, points=161)
            if not np.allclose([r["optimal_profit"] for r in summary], [r["optimal_profit"] for r in dense], atol=1e-7, rtol=0):
                raise AssertionError("Denser search disagrees.")
            for row in rows+summary:
                row["scenario"] = scenario
            paths.extend(rows)
            summaries.extend(summary)
            residuals.append(residual)
            print(f"Solved {scenario}, observed h0={h0}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    histories, expected = pd.DataFrame(paths), pd.DataFrame(summaries)
    histories.to_csv(args.output / "supply_history_outcomes.csv", index=False)
    expected.to_csv(args.output / "supply_expected_outcomes.csv", index=False)
    expected.to_csv(args.output / "optimal_policies.csv", index=False)
    figures(histories, expected, args.output, args.delta)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    information="h0 observed, gamma unobserved and fixed; h1 observed before p2",
                    gamma_values=GAMMAS.tolist(), priors={k: v.tolist() for k, v in PRIORS.items()},
                    maximum_foc_residual=max(residuals), denser_search_verified=True,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    solver_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / "run_supply_exercises.py").read_bytes()).hexdigest())
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved figures and tables to {args.output}")


if __name__ == "__main__":
    main()
