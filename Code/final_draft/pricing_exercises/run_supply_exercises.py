#!/usr/bin/env python3
"""Run all three theoretical supply cases, prices, ratios and profit, in one file.

No empirical input is read. Dependencies: numpy, pandas, matplotlib.
python3 run_supply_exercises.py [--self-test] [--output DIRECTORY]
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

def golden_maximum(function, lo, hi):
    ratio = (math.sqrt(5)-1)/2
    left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fl, fr = function(left), function(right)
    for _ in range(100):
        if hi-lo <= 1e-10*(1+abs(left)+abs(right)):
            break
        if fl > fr:
            hi, right, fr = right, left, fl
            left = hi-ratio*(hi-lo)
            fl = function(left)
        else:
            lo, left, fl = left, right, fr
            right = lo+ratio*(hi-lo)
            fr = function(right)
    return (left, fl) if fl > fr else (right, fr)

def scalar_maximum(function, bounds, points=161):
    """Search every grid-local maximum, including boundaries (mixtures may multimode)."""
    grid = np.linspace(*bounds, points)
    # Supply objectives support array evaluation; profiles are scalar functions.
    try:
        values = np.asarray(function(grid))
        if values.shape != grid.shape:
            raise ValueError("Scalar objective")
    except (ValueError, TypeError):
        values = np.array([function(x) for x in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, points - 1):
        if values[i] >= max(values[i - 1], values[i + 1]) and values[i] > min(values[i - 1], values[i + 1]):
            price, value = golden_maximum(function, grid[i - 1], grid[i + 1])
            candidates.append((value, price))
    value, price = max(candidates)
    return float(price), float(value)

def static_price(intercept, alpha, effective_cost=0.):
    """Signed alpha<0; monotone transformed FOC, including introductory subsidies."""
    def foc(p):
        return alpha*(p-effective_cost)+1+np.exp(np.clip(intercept+alpha*p, -745, 700))
    lo, hi = effective_cost, effective_cost+max(2., 2/abs(alpha))
    while foc(hi) > 0:
        hi = lo+2*(hi-lo)
    for _ in range(100):
        mid = (lo+hi)/2
        if foc(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def supply_solution(gamma, prior=.5, beta=2., alpha=-1., delta=.92, cost=0.):
    """Known type; hidden h0, then either observe h1 or commit to p1,p2."""
    if not (np.isfinite([gamma, prior, beta, alpha, delta, cost]).all()
            and alpha < 0 and 0 <= prior <= 1 and 0 <= delta <= 1):
        raise ValueError("Need finite inputs, alpha<0, prior/delta in [0,1].")
    # Index 0 is previously inside j; index 1 is previously outside 0.
    intercepts = beta + np.array([-gamma, gamma])
    weights = np.array([prior, 1 - prior])
    p2 = np.array([static_price(x, alpha, cost) for x in intercepts])
    s2 = expit(intercepts + alpha * p2)
    W2 = (p2 - cost) * s2
    adjustment = delta * (W2[0] - W2[1])

    def share1(p):
        return expit(intercepts + alpha * np.asarray(p)[..., None]) @ weights

    def adaptive_value(p):
        s = share1(p)
        return s * (p - cost) + delta * (s * W2[0] + (1 - s) * W2[1])

    scale = (20 + abs(beta) + abs(gamma) + abs(alpha * cost)) / abs(alpha)
    bounds = (cost - scale, cost + scale)
    pooled_p1, adaptive = scalar_maximum(adaptive_value, bounds)
    # Rechecking a wider domain catches a binding numerical search boundary.
    wide_p1, wide_value = scalar_maximum(adaptive_value, (cost - 2*scale, cost + 2*scale), 321)
    if wide_value > adaptive + 1e-8:
        pooled_p1, adaptive = wide_p1, wide_value
    known_p1 = np.array([static_price(x, alpha, cost - adjustment) for x in intercepts])
    known_values = []
    for x, price in zip(intercepts, known_p1):
        s = expit(x + alpha * price)
        known_values.append(s*(price-cost) + delta*(s*W2[0]+(1-s)*W2[1]))

    def committed_profile(second, return_price=False):
        terminal = expit(intercepts + alpha * second)
        Wj, W0 = terminal*(second-cost)
        def objective(first):
            s = share1(first)
            return s*(first-cost)+delta*(s*Wj+(1-s)*W0)
        price, value = scalar_maximum(objective, (cost-2*scale, cost+2*scale))
        return (price, value) if return_price else value

    # Profile the joint problem, optimizing p1 anew at EACH candidate p2.
    # Refining every grid-local mode avoids assuming a globally unique FOC.
    if delta == 0:
        cp1, cp2, committed = pooled_p1, static_price(beta, alpha, cost), adaptive
    else:
        cp2, committed = scalar_maximum(committed_profile, (cost, cost+scale))
        cp1, committed = committed_profile(cp2, True)
        # Independent wider and denser profile scan.
        check_p2, check_value = scalar_maximum(committed_profile, (cost, cost+2*scale), 321)
        if check_value > committed+1e-9:
            cp2 = check_p2
            cp1, committed = committed_profile(cp2, True)
        if min(cp2-cost, cost+2*scale-cp2) < scale*1e-5:
            raise RuntimeError("Committed optimum is near a search bound; inspect calibration.")
    known = float(weights @ known_values)
    if not known + 1e-7 >= adaptive >= committed - 1e-7:
        raise AssertionError("Information-value ordering failed.")
    s = share1(pooled_p1)
    derivative = float(weights @ (alpha*expit(intercepts+alpha*pooled_p1)
                                  * expit(-intercepts-alpha*pooled_p1)))
    committed_first_share = share1(cp1)
    committed_first_derivative = float(weights @ (alpha*expit(intercepts+alpha*cp1)*expit(-intercepts-alpha*cp1)))
    terminal = expit(intercepts+alpha*cp2)
    terminal_weights = np.array([committed_first_share, 1-committed_first_share])
    committed_foc1 = committed_first_share+committed_first_derivative*(cp1-cost+delta*(cp2-cost)*(terminal[0]-terminal[1]))
    committed_foc2 = float(terminal_weights @ (terminal+alpha*(cp2-cost)*terminal*(1-terminal))) if delta > 0 else 0.
    return dict(gamma=gamma, prior_inside=prior, beta=beta, alpha=alpha, delta=delta, cost=cost,
                known_p1_inside=known_p1[0], known_p1_outside=known_p1[1],
                known_value=known, hidden_p1=pooled_p1, adaptive_p2_inside=p2[0],
                adaptive_p2_outside=p2[1], hidden_value=adaptive,
                committed_p1=cp1, committed_p2=cp2, committed_value=committed,
                initial_information_value=known-adaptive,
                second_period_flexibility_value=adaptive-committed,
                committed_foc1=committed_foc1, committed_foc2=committed_foc2,
                hidden_foc=s+derivative*(pooled_p1-cost+adjustment))

def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_exercises_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


CASES = ["known_state", "hidden_initial_state", "ex_ante_commitment"]
LABELS = dict(zip(CASES, ["Known state (base)", "Hidden initial state", "Ex ante commitment"]))


def outcome_tables(result):
    """All four realized histories per case, plus correctly weighted ex ante objects."""
    g, beta, alpha, delta, cost, pi = [result[k] for k in ["gamma", "beta", "alpha", "delta", "cost", "prior_inside"]]
    paths, summary = [], []
    for case in CASES:
        case_paths = []
        for h0, weight in [("j", pi), ("0", 1-pi)]:
            if case == "known_state":
                p1 = result["known_p1_inside" if h0 == "j" else "known_p1_outside"]
            else:
                p1 = result["hidden_p1" if case == "hidden_initial_state" else "committed_p1"]
            s1 = float(expit(beta+(-1 if h0 == "j" else 1)*g+alpha*p1))
            conditional_value = 0.
            local = []
            for h1, transition in [("j", s1), ("0", 1-s1)]:
                p2 = result["committed_p2"] if case == "ex_ante_commitment" else result["adaptive_p2_inside" if h1 == "j" else "adaptive_p2_outside"]
                s2 = float(expit(beta+(-1 if h1 == "j" else 1)*g+alpha*p2))
                # Realize the first choice; integrate the terminal taste shock.
                conditional_profit = (p1-cost)*(h1 == "j")+delta*(p2-cost)*s2
                conditional_value += transition*conditional_profit
                row = dict(gamma=g, prior_inside=pi, case=case, h0=h0, h1=h1,
                           initial_probability=weight, transition_probability=transition,
                           path_probability=weight*transition, p1=p1, p2=p2,
                           price_ratio=p1/p2 if abs(p2) > 1e-12 else np.nan,
                           first_purchase_probability=s1, second_purchase_probability=s2,
                           conditional_profit=conditional_profit)
                local.append(row)
            for row in local:
                row["optimal_profit_given_h0"] = conditional_value
            case_paths.extend(local)
        probability = np.array([r["path_probability"] for r in case_paths])
        if not np.isclose(probability.sum(), 1.):
            raise AssertionError("Path probabilities do not sum to one.")
        mean_p1, mean_p2, profit = [float(probability @ np.array([r[k] for r in case_paths]))
                                    for k in ["p1", "p2", "conditional_profit"]]
        expected_ratio = float(probability @ np.array([r["price_ratio"] for r in case_paths]))
        expected_key = {"known_state": "known_value", "hidden_initial_state": "hidden_value", "ex_ante_commitment": "committed_value"}[case]
        if not np.isclose(profit, result[expected_key], atol=1e-8):
            raise AssertionError("Path profits do not reproduce optimized objective.")
        summary.append(dict(gamma=g, prior_inside=pi, case=case, expected_p1=mean_p1,
                            expected_p2=mean_p2, ratio_of_expected_prices=mean_p1/mean_p2 if abs(mean_p2) > 1e-12 else np.nan,
                            expected_price_ratio=expected_ratio, optimal_profit=profit))
        paths.extend(case_paths)
    return paths, summary


def make_figures(paths, summary, output):
    plt = plotting()
    for prior in summary.prior_inside.unique():
        stem = f"prior_{prior:g}".replace(".", "p")
        subset = paths[paths.prior_inside == prior]
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        for column, case in enumerate(CASES):
            data = subset[subset.case == case]
            if case == "known_state":
                for h0 in ["0", "j"]:
                    initial = data[(data.h0 == h0) & (data.h1 == "j")]
                    axes[0, column].plot(initial.gamma, initial.p1, label=f"p1(h0={h0})")
            else:
                initial = data[(data.h0 == "0") & (data.h1 == "j")]
                axes[0, column].plot(initial.gamma, initial.p1, label="p1")
            if case == "ex_ante_commitment":
                axes[0, column].plot(initial.gamma, initial.p2, label="p2")
            else:
                for h1 in ["0", "j"]:
                    terminal = data[(data.h0 == "0") & (data.h1 == h1)]
                    axes[0, column].plot(terminal.gamma, terminal.p2, "--", label=f"p2(h1={h1})")
            for (h0, h1), history in data.groupby(["h0", "h1"]):
                # Duplicate ratios in restricted cases need only one curve.
                if case != "known_state" and h0 != "0":
                    continue
                if case == "ex_ante_commitment" and h1 != "j":
                    continue
                axes[1, column].plot(history.gamma, history.price_ratio, label=f"h0={h0}, h1={h1}")
            for h0 in ["0", "j"]:
                initial = data[(data.h0 == h0) & (data.h1 == "j")]
                axes[2, column].plot(initial.gamma, initial.optimal_profit_given_h0, label=f"E[profit|h0={h0}]")
            expected = summary[(summary.prior_inside == prior) & (summary.case == case)]
            axes[2, column].plot(expected.gamma, expected.optimal_profit, "k--", label="Ex ante E[profit]")
            axes[0, column].set_title(LABELS[case])
        for row, ylabel in enumerate(["Optimal prices", "Path-specific p1 / p2", "Optimal discounted profit"]):
            for ax in axes[row]:
                ax.set(xlabel="gamma", ylabel=ylabel)
                ax.axhline(0, color="gray", linewidth=.5)
                ax.legend(fontsize=7)
        fig.suptitle(f"Supply cases: Pr(h0=j)={prior:g}")
        fig.tight_layout()
        fig.savefig(output / f"supply_cases_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_cases_{stem}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for case, group in summary[summary.prior_inside == prior].groupby("case", sort=False):
            axes[0].plot(group.gamma, group.optimal_profit, label=LABELS[case])
            axes[1].plot(group.gamma, group.ratio_of_expected_prices, label=LABELS[case])
            axes[2].plot(group.gamma, group.expected_price_ratio, label=LABELS[case])
        for ax, title in zip(axes, ["Optimal expected profit", "E[p1] / E[p2]", "E[p1 / p2]"]):
            ax.set(xlabel="gamma", title=title)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"supply_comparison_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_comparison_{stem}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, case in zip(axes, CASES):
            data = subset[subset.case == case]
            for (h0, h1), history in data.groupby(["h0", "h1"]):
                ax.plot(history.gamma, history.conditional_profit, label=f"h0={h0}, h1={h1}")
            ax.set(xlabel="gamma", ylabel="Path-conditional discounted profit", title=LABELS[case])
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"supply_history_profit_{stem}.png", dpi=180)
        fig.savefig(output / f"supply_history_profit_{stem}.pdf")
        plt.close(fig)


def self_test():
    for gamma, prior, delta in [(0., .5, .92), (-2., .25, .92), (1.5, .75, .92),
                                 (-1., 0., .92), (1., 1., .92), (1., .5, 0.)]:
        solution = supply_solution(gamma, prior, delta=delta)
        paths, summary = outcome_tables(solution)
        assert len(paths) == 12 and len(summary) == 3
        for key in ["hidden_foc", "committed_foc1", "committed_foc2"]:
            assert abs(solution[key]) < 1e-6
        if gamma == 0:
            assert np.ptp([row["optimal_profit"] for row in summary]) < 1e-7
            assert all(np.isclose(row["price_ratio"], 1., atol=1e-6) for row in paths)
        if prior in (0., 1.):
            assert np.isclose(solution["known_value"], solution["hidden_value"], atol=1e-7)
    print("Supply checks passed: all cases, four histories, probabilities, profit reconstruction, FOCs, degenerate priors, gamma=0, delta=0.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "supply_outputs")
    parser.add_argument("--gamma-min", type=float, default=-3.)
    parser.add_argument("--gamma-max", type=float, default=3.)
    parser.add_argument("--gamma-points", type=int, default=61)
    parser.add_argument("--priors", type=float, nargs="+", default=[.25, .5, .75])
    parser.add_argument("--beta", type=float, default=2.)
    parser.add_argument("--alpha", type=float, default=-1., help="Signed price coefficient, negative.")
    parser.add_argument("--delta", type=float, default=.92)
    parser.add_argument("--cost", type=float, default=0.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.gamma_points < 2 or args.gamma_max <= args.gamma_min:
        parser.error("Need >=2 gamma points and gamma-max > gamma-min.")
    args.output.mkdir(parents=True, exist_ok=True)
    solutions, paths, summaries = [], [], []
    for prior in args.priors:
        for i, gamma in enumerate(np.linspace(args.gamma_min, args.gamma_max, args.gamma_points)):
            result = supply_solution(gamma, prior, args.beta, args.alpha, args.delta, args.cost)
            history_rows, summary_rows = outcome_tables(result)
            solutions.append(result)
            paths.extend(history_rows)
            summaries.extend(summary_rows)
            if i % 10 == 0 or i == args.gamma_points-1:
                print(f"Prior {prior:g}: solved {i+1}/{args.gamma_points} gamma values", flush=True)
    raw, histories, expected = pd.DataFrame(solutions), pd.DataFrame(paths), pd.DataFrame(summaries)
    raw.to_csv(args.output / "supply_solutions.csv", index=False)
    histories.to_csv(args.output / "supply_history_outcomes.csv", index=False)
    expected.to_csv(args.output / "supply_expected_outcomes.csv", index=False)
    residual = float(raw[["hidden_foc", "committed_foc1", "committed_foc2"]].abs().to_numpy().max())
    if residual > 1e-6:
        raise AssertionError(f"FOC residual too large: {residual}")
    make_figures(histories, expected, args.output)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    maximum_foc_residual=residual, calibrations=len(raw), cases=CASES,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"All three supply cases completed. CSVs and PNG/PDF figures: {args.output}")


if __name__ == "__main__":
    main()
