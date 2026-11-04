#!/usr/bin/env python3
"""Recreate supply_information for unknown gamma and observed initial history."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

if __package__:
    from .run_unknown_type import GAMMAS, solve, expit, scalar_maximum, supply_solution, plotting
else:
    from run_unknown_type import GAMMAS, solve, expit, scalar_maximum, supply_solution, plotting

# === Parameters: edit these values === #
output = Path(__file__).resolve().parent / "outputs"
probability_points = 41
beta = 2.0
alpha = -1.0
delta = 0.92
cost = 0.0


def committed_policy(h0, weights, beta, alpha, delta, cost, points=81):
    """Two common prices chosen before purchases; h0 is observed, gamma is not."""
    intercepts = beta+(-1 if h0 == "j" else 1)*GAMMAS
    scale = (20+abs(beta)+max(abs(GAMMAS))+abs(alpha*cost))/abs(alpha)

    def profile(second, details=False):
        sj = expit(beta-GAMMAS+alpha*second)
        s0 = expit(beta+GAMMAS+alpha*second)

        def objective(first):
            shares = expit(intercepts+alpha*np.asarray(first)[..., None])
            return ((np.asarray(first)-cost)*(shares @ weights)
                    + delta*(second-cost)*((shares*sj+(1-shares)*s0) @ weights))

        first, value = scalar_maximum(objective, (cost-2*scale, cost+2*scale), points)
        return (first, value) if details else value

    if delta:
        second, value = scalar_maximum(profile, (cost, cost+2*scale), points)
        first, value = profile(second, True)
        if min(second-cost, cost+2*scale-second) < 1e-6*scale:
            raise RuntimeError("Committed terminal price is near a search bound.")
    else:
        second = cost+1/abs(alpha)  # Unidentified by the objective at delta=0.
        first, value = profile(second, True)
    shares = expit(intercepts+alpha*first)
    sj = expit(beta-GAMMAS+alpha*second)
    s0 = expit(beta+GAMMAS+alpha*second)
    ds = alpha*shares*(1-shares)
    foc1 = float(weights @ (shares+ds*(first-cost+delta*(second-cost)*(sj-s0))))
    foc2 = float(weights @ (shares*(sj+alpha*(second-cost)*sj*(1-sj))
                           +(1-shares)*(s0+alpha*(second-cost)*s0*(1-s0)))) if delta else 0.
    if max(abs(foc1), abs(foc2)) > 1e-6:
        raise AssertionError("Committed policy FOC failed.")
    return first, second, value


def draw_figures(results, output):
    """Use the original figure's horizontal three-panel line-plot layout."""
    plt = plotting()
    for (focal_gamma, h0), data in results.groupby(["focal_gamma", "h0"], sort=False):
        state = "inside" if h0 == "j" else "outside"
        tag = {-1.25: "negative", 0.: "zero", 1.25: "positive"}[focal_gamma]
        stem = ("supply_information" if focal_gamma == 1.25 and h0 == "0"
                else f"supply_information_{tag}_h0_{state}")
        x = data.focal_probability
        with plt.rc_context({"font.size": 11, "axes.titlesize": 13}):
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
            for key, label in [("p1", "Unknown gamma: adaptive p1"),
                               ("committed_p1", "Unknown gamma: committed p1"),
                               ("committed_p2", "Unknown gamma: committed p2")]:
                axes[0].plot(x, data[key], label=label, linewidth=2)
            axes[0].set(title="Optimal prices", ylabel="Price")
            for key, label in [("known_gamma_profit", "Gamma revealed before pricing"),
                               ("optimal_profit", "Gamma unknown: observe h1"),
                               ("committed_profit", "Gamma unknown: commit to both prices")]:
                axes[1].plot(x, data[key], label=label, linewidth=2)
            axes[1].set(title="Expected two-period profit", ylabel="Discounted expected profit")
            axes[2].plot(x, data.value_observing_h1, linewidth=2,
                         label="Adaptive profit − committed profit")
            axes[2].set(title="Value of observing h1", ylabel="Expected profit gain")
            axes[2].set_ylim(bottom=0)
            for ax in axes:
                ax.set(xlabel=f"Pr(gamma = {focal_gamma:g})", xlim=(0, 1))
                ax.legend(fontsize=8, frameon=True)
            fig.suptitle(f"Known h0: {state} | Gamma unobserved in {{−1.25, 0, +1.25}} | Remaining probability split equally",
                         fontsize=13)
            for extension in ["png", "pdf"]:
                fig.savefig(output / f"{stem}.{extension}", dpi=180)
            plt.close(fig)


def run_information(settings):
    if settings.probability_points < 3:
        raise ValueError("Need at least three probability points.")
    grid = np.unique(np.r_[np.linspace(0, 1, settings.probability_points), 1/3, .5])
    rows = []
    for h0 in ["0", "j"]:
        # A benchmark only: reveal gamma before pricing, while retaining known h0.
        known_values = np.array([supply_solution(g, float(h0 == "j"), settings.beta,
                                 settings.alpha, settings.delta, settings.cost)["known_value"] for g in GAMMAS])
        for focal_index, gamma in enumerate(GAMMAS):
            for probability in grid:
                weights = np.full(3, (1-probability)/2)
                weights[focal_index] = probability
                _, policy, _ = solve(h0, weights, settings.beta, settings.alpha, settings.delta, settings.cost)
                first, second, committed_value = committed_policy(h0, weights, settings.beta,
                                                 settings.alpha, settings.delta, settings.cost)
                adaptive_value = policy[0]["optimal_profit"]
                known_value = float(weights @ known_values)
                if not known_value+1e-7 >= adaptive_value >= committed_value-1e-7:
                    raise AssertionError("Known gamma >= adaptive unknown gamma >= committed unknown gamma failed.")
                if any(np.isclose(probability, p, atol=1e-12) for p in [0., 1/3, .5, 1.]):
                    _, dense_policy, _ = solve(h0, weights, settings.beta, settings.alpha,
                                              settings.delta, settings.cost, points=161)
                    _, _, dense_committed = committed_policy(h0, weights, settings.beta, settings.alpha,
                                                            settings.delta, settings.cost, points=161)
                    assert np.isclose(dense_policy[0]["optimal_profit"], adaptive_value, atol=1e-7, rtol=0)
                    assert np.isclose(dense_committed, committed_value, atol=1e-7, rtol=0)
                row = dict(policy[0], focal_gamma=gamma, focal_probability=probability,
                           committed_p1=first, committed_p2=second, committed_profit=committed_value,
                           known_gamma_profit=known_value,
                           value_observing_h1=max(0., adaptive_value-committed_value))
                rows.append(row)
            print(f"Information curves: known h0={h0}, varying Pr(gamma={gamma:g})", flush=True)
    settings.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(settings.output / "supply_information_results.csv", index=False)
    draw_figures(results, settings.output)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(settings).items()},
                    information="h0 known in every case; gamma uncertain except in revealed-type benchmark",
                    varying="Probability of one gamma; other two types split remaining probability equally",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    solver_sha256=hashlib.sha256(Path(__file__).with_name("run_unknown_type.py").read_bytes()).hexdigest())
    (settings.output / "supply_information_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved six additional supply-information PNG/PDF figures to {settings.output}")


def read_command_line(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probability-points", type=int)
    for name in ["beta", "alpha", "delta", "cost"]:
        parser.add_argument(f"--{name}", type=float)
    parser.set_defaults(**vars(settings))
    return parser.parse_args()


if __name__ == "__main__":
    settings = SimpleNamespace(output=output, probability_points=probability_points,
                               beta=beta, alpha=alpha, delta=delta, cost=cost)
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_information(settings)
