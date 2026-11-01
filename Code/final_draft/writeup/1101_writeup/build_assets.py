"""Refresh committee-update tables and figures from saved results; no estimation."""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=CODE.parent / "shell_files/results_se")
    parser.add_argument("--hhi", type=Path, help="Detailed-flavor HHI PDF or saved household CSV.")
    args = parser.parse_args()
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    fit = json.loads((args.results / "fit.json").read_text())
    estimates = pd.read_csv(args.results / "estimates.csv")
    first = pd.read_csv(args.results / "first_stage.csv")
    boot = pd.read_csv(args.results / "bootstrap_draws.csv")
    assert fit["capacity"] == 0, "Update the demand equations before using an outside-option fit."
    assert fit["model"] == "heterogeneous"
    values = estimates.set_index("parameter")["estimate"]
    assert len(estimates) == 7
    names = {
        "beta_berry": r"$\beta_{\mathrm{berry}}$",
        "beta_plain": r"$\beta_{\mathrm{plain}}$",
        "mu_log_minus_alpha": r"$\mu_\alpha$",
        "mean_gamma": r"$\mu_\gamma$",
        "sd_log_minus_alpha": r"$\sigma_\alpha$",
        "sd_gamma": r"$\sigma_\gamma$",
        "price_control": r"$\rho$",
    }
    rows = []
    for row in estimates.itertuples():
        rows.append(f"{names[row.parameter]} & {row.estimate:.4f} & {row.std_error:.4f} & "
                    f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] " + r"\\")
    (assets / "estimates.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for row in first.itertuples():
        rows.append(f"{row.flavor.title()} & {row.iv_coefficient:.4f} & {row.iv_cluster_se:.4f} & "
                    f"{row.iv_cluster_F:,.2f} & {row.partial_R2:.4f} & "
                    f"{row.observations:,} & {row.households:,} " + r"\\")
    (assets / "first_stage.tex").write_text("\n".join(rows) + "\n")
    settings = fit["settings"]
    settings_rows = [
        ("Years", f"{settings['first_year']}--{settings['last_year']}"),
        ("Households", f"{fit['households']:,}"),
        ("Household--weeks", f"{fit['weeks']:,}"),
        ("Capacity setting", str(fit["capacity"])),
        (r"$\lambda$", str(settings["habit_weight"])),
        ("Integration draws", str(settings["draws"])),
        ("Main / bootstrap starts", f"{settings['starts']} / {settings['bootstrap_starts']}"),
        ("Bootstrap fits: usable / requested", f"{fit['inference']['usable']} / {fit['inference']['requested']}"),
        ("Converged", "Yes" if fit["converged"] else "No"),
        ("Parameters at bounds", str(int(estimates["at_bound"].sum()))),
        ("Negative log likelihood", f"{fit['nll']:,.4f}"),
    ]
    (assets / "settings.tex").write_text("\n".join(f"{k} & {v} " + r"\\" for k, v in settings_rows) + "\n")
    # Verify the source SEs and percentile intervals before displaying them.
    usable = boot.loc[boot["converged"]]
    for row in estimates.itertuples():
        np.testing.assert_allclose(usable[row.parameter].std(ddof=1), row.std_error)
        np.testing.assert_allclose(usable[row.parameter].quantile([.025, .975]), [row.ci_lower, row.ci_upper])
    for name in ["fit.json", "estimates.csv", "first_stage.csv", "bootstrap_draws.csv", "covariance.csv"]:
        shutil.copy2(args.results / name, assets / name)

    mean, sd = values["mean_gamma"], values["sd_gamma"]
    x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1001)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    density = norm.pdf(x, mean, sd)
    ax.plot(x, density, color="black", lw=1.6)
    ax.axvline(0, color="gray", ls="--", lw=.8)
    ax.set(xlabel=r"$\gamma_i$", ylabel="Density")
    fig.tight_layout()
    fig.savefig(assets / "gamma_distribution.pdf")
    plt.close(fig)

    data = CODE / "final_draft/data_code"
    hhi = args.hhi or next((p for p in [data / "flavor_hhi_histogram.pdf", data / "yogurt_household_flavors.csv"] if p.exists()), None)
    (assets / "hhi_figure.tex").write_text("")
    if hhi:
        target = assets / "flavor_hhi_histogram.pdf"
        if hhi.suffix.lower() == ".pdf":
            shutil.copy2(hhi, target)
        else:
            hh = pd.read_csv(hhi, usecols=["flavor_HHI"])["flavor_HHI"].dropna().to_numpy()
            if not len(hh) or not ((hh > 0) & (hh <= 1)).all():
                raise ValueError("HHI values must lie in (0, 1].")
            fig, ax = plt.subplots(figsize=(7, 3.7))
            ax.hist(hh, bins=np.linspace(0, 1, 41), weights=np.full(len(hh), 100 / len(hh)), color="steelblue")
            ax.set(xlim=(0, 1), xlabel="Household flavor HHI", ylabel="Households (%)")
            fig.tight_layout()
            fig.savefig(target)
            plt.close(fig)
        (assets / "hhi_figure.tex").write_text(
            r"\begin{figure}[htbp]\centering" + "\n" +
            r"\includegraphics[width=.83\linewidth]{assets/flavor_hhi_histogram.pdf}" + "\n" +
            r"\caption{Household flavor HHI; unaggregated flavor labels, package-quantity shares.}" + "\n" +
            r"\end{figure}" + "\n")
        print(f"Included detailed-flavor HHI from {hhi}")
    else:
        print("Detailed-flavor HHI input not present; no historical substitute included.")
    shares = data / "flavor_shares.pdf"
    (assets / "flavor_shares_figure.tex").write_text("")
    if shares.exists():
        shutil.copy2(shares, assets / "flavor_shares.pdf")
        (assets / "flavor_shares_figure.tex").write_text(
            r"\begin{figure}[htbp]\centering" + "\n" +
            r"\includegraphics[width=.95\linewidth]{assets/flavor_shares.pdf}" + "\n" +
            r"\caption{Package-quantity shares; 20 most purchased unaggregated flavor labels.}" + "\n" +
            r"\end{figure}" + "\n")

    source = CODE / "final_draft/supply_code/writeup_math_graphs/figures"
    solutions = pd.read_csv(source / "solutions.csv")
    # Check the displayed formulas in positive-price-sensitivity notation.
    s, u, v = solutions.s, solutions.u, solutions.v
    q = np.where(solutions.initial_state == "j", -1, 1)
    r = 1 - s
    alpha, delta = 1., .92
    np.testing.assert_allclose(solutions.p1, 1 / (alpha * r) - delta * solutions.DeltaV, atol=1e-10)
    np.testing.assert_allclose(solutions.dp1, (q * s + delta * r * (u + v)) / alpha, atol=1e-10)
    np.testing.assert_allclose(solutions.dprofit, (q * s - delta * s * u + delta * r * v) / alpha, atol=1e-10)
    np.testing.assert_allclose(solutions.d2p1,
        (s * r**2 * (q - delta * (u + v))**2 + delta * r * (-u * (1-u)**2 + v * (1-v)**2)) / alpha,
        atol=1e-10)
    shutil.copy2(source / "solutions.csv", assets / "supply_solutions.csv")
    groups = {state: solutions.loc[solutions.initial_state == state] for state in ["j", "0"]}
    specs = [
        ("matched_ratio_derivatives", "d_matched_ratio", r"$\mathrm{d}R_h/\mathrm{d}\gamma$"),
        ("profit", "profit", r"$\Pi^*(h)$"),
        ("profit_derivative", "dprofit", r"$\mathrm{d}\Pi^*(h)/\mathrm{d}\gamma$"),
        ("price_derivative", "dp1", r"$\mathrm{d}p_1^*(h)/\mathrm{d}\gamma$"),
    ]
    for filename, column, label in specs:
        fig, ax = plt.subplots(figsize=(7, 3.7))
        for state, linestyle in [("j", "-"), ("0", "--")]:
            frame = groups[state]
            ax.plot(frame.gamma, frame[column], linestyle, label=f"$h={state}$", lw=1.6)
        ax.axhline(0, color="gray", lw=.6)
        ax.set(xlabel=r"$\gamma$", ylabel=label)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(assets / f"{filename}.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    frame = groups["0"]
    ax.plot(frame.gamma, frame.d_induce_ratio, color="black", lw=1.6)
    ax.axhline(0, color="gray", lw=.6)
    ax.set(xlabel=r"$\gamma$", ylabel=r"$\mathrm{d}R_{0\to j}/\mathrm{d}\gamma$")
    fig.tight_layout()
    fig.savefig(assets / "path_ratio_derivative.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for state, ls in [("j", "-"), ("0", "--")]:
        axes[0].plot(groups[state].gamma, groups[state].p1, ls, label=f"$h={state}$", lw=1.6)
        axes[1].plot(groups["j"].gamma, groups["j"]["p2j" if state == "j" else "p20"],
                     ls, label=f"$a={state}$", lw=1.6)
    for ax, label in zip(axes, [r"$p_1^*(h)$", r"$p_2^*(a)$"]):
        ax.set(xlabel=r"$\gamma$", ylabel=label)
        ax.axhline(0, color="gray", lw=.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(assets / "prices.pdf")
    plt.close(fig)
    selected = solutions.loc[solutions.gamma.isin([-3, -1, 0, 1, 3])]
    rows = [f"${r.initial_state}$ & {r.gamma:.0f} & {r.p1:.3f} & {r.p2j:.3f} & {r.p20:.3f} & "
            f"{r.profit:.3f} & {r.dprofit:.3f} " + r"\\" for r in selected.itertuples()]
    (assets / "supply_values.tex").write_text("\n".join(rows) + "\n")
    for name in ["settings", "estimates", "first_stage", "supply_values"]:
        table = assets / f"{name}.tex"
        table.write_text(table.read_text() + "\\bottomrule\n")
    print("Refreshed tables, verified bootstrap SEs/intervals, and generated seven figures.")


if __name__ == "__main__":
    main()
