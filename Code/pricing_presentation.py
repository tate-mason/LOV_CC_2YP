"""Shared pricing labels and aggregate-only figures for current and legacy runners.

Calculations retain their internal policy keys. Presentation compares common
and personalized pricing; posted-price diagnostics are never a displayed series.
"""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import pandas as pd

POLICY_LABELS = {
    "optimized_common_coupons": "Common pricing",
    "common_pricing": "Common pricing",
    "personalized_dynamic": "Personalized pricing",
    "personalized_pricing": "Personalized pricing",
    "known": "Personalized pricing",
    "unknown": "Common pricing",
}
GROUPS = ["low", "middle", "high"]
COLORS = {"Common pricing": "#3575AD", "Personalized pricing": "#D17925"}


def prior_labels(scenarios):
    """Name only the dominant type probability; equal priors use mu_r."""
    labels = []
    for scenario in scenarios:
        name = str(scenario).lower()
        if "equal" in name:
            label = r"$\mu_r=0.33$"
        elif any(term in name for term in ["low", "habit", "negative"]):
            label = r"$\mu_L=0.5$"
        elif any(term in name for term in ["medium", "middle", "neutral", "zero"]):
            label = r"$\mu_0=0.5$"
        elif any(term in name for term in ["high", "novelty", "positive"]):
            label = r"$\mu_H=0.5$"
        else:
            raise ValueError(f"Unrecognized type prior: {scenario}")
        labels.append(label)
    return labels


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pricing_presentation_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def displayed_policies(table):
    """A copy suitable for reporting; never mutate a solver's internal table."""
    result = table.loc[table.policy != "no_coupon"].copy()
    result["policy"] = result.policy.replace(POLICY_LABELS)
    return result


def group_results(table, common_groups=None):
    personal = table.loc[table.policy.isin(["personalized_dynamic", "Personalized pricing"])]
    sums = [c for c in ["revenue1", "revenue2", "discounted_revenue", "coupon_expenditure"] if c in personal]
    means = [c for c in ["p1", "expected_p2", "discount1", "expected_discount2"] if c in personal]
    grouped = personal.groupby("group")[sums].sum().join(personal.groupby("group")[means].mean())
    grouped["eligible_households"] = personal.groupby("group").size()
    grouped["policy"] = "Personalized pricing"
    result = grouped.reset_index()
    if common_groups is not None and len(common_groups):
        result = pd.concat([pd.DataFrame(common_groups), result], ignore_index=True)
    return result


def save_figure(fig, output, stems):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"{stem}.{extension}", dpi=180)


def price_paths(groups, output, flavor="berry", stems=("pricing_mechanism",)):
    """Only p1 -> E[p2], with one line per pricing policy in each type panel."""
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, group in zip(axes, GROUPS):
        selected = groups.loc[groups.group == group]
        for row in selected.itertuples():
            ax.plot([1, 2], [row.p1, row.expected_p2], marker="o", linewidth=2,
                    color=COLORS[row.policy], label=row.policy)
        ax.set(xticks=[1, 2], xticklabels=[r"$p_1$", r"$p_2$"], title=group.title(), xlim=(.9, 2.1))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Price ($)")
    fig.suptitle(f"{flavor.title()}: price paths")
    save_figure(fig, output, stems)
    plt.close(fig)


def benchmark_figure(benchmarks, output, flavor):
    plt = plotting()
    shown = displayed_policies(benchmarks)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(shown.policy, shown.discounted_revenue, color=[COLORS[p] for p in shown.policy])
    ax.set(title=f"{flavor.title()}: pricing comparison", ylabel="Expected two-period revenue ($)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    save_figure(fig, output, ["pricing_policy_benchmarks"])
    plt.close(fig)


def revenue_gain_figure(aggregate, output):
    """Show the proportional lift over common pricing for each horizon."""
    from matplotlib.ticker import PercentFormatter

    metrics = ["revenue1", "revenue2", "discounted_revenue"]
    policies = displayed_policies(aggregate).set_index("policy")
    common = policies.loc["Common pricing", metrics].to_numpy(dtype=float)
    personalized = policies.loc["Personalized pricing", metrics].to_numpy(dtype=float)
    if np.any(common <= 0) or not np.isfinite([common, personalized]).all():
        raise ValueError("Revenue gain requires finite revenues and a positive common-pricing baseline.")
    gains = (personalized / common - 1) * 100

    plt = plotting()
    fig, ax = plt.subplots(figsize=(8.4, 3.5), layout="constrained")
    bars = ax.barh(["Period 1", "Period 2", "Both periods"], gains, height=.54,
                   color=["#667E90", "#667E90", COLORS["Personalized pricing"]])
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{value:+.2f}%" for value in gains],
                 padding=7, fontsize=12, fontweight="semibold")
    ax.axvline(0, color="#4A5560", linewidth=.9)
    ax.set_xlabel("Revenue gain over common pricing", fontsize=12, labelpad=9)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    span = max(gains.max(), 0) - min(gains.min(), 0)
    padding = max(.18 * span, .5)
    ax.set_xlim(min(gains.min() - padding, 0), max(gains.max() + padding, 0))
    ax.tick_params(axis="both", length=0, labelsize=12, pad=7)
    ax.grid(axis="x", color="#DCE2E7", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    save_figure(fig, output, ["pricing_revenue_gain"])
    plt.close(fig)


def information_price_paths_comparison(type_results, information, output, flavor):
    """Compare known-type offers with common offers under the equal-type prior."""
    fields = ["known_p1", "known_expected_p2"]
    # Known-type offers do not depend on the prior. Repeated prior rows must agree.
    paths = type_results[["type", "h0", *fields]].drop_duplicates()
    paths = paths.set_index(["type", "h0"], verify_integrity=True)
    common = information.loc[(information["scenario"] == "Equal types") &
                             (information["case"] == "unknown")].set_index("h0", verify_integrity=True)
    types = [("low", "#2E6896", "o"), ("medium", "#208475", "s"), ("high", "#B35C21", "^")]
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
    for ax, (state, title) in zip(axes, [("j", f"Initially inside ({flavor})"), ("0", "Initially outside")]):
        for switching_type, color, marker in types:
            prices = paths.loc[(switching_type, state), fields].to_numpy(dtype=float)
            if not np.isfinite(prices).all():
                raise ValueError("Complete-information prices must be finite.")
            ax.plot([1, 2], prices, label=f"{switching_type.title()} type", color=color,
                    marker=marker, linewidth=1.9, markersize=5)
        common_prices = common.loc[state, ["expected_p1", "expected_p2"]].to_numpy(dtype=float)
        if not np.isfinite(common_prices).all():
            raise ValueError("Common-pricing offers must be finite.")
        ax.plot([1, 2], common_prices, label="Common price", color="#232C35", marker="D",
                markerfacecolor="white", linestyle="--", linewidth=2.1, markersize=6)
        ax.set_title(title, fontsize=13, pad=11)
        ax.set_xticks([1, 2], [r"Initial offer $p_1$", r"Expected offer $\mathbb{E}[p_2]$"])
        ax.set_xlim(.75, 2.25)
        ax.margins(y=.13)
        ax.tick_params(axis="both", length=0, labelsize=11, pad=8)
        ax.grid(axis="y", color="#DCE2E7", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B8C2CB")
    axes[0].set_ylabel("Price ($)", fontsize=12, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11,
               title="Solid: complete information   |   Dashed: type unobserved")
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.75, wspace=.13)
    save_figure(fig, output, ["information_price_paths_comparison"])
    plt.close(fig)


def pricing_comparison_figures(pricing_folder, output, flavor):
    """Render slide figures from the same aggregate tables available in RESULTS_ONLY."""
    aggregate = pd.read_csv(pricing_folder / "targeted_revenue_aggregate.csv")
    revenue_gain_figure(aggregate, output)
    status = json.loads((pricing_folder / "information_calibration_status.json").read_text())
    if status["status"] == "complete":
        types = pd.read_csv(pricing_folder / "information_friction_by_group.csv", dtype={"h0": str})
        information = pd.read_csv(pricing_folder / "information_friction_aggregate.csv", dtype={"h0": str})
        information_price_paths_comparison(types, information, output, flavor)


def empirical_figures(table, output, flavor="berry", delta=1., common_groups=None):
    groups = group_results(table, common_groups)
    price_paths(groups, output, flavor,
                ["pricing_mechanism", "discounts_by_consumer_type", "terminal_offers_by_purchase"])
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    for policy, segment in groups.groupby("policy", sort=False):
        weight = segment.eligible_households
        p1 = np.average(segment.p1, weights=weight)
        p2 = np.average(segment.expected_p2, weights=weight)
        axes[0].plot([1, 2], [p1, p2], marker="o", linewidth=2, color=COLORS[policy], label=policy)
        values = segment.set_index("group").reindex(GROUPS)
        axes[1].plot(range(3), values.discounted_revenue / values.eligible_households,
                     marker="o", linewidth=2, color=COLORS[policy], label=policy)
    axes[0].set(title="Price paths", ylabel="Price ($)", xticks=[1, 2], xticklabels=[r"$p_1$", r"$p_2$"])
    axes[1].set(title="Expected two-period revenue", ylabel="Mean revenue per household ($)")
    wide = groups.pivot(index="group", columns="policy", values="discounted_revenue").reindex(GROUPS)
    if "Common pricing" in wide:
        counts = groups.loc[groups.policy == "Personalized pricing"].set_index("group").eligible_households.reindex(GROUPS)
        axes[2].plot(range(3), (wide["Personalized pricing"] - wide["Common pricing"]) / counts,
                     marker="o", linewidth=2, label="Personalized − common")
        axes[2].set(title="Revenue gain", ylabel="Mean gain per household ($)")
    else:
        # Legacy solvers have no optimized common policy. Do not relabel posted prices.
        personal = groups.set_index("group").reindex(GROUPS)
        for column, label in [("revenue1", r"$p_1$"), ("revenue2", r"$p_2$")]:
            axes[2].plot(range(3), personal[column] / personal.eligible_households,
                         marker="o", linewidth=2, label=label)
        axes[2].set(title="Revenue by period", ylabel="Mean revenue per household ($)")
    for ax in axes[1:]:
        ax.set(xticks=range(3), xticklabels=[g.title() for g in GROUPS], xlabel="Switching type group")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{flavor.title()} vs outside | δ = {delta:g}")
    save_figure(fig, output, ["pricing_theory_comparison", "household_coupon_results"])
    plt.close(fig)
    return groups


def information_figures(table, output, delta=1., filename_prefix="information_friction"):
    """The same two policy names and p1 -> p2 paths for type-information results."""
    plt = plotting()
    scenarios = list(dict.fromkeys(table.scenario))
    probability_labels = prior_labels(scenarios)
    probability_axis = r"$\Pr(\gamma=\gamma_r)=\mu_r$"
    for state, state_name in [("j", "flavor"), ("0", "outside")]:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
        selected = table.loc[table.h0 == state]
        for case in ["unknown", "known"]:
            rows = selected.loc[selected.case == case].set_index("scenario").reindex(scenarios)
            label = POLICY_LABELS[case]
            # Priors are separate experiments; never average prices across priors.
            for index, (_, values) in enumerate(rows.iterrows()):
                axes[0].plot([1, 2], [values.expected_p1, values.expected_p2],
                             marker="o", linewidth=1.5, color=COLORS[label],
                             linestyle=["-", "--", "-.", ":"][index % 4],
                             label=label + ": " + ["Equal", "Low-heavy", "Middle-heavy", "High-heavy"][index])
            axes[1].plot(range(len(rows)), rows.persistent_type_revenue,
                         marker="o", linewidth=2, color=COLORS[label], label=label)
        gains = selected.loc[selected.case == "known"].set_index("scenario").reindex(scenarios)
        axes[2].plot(range(len(gains)), gains.value_of_information, marker="o", linewidth=2,
                     label="Personalized − common")
        axes[0].set(title="Price paths", ylabel="Price ($)", xticks=[1, 2], xticklabels=[r"$p_1$", r"$p_2$"])
        axes[1].set(title="Expected two-period revenue", ylabel="Revenue per model consumer ($)")
        axes[2].set(title="Revenue gain", ylabel="Expected gain ($)")
        for ax in axes[1:]:
            ax.set(xticks=range(len(scenarios)), xticklabels=probability_labels, xlabel=probability_axis)
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Initially {'inside' if state == 'j' else 'outside'}" + (f" | δ = {delta:g}" if delta is not None else ""))
        save_figure(fig, output, [f"information_theory_{state_name}"])
        plt.close(fig)
    # Keep existing filenames usable by slide sources and export allowlists.
    for stem, metric, ylabel in [("profits", "persistent_type_revenue", "Expected two-period revenue ($)"),
                                 ("information_value", "value_of_information", "Personalized − common ($)")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
        for ax, state, state_name in zip(axes, ["j", "0"], ["flavor", "outside"]):
            for case in (["known"] if stem == "information_value" else ["unknown", "known"]):
                values = table.loc[(table.h0 == state) & (table.case == case)].set_index("scenario").reindex(scenarios)
                label = "Personalized − common" if stem == "information_value" else POLICY_LABELS[case]
                ax.bar(np.arange(len(scenarios)) + (0 if stem == "information_value" else (-.18 if case == "unknown" else .18)),
                       values[metric], width=.6 if stem == "information_value" else .36, label=label)
            ax.set(xticks=range(len(scenarios)), xticklabels=probability_labels, xlabel=probability_axis, title="Initially " + ("inside" if state == "j" else "outside"))
            ax.legend(fontsize=8)
        axes[0].set_ylabel(ylabel)
        save_figure(fig, output, [filename_prefix + "_" + stem])
        plt.close(fig)
    # Each prior gets its own p1 -> p2 path; no conditional terminal-price series.
    fig, axes = plt.subplots(2, len(scenarios), figsize=(14, 6), sharey=True, squeeze=False, layout="constrained")
    for row, (state, state_name) in enumerate([("j", "flavor"), ("0", "outside")]):
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            for values in table.loc[(table.h0 == state) & (table.scenario == scenario)].itertuples():
                label = POLICY_LABELS[values.case]
                ax.plot([1, 2], [values.expected_p1, values.expected_p2], marker="o", linewidth=2, color=COLORS[label], label=label)
            ax.set(xticks=[1, 2], xticklabels=[r"$p_1$", r"$p_2$"], title=("Inside" if state == "j" else "Outside") + "\n" + probability_labels[col].replace("\n", " "))
            ax.set_title(ax.get_title(), fontsize=9)
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Price ($)")
    axes[1, 0].set_ylabel("Price ($)")
    save_figure(fig, output, [filename_prefix + "_first_prices", filename_prefix + "_terminal_prices", filename_prefix + "_price_paths"])
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render slide figures from exported aggregate pricing tables.")
    parser.add_argument("--pricing-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flavor", required=True)
    args = parser.parse_args()
    pricing_comparison_figures(args.pricing_folder, args.output, args.flavor)
