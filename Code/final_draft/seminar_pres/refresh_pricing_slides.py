"""Restyle recovered aggregate pricing results; never read empirical records.

This is a presentation refresh, not a re-estimation or a delta=1 pricing rerun.
Only the three named aggregate CSVs in each flavor directory are read.
"""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Code/pricing_presentation.py").is_file())
sys.path.insert(0, str(ROOT / "Code"))
from pricing_presentation import (POLICY_LABELS, displayed_policies, price_paths,
                                  benchmark_figure, information_figures, plotting, save_figure)


def refresh(source, output):
    provenance = {"operation": "aggregate presentation refresh; estimates and pricing policies unchanged",
                  "source": str(source), "flavors": {}}
    comparison = []
    for flavor in ["plain", "berry"]:
        folder = source / flavor
        destination = output / flavor
        destination.mkdir(parents=True, exist_ok=True)
        groups = pd.read_csv(folder / "targeted_revenue_by_switching_group.csv")
        benchmarks = pd.read_csv(folder / "pricing_policy_benchmarks.csv")
        benchmark_figure(benchmarks, destination, flavor)
        shown = displayed_policies(benchmarks)
        reference = float(shown.loc[shown.policy == "Common pricing", "discounted_revenue"].iloc[0])
        shown["revenue_change_pct"] = 100 * (shown.discounted_revenue / reference - 1)
        shown.to_csv(destination / "pricing_policy_benchmarks.csv", index=False)
        comparison.extend(dict(flavor=flavor, **row) for row in shown.to_dict("records"))
        personal = groups.loc[groups.policy.isin(["personalized_dynamic", "Personalized pricing"])].copy()
        personal["policy"] = "Personalized pricing"
        paths = personal.copy()
        # Common coupons are constant within period for targeted low/high types.
        # Recover their price paths exactly from posted-price group means.
        common = benchmarks.loc[benchmarks.policy.isin(["optimized_common_coupons", "Common pricing"])]
        posted = groups.loc[groups.policy == "no_coupon"].copy()
        if len(common) and len(posted):
            d1, d2 = common.iloc[0][["discount1", "discount2"]]
            posted["expected_p2"] = posted.p1 * (1 - (posted.group != "middle") * d2)
            posted["p1"] *= 1 - (posted.group != "middle") * d1
            posted["policy"] = "Common pricing"
            paths = pd.concat([posted, personal], ignore_index=True)
        price_paths(paths, destination, flavor, ["pricing_mechanism", "discounts_by_consumer_type",
                    "discounts_by_household_type", "terminal_offers_by_purchase"])
        paths[["policy", "group", "p1", "expected_p2"]].to_csv(destination / "price_paths.csv", index=False)
        # Same aggregate totals as the source run, with just the two policies.
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for row in paths.itertuples():
            axes[0].plot([1, 2], [row.p1, row.expected_p2], marker="o", label=f"{row.policy}: {row.group}")
        axes[0].set(xticks=[1, 2], xticklabels=[r"$p_1$", r"$p_2$"], ylabel="Price ($)", title="Price paths")
        axes[0].legend(fontsize=7)
        axes[1].bar(shown.policy, shown.discounted_revenue, color=["#3575AD", "#D17925"])
        axes[1].set(title="Expected two-period revenue", ylabel="Revenue ($)")
        fig.suptitle(flavor.title())
        save_figure(fig, destination, ["household_coupon_results", "pricing_theory_comparison"])
        plt.close(fig)
        information = pd.read_csv(folder / "information_friction_aggregate.csv")
        information_figures(information, destination, delta=None)
        # Original pricing discount factor is recoverable from aggregate accounting.
        nonzero = groups.revenue2 != 0
        original_delta = ((groups.loc[nonzero, "discounted_revenue"] - groups.loc[nonzero, "revenue1"]) /
                          groups.loc[nonzero, "revenue2"]).median()
        provenance["flavors"][flavor] = {"original_pricing_delta": float(original_delta),
                                         "pricing_rerun": False}
    pd.DataFrame(comparison).to_csv(output / "flavor_comparison.csv", index=False)
    lines = [r"\begin{tabular}{llrr}\toprule",
             r"Flavor & Policy & Revenue & Gain (\%)\\\midrule"]
    for row in comparison:
        lines.append(f"{row['flavor'].title()} & {row['policy']} & {row['discounted_revenue']:,.1f} & {row['revenue_change_pct']:.2f}" + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (output / "flavor_comparison.tex").write_text("\n".join(lines) + "\n")
    (output / "presentation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "res_pricing1108/pricing")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "assets/pricing")
    args = parser.parse_args()
    refresh(args.source, args.output)
