# Short seminar version

`pres_short.tex` and `pres_short.pdf` are separate from the original `pres.tex`/`pres.pdf`.

This version has 37 main slides, 24 appendix slides, and the original title page. All original content is retained; the Demand Model content is now split across two main slides. The Motivation and Research Question frames use the user-approved revised introduction. Literature is now the third main slide for the progress presentation. Theory figures and tables now use delta=1. Pricing figures use common/personalized labels and p1-to-p2 paths; empirical figures retain the recovered run results. Two literal ampersands in existing author names are escaped for TeX compilation, without changing rendered wording. The Pricing Exercise includes revenue levels, gains from personalization, and one two-period price path for each flavor.

Detail buttons link from the main sequence to related appendix slides; each appendix frame has a return button to its originating main frame. Appendix slides do not increase the main slide counter. Main and appendix section navigation are separate.

For a 75-minute hard stop, budget 63 minutes of delivery and 12 for questions/interruption/appendix excursions. Suggested blocks: introduction/data motivation 8, supply theory 14, data 5, demand 11, pricing 22, conclusion 3 minutes. If discussion uses the reserve, do not extend past 75 minutes.

This is a structural edition, not an empirical refresh. Existing pricing figures remain as supplied; the Procedure slide refers to the estimated demand model. Replace them separately after the uncontrolled three/five server results are recovered; this edition does not claim those reruns have completed.

Compile from `Code/final_draft/seminar_pres`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
```

HHI is retained in the opening Motivation from Data section. Its Flavor Popularity button opens the unchanged popularity figure in the appendix, with a return to HHI. The aggregated-flavor heatmap is in the later Data section, after Data Sources and Switching Stats.

The full-versus-sample demographics comparison is now in the main Data section, immediately after Data Sources. The original table wording and values are unchanged.

The annotated utility equation remains unchanged on Demand Model. The unboxed share, novelty and habit definitions are on a separate Novelty and Habit slide, avoiding a crowded layout.

## Main sequence

| New order | Label | Title |
|---|---|---|
| 1 | short-1 | Motivation |
| 2 | short-2 | Research Question |
| 3 | short-3 | Literature |
| 4 | short-4 | HHI |
| 5 | short-6 | Supply Setting |
| 6 | short-7 | Consumer's Problem |
| 7 | short-8 | Firm's Problem |
| 8 | short-10 | First Period Optimization |
| 9 | short-14 | Optimal Prices |
| 10 | short-prices-profit | Optimal Price Ratios and Profit |
| 11 | short-18 | Information Friction: Fixed Beliefs about Switching Type |
| 12 | short-20 | Value of Observing Switching Type |
| 13 | short-21 | Data Sources |
| 14 | short-23 | Sample Statistics: Demographics |
| 15 | short-24 | Switching Stats |
| 16 | short-25 | Persistence Across Aggregated Flavor Groups |
| 17 | short-26 | Demand Setting and Objective |
| 18 | short-27 | Demand Model |
| 19 | short-history | Novelty and Habit |
| 20 | short-29 | Estimation |
| 21 | short-32 | Distributions |
| 22 | short-31 | Implied Willingness to Pay |
| 23 | short-34 | Connecting Supply and Demand |
| 24 | short-35 | Procedure |
| 25 | short-40 | Pricing Benchmarks: Plain |
| 26 | short-41 | Pricing Benchmarks: Berry |
| 27 | short-42 | Plain: Initial Offers and Subsequent Pricing |
| 28 | short-44 | Comparison |
| 29 | short-45 | Empirical Type Information: Fixed Beliefs |
| 30 | short-46 | Plain: Revenue under Type Uncertainty |
| 31 | short-47 | Plain: Personalized vs Common Pricing |
| 32 | short-57 | Plain: Price Paths under Type Uncertainty (p1 to p2) |
| 33 | short-48 | Berry: Revenue under Type Uncertainty |
| 34 | short-49 | Berry: Personalized vs Common Pricing |
| 35 | short-59 | Berry: Price Paths under Type Uncertainty (p1 to p2) |
| 36 | short-50 | Conclusion |
| 37 | short-51 | Next Steps |

## Appendix sequence

| New order | Label | Title |
|---|---|---|
| 0 | short-22 | Sample Statistics: Purchase Activity |
| 1 | short-5 | Flavor Popularity |
| 2 | short-9 | Optimal Terminal Price and Profit |
| 3 | short-11 | Optimal Profit Given Switching Type |
| 4 | short-52 | Shares and Elasticities |
| 5 | short-12 | Pricing Simulations Specification |
| 6 | short-13 | Results |
| 7 | short-15 | Optimal Profit |
| 8 | short-16 | Optimal Price Ratio |
| 10 | short-17 | Discussion |
| 11 | short-19 | Known versus Uncertain Switching Type |
| 12 | short-53 | Type Uncertainty: Initial Offers |
| 13 | short-54 | Type Uncertainty: Terminal Offers |
| 14 | short-28 | Probability and Shares |
| 15 | short-55 | Demand: Price-Control Comparison |
| 16 | short-56 | Controlled Demand Estimates |
| 17 | short-61 | Controlled Fit: Price First Stages |
| 18 | short-30 | Estimates |
| 19 | short-33 | Discussion |
| 20 | short-62 | Controlled Fit: Willingness to Pay |
| 21 | short-36 | Results: Berry Inside |
| 22 | short-37 | Results: Plain Inside |
| 23 | short-38 | Berry: Optimal Offers by Consumer Type |
| 24 | short-39 | Plain: Optimal Offers by Consumer Type |


The slide immediately after Optimal Prices puts price ratios and profit side by side. Updated figures live in `seminar_pres/assets/theory_delta1` and `seminar_pres/assets/pricing`. `refresh_pricing_slides.py` restyles recovered aggregate results; `presentation_provenance.json` records their original pricing delta (0.92). This display refresh is not a delta=1 empirical rerun.

Removed the duplicate main Plain: Price Paths slide (short-43). The information-friction slides define Pr(gamma=gamma_r)=mu_r and display mu_H=mu_0=mu_L=0.33 for the equal case. Computation retains exact thirds.
