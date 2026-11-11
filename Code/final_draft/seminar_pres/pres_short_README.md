# Short seminar version

`pres_short.tex` and `pres_short.pdf` are separate from the original `pres.tex`/`pres.pdf`.

This version has 33 main slides, 22 appendix slides, and the title page. The Pricing Exercise uses plain as the inside flavor. Its figures and tables use the uncontrolled three-flavor fit with previous-share novelty and a discount factor of 1. The empirical pricing exports are in `Results/pricing_no_disc`.

Detail buttons link from the main sequence to related appendix slides; each appendix frame has a return button to its originating main frame. Appendix slides do not increase the main slide counter. Main and appendix section navigation are separate.

For a 75-minute hard stop, budget 63 minutes of delivery and 12 for questions/interruption/appendix excursions. Suggested blocks: introduction/data motivation 8, supply theory 14, data 5, demand 11, pricing 22, conclusion 3 minutes. If discussion uses the reserve, do not extend past 75 minutes.

Compile from `Code/final_draft/seminar_pres`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
pdflatex -interaction=nonstopmode -halt-on-error pres_short.tex
```

HHI is retained in the opening Motivation from Data section. Its Flavor Popularity button opens the unchanged popularity figure in the appendix, with a return to HHI. The aggregated-flavor heatmap is in the later Data section, after Data Sources and Switching Stats.

The full-versus-sample demographics comparison is now in the main Data section, immediately after Data Sources. The original table wording and values are unchanged.

The Demand Model slide presents the utility equation, weekly purchase share, novelty, and habit together.

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
| 19 | short-29 | Estimation |
| 20 | short-32 | Distributions |
| 21 | short-31 | Implied Willingness to Pay |
| 22 | short-34 | Connecting Supply and Demand |
| 23 | short-35 | Complete Information: Type-Specific Offers |
| 24 | short-40 | What If Switching Type Is Hidden? |
| 25 | short-42 | Common Pricing When Type Is Hidden |
| 26 | short-44 | Plain: Revenue Gain from Personalized Pricing |
| 27 | short-46 | Plain: Revenue with Observed or Hidden Types |
| 28 | short-plain-complete-paths | Plain: Type-Specific and Common Price Paths |
| 29 | short-47 | Plain: Personalized vs Common Pricing |
| 30 | short-57 | Plain: Price Paths under Type Uncertainty (p1 to p2) |
| 31 | short-plain-info-table | Plain: Expected Revenue by Initial State and Prior |
| 32 | short-50 | Conclusion |
| 33 | short-51 | Next Steps |

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
| 22 | short-37 | Results: Plain Inside |
| 24 | short-39 | Plain: Optimal Offers by Consumer Type |


The slide immediately after Optimal Prices puts price ratios and profit side by side. The theory figures are in `seminar_pres/assets/theory_delta1`.

Removed the duplicate main Plain: Price Paths slide (short-43). The information-friction slides define Pr(gamma=gamma_r)=mu_r and display mu_H=mu_0=mu_L=0.33 for the equal case. Computation retains exact thirds.
