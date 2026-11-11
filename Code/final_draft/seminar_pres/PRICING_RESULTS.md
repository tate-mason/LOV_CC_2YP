# Completed revised pricing run

Imported from `new_pricing_res/`; aggregate outputs only. The saved likelihood
reconstructs exactly and the solver hash matches the corrected local source.
Coverage: 1,839 eligible pricing environments, with 1,324 initially inside plain
and 515 initially outside. The demand fit was not reestimated; cumulative habit
history is retained. Price sensitivity and habit are known estimated means;
only novelty is hidden under limited information. Both regimes adapt to history.

## Fitted-distribution results

All prices are means across the relevant environments. Revenue is expected
revenue per model consumer over both periods; gamma types persist.

| Initial state | Information | Observed price | Optimal p1 | Expected p2 | Revenue |
|---|---|---:|---:|---:|---:|
| Inside | Perfect | $2.399 | $2.460 | $2.541 | $10.889 |
| Inside | Limited | $2.399 | $2.469 | $2.539 | $10.873 |
| Outside | Perfect | $2.293 | $1.597 | $1.900 | $4.076 |
| Outside | Limited | $2.293 | $1.729 | $1.962 | $3.988 |

Knowing novelty adds $0.016 per initially inside consumer (0.15%) and $0.088 per
initially outside consumer (2.21%). Outside consumers receive lower acquisition
prices and generate less total revenue, while offering a larger incremental
information gain. These are model comparisons, not realized household outcomes.

## Validation

Node aggregates reconstruct both group and overall totals. Purchase-branch masses
sum to the integration weights; conditional branch revenues reconstruct total
revenue. Full-information revenue weakly dominates limited-information revenue
for every exported type/state/prior cell. Every supplied figure-source hash
matches. No individual households or purchase histories were opened locally.

The new plots mirror supply: optimal prices by gamma, price ratios and profit by
history, type-specific/common price paths, information revenues and gains. The
price-change plot compares offers with observed prices. Earlier binary and shared
schedule calculations remain labeled as earlier appendix benchmarks.
