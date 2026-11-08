# Pricing only, reusing an existing full rebuild

Submit the updated shell file on the research server:

```bash
sbatch Code/final_simple/server/run_split_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID
```

Replace JOB_ID with the existing completed rebuild's job ID. The shell calls
`Code/final_simple/demand/plain_berry_pricing.py` using that run's saved controlled
fit. It does not rebuild raw data, matching, summaries or demand estimates.
If full_rebuild already removed weekly_inputs.csv, only the temporary weekly
export is reconstructed from the existing prepared sample and saved fit, with
a likelihood check. To use an existing weekly export, pass its server path as
the second argument; a missing explicit path stops the job.

Original pricing and new information-friction tables/figures are written to
`run_JOB_ID/pricing_only_NEW_JOB_ID/pricing/plain_berry`. The existing export
allowlist stages these alongside the prior run's fit/summary outputs in
`run_JOB_ID/pricing_only_NEW_JOB_ID/RESULTS_ONLY`. Recover only that directory.
All data, temporary weekly records and individual policy files stay on the
server. No ZIP is needed.

## Type uncertainty

The three states mean low, medium and high **predicted types**, not fixed
numerical gamma values. The existing posterior novelty ranking defines bottom
25%, middle 50%, and top 25% groups. Each group's mean fitted posterior novelty
coefficient represents its gamma; there is no hardcoded gamma support.
The four probability vectors over [low, medium, high] are:

| Scenario | Low | Medium | High |
|---|---:|---:|---:|
| Equal | 1/3 | 1/3 | 1/3 |
| Low more likely | .50 | .25 | .25 |
| Medium more likely | .25 | .50 | .25 |
| High more likely | .25 | .25 | .50 |

These type probabilities remain fixed in every subsequent basket, following
the current gamma_uncertainty.py information specification. Initial history is
observed. Compare known type with uncertain type; no learning or commitment
case enters this extension. Price sensitivity and habit remain at each
household's posterior means, and all eligible households receive optimized
coupons. The full multinomial demand, stock histories and coupon constraints
remain; the theory's symmetric binary utility is not imposed on fitted demand.

`profit` reports the fixed-belief objective in the uncertain case.
`persistent_type_revenue` separately evaluates its chosen offers with the same
gamma in both periods. They can differ. The known-type bound is checked on the
latter; the signed known-minus-fixed-belief objective is reported as defined.

Each focal folder adds information_friction_aggregate.csv, by_group.csv,
states.csv, settings.json and results.tex (each with the information_friction_
prefix), plus profits, first_prices, terminal_prices and information_value in
PDF/PNG. Figures use the four priors and observed initially-inside/outside
panels. Settings/state tables record estimated gamma representatives and fixed
probabilities. All new aggregate artifacts are in the RESULTS_ONLY allowlist.

Shell/Python syntax was checked locally. No empirical execution was performed.
Server-only numerical checks: `python3 Code/final_simple/demand/split_pricing.py
--self-test`.
