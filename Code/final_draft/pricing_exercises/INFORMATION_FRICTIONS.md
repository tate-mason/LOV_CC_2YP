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

These type probabilities remain fixed in every subsequent state, following the
current gamma_uncertainty.py information specification. Compare known with
uncertain type; there is no learning or commitment case in this extension.

## Calibrate supply theory to observed prices

The gamma extension now uses the binary supply theory directly, implemented in
`Code/final_simple/demand/gamma_calibration.py`. For each focal flavor:

1. Take actual observed posted prices exactly 7 and 14 days after each household's
   initial observation. Do not carry initial prices forward. Exclude households
   missing either positive observed price; report exclusion counts.
2. Form mean observed p1 and p2 targets by predicted type and observed initial
   inside/outside state. Inside means the initial basket contained the focal flavor.
3. Hold the three fitted gamma group means fixed. Calibrate a common signed
   alpha<0 and beta to the price targets by household-count-weighted relative
   least squares. Cost is fixed at zero and delta defaults to .92. Three starts
   are compared; failed fits, binding parameter bounds and rank-deficient price
   targets stop the run.
4. Solve the calibrated symmetric binary theory for both initial states, known
   type and uncertain type, under the four fixed probability vectors above.
   Numerical price searches audit every local grid mode and use a wider scan.

This assumes observed offered prices target the known-type model's optimal
prices. It is a calibration assumption, not a claim that observed retail prices
were personalized. The price fit can be imperfect; output records the fit errors.
The supply intercept and price sensitivity are calibrated here rather than
copied from demand. Fitted novelty group means map to theoretical gamma; the
binary theory does not separately include the fitted own-stock habit coefficient
or rival-flavor identities. The original fitted-demand coupon exercise still
runs separately, but its optimized prices are never calibration targets.

## Results

Observed calibration targets and fitted target-cell prices stay in server memory.
No calibration-target CSV, observed-versus-fitted scatter plot, household posterior
file, individual policy file or household scatter plot is saved. State-policy
export is disabled. Calibration parameters and one overall fitting-error statistic
are recorded in `information_friction_settings.json`.

`information_friction_aggregate.csv` reports theoretical prices and profit per
model consumer by prior, initial state and information regime. These are not
empirical population revenue totals. `information_friction_by_group.csv` reports
known-type predictions and performance of the uncertain policy conditional on
each true type. `information_friction_states.csv` records type probabilities and
unchanged beliefs. `information_friction_results.tex` is the theory result table.
The profits, first_prices, terminal_prices and information_value figures are
saved as information_friction_*.pdf/png.

The uncertain-case profit is the fixed-belief objective used by the current
theory file. Its known-minus-uncertain difference need not be nonnegative for
arbitrary empirical calibrations. Conditional persistent-type policy performance
is evaluated separately in the by-group table and checked against known-type
optimal profit.

Only aggregate comparison tables, aggregate figures and model settings enter
RESULTS_ONLY. No ZIP is needed. Requires NumPy, pandas, SciPy
and matplotlib on the server.

Python/shell syntax and export wiring were checked locally; no empirical
execution was performed. The server-only self-test checks recovery of known
calibration parameters from theoretical price targets, degenerate type priors,
fixed beliefs and wider price searches:

```bash
python3 Code/final_simple/demand/split_pricing.py --self-test
```
