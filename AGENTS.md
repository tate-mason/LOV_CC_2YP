# Research data boundary

Never open, copy, download, generate or persist raw research records, household
panels, purchase histories, weekly inputs, matched retail records or data
partitions on the local computer. Empirical I/O is research-server-only.
Do not bypass Code/privacy/server_data_policy.py or use local path overrides.
Only estimates, predictions without household identifiers, and aggregates may
be recovered locally. Preserve user-authored files. Inspect filenames/source
only when auditing; never print raw record contents. Do not change Git history
or remove user files without an explicit current instruction.

# Plot conventions

All welfare and revenue plots must be expressed as percentage changes relative
to the relevant baseline. Label axes and captions clearly with the percent-change
units and identify the baseline.
Use 100 × (new − baseline) / |baseline| so improvements remain positive even
when baseline welfare is negative. Flag zero baselines as undefined.

# Empirical theory analogue

The current option-1 experiment is yogurt versus no yogurt on consecutive
observed shopping weeks, including weeks with no yogurt; calendar gaps
do not advance time. Observations are weekly, not daily or trip-level.
Initialize at the first observed shopping week, then model two further observed
shopping weeks. Reestimate no-habit category switching while
retaining plain/berry/other flavor tastes. Report both a common yogurt price
and separate flavor prices in separate figures/tables, using fixed empirical
alpha and flavor beta in pricing. Equal optima are acceptable under the simplified
model. Do not present population-input benchmarks as observed-household initialized.
The proposed pooled-category likelihood is binary at the weekly level: any yogurt
versus no yogurt, regardless of flavor or package quantity. Do not introduce the
old seven-opportunity outside-count construction or a within-week consumption
sequence into this new specification. Gamma is shared across flavors.
The uniform firm knows the population gamma distribution and individual history,
but not individual type or quintile. The informed firm additionally knows the
quintile. The population type distribution is assumed identical across initial
histories. No learning or posterior updating occurs in the pricing periods.
The coupon model is outside this exercise.
Offered prices must be nonnegative in every pricing period and history. Optimize
subject to this constraint; do not merely clip plotted prices after solving.
