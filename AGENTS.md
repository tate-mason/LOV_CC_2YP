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

Initialize each household at its first observed yogurt basket h0, then model
the next two consecutive purchase weeks, skipping calendar gaps and zero-yogurt
weeks. These are purchase-occasion periods, not necessarily consecutive calendar
weeks. Do not substitute hypothetical initial
states and describe the result as observed-household initialized. Use the new
no-habit estimates in the binary theory, with fixed empirical alpha and beta.
The uniform firm knows the population gamma distribution and individual history,
but not individual type or quintile. The informed firm additionally knows the
quintile. The population type distribution is assumed identical across initial
histories. No learning or posterior updating occurs in the pricing periods.
The coupon model is outside this exercise.
