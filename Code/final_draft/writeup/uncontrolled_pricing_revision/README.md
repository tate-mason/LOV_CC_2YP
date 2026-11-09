# Uncontrolled-only pricing rerun

The installed pricing-only job and full-rebuild pricing step now select `fits/no_control`. Demand need not be reestimated. The original novelty measure remains required. Slide TeX and existing result assets have not been edited by this correction.

From the server checkout, after syncing the corrected source files:

```bash
sbatch Code/final_simple/server/run_split_pricing.sh \
  /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID
```

Replace `run_JOB_ID` with the completed rebuild containing the original uncontrolled fit and its server-side prepared sample. The job regenerates matching weekly inputs on the server and verifies the saved likelihood. It recomputes plain/berry coupons, policy benchmarks, offer mechanism figures, and calibrated type-information results. Temporary weekly inputs are removed on exit; source samples remain on the server.

The uninformed firm chooses offers with fixed type probabilities after either first-period purchase outcome. Reported policy-comparison revenue averages full two-period paths under persistent consumer types. The CSV `profit` column retains the optimization objective; `persistent_type_revenue` reports common-evaluation revenue, and `value_of_information` is the known-type minus uninformed-policy persistent revenue.

Recover only:

```
run_JOB_ID/pricing_uncontrolled_JOB_ID/RESULTS_ONLY/
```

The export includes `no_control_fit`, aggregate `pricing` outputs, presentation-named `empirical_information` PDFs, and `pricing_provenance.json`. Provenance verifies both focal flavors used the same uncontrolled parameter files. No household inputs, internal manifests, or research logs are exported. Full manifests and source inputs stay on the server.

Pricing figure PDFs retain their filenames under `pricing/plain` and `pricing/berry`. Updated information PDFs use `empirical_information/plain` and `empirical_information/berry`. The full comparison CSV is `pricing/focal_flavor_comparison.csv`; use it to update the slide's comparison table and headline numbers after the run. Do not reuse the old 13.35% / 7.26% figures without checking the new results.

No empirical rerun was submitted from this local session.
