# Empirical results refresh

Source: `/Users/tate/LOV_CC_2YP/res_pricing1108`. Its demand estimate tables match `res1108`. The original exports were left intact. Only estimates, covariance matrices, aggregate tables and figures were read. No empirical estimation or household processing was run.

## Replacements

- Refreshed no-control demand estimates, standard errors, WTP and distributions; novelty is -0.7732 (SE 0.0095), habit 0.0369. Mean novelty WTP is -$1.7451; habit WTP is $0.0833.
- Refreshed berry/plain revenues, optimal coupons, mechanisms, household-type figures and comparison table. Personalized gains versus no coupons are 13.35% for plain and 7.26% for berry.
- Updated the intro and slide numerical claims. Estimation equations now use the configured 128 draws instead of 64. Saved draw counts and convergence metadata are not included in the export, so those settings cannot be independently certified from the pulled files.
- Refreshed the existing summary-statistics figures and dynamics table from the new export. The full/yogurt sample comparison numbers are unchanged.

## Additions

- Controlled versus uncontrolled demand comparison in the main paper and slide appendix. Full controlled estimates, WTP, distributions and price first stages in the paper appendix. Pricing uses the controlled fit.
- Common-coupon benchmarks: plain gains 3.53% versus no coupons; berry's optimum is no coupon. Personalized policies gain 13.35% and 7.26%, respectively.
- Activated previously pending benchmark plots and conditional terminal-offer plots for both plain and berry.
- Empirical fixed-belief type-friction setup and revenue comparisons in the main paper and slides. Detailed tables and initial/terminal offers in appendices.
- Updated slide history equations to match weekly basket shares and cumulative own-flavor shares.

## Information-friction evaluation

The original exported `profit` and `value_of_information` fields compare different belief-based objectives. They are not used as comparable realized revenue. New presentation charts use `persistent_type_revenue` for both known and fixed-belief policies, with differences computed from those aggregate values. This evaluation does not update the firm's beliefs or reoptimize its offers. Known-type revenue weakly exceeds fixed-belief-policy revenue in every reported comparison. The fixed-belief optimization objective remains explicitly distinguished in the text.

## Tables and transformations

Price coefficients are reported on the natural coefficient scale. Mean and SD standard errors use the delta method with the exported covariance matrix. WTP population SDs are labeled separately from estimation standard errors. No interpretation prose was added beyond definitions and the explanation needed to distinguish the two revenue measures.
