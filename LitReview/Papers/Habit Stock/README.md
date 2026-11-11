# Habit stock: decay, averaging, and estimation

Saved November 11, 2026. These are literature documents, not research inputs.

## Saved full-text PDFs

1. **Guadagni, Peter M., and John D. C. Little (1983).** “A Logit Model of Brand Choice Calibrated on Scanner Data.” *Marketing Science* 2(3): 203–238.
   - Local: [Guadagni_Little_1983.pdf](Guadagni_Little_1983.pdf)
   - Source: https://firstanalytics.com/wp-content/uploads/Guadagni-Little.MS1983.pdf
   - DOI: https://doi.org/10.1287/mksc.2.3.203
   - Relevance: loyalty is an exponentially weighted history of brand purchases. Its purchase-occasion clock differs from weekly calendar-time decay. This supports a memory specification, not a particular weekly retention rate for yogurt flavors.

2. **Anderson, Eric T. (2002).** “A Guadagni-Little Likelihood Can Have Multiple Maxima.” *Marketing Letters*.
   - Local: [Anderson_2002.pdf](Anderson_2002.pdf)
   - Author-hosted PDF: https://www.kellogg.northwestern.edu/faculty/anderson_e/htm/personalpage_files/Papers/A%20Guadagni%20Little%20Model%20Can%20Have%20Multiple%20Maxima.pdf
   - Author's summary: https://www.kellogg.northwestern.edu/academics-research/research/detail/2002/a-guadagni-little-likelihood-can-have-multiple-maxima/
   - Relevance: loyalty smoothing and its utility coefficient can trade off; likelihoods can have multiple maxima, and initialization can matter. Author summary and PDF show inconsistent pagination, so pagination is omitted here.

## Saved citation and access link; full text unavailable

3. **Sarigöllü, Emine (1998).** “Satiation and switching: the dynamic attribute satiation model meets observed choice patterns.” *Applied Stochastic Models and Data Analysis* 14(2): 175–187.
   - Publisher: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-0747%28199806%2914%3A2%3C175%3A%3AAID-ASM346%3E3.0.CO%3B2-T
   - Relevance: the publisher abstract describes estimation of retention in a dynamic attribute-satiation model using purchase data, with Monte Carlo checks.
   - Download status: publisher PDF request returned HTTP 403. No full-text PDF was saved. Notes are based on the abstract, not a full-text reading.

## Current implementation decision

Use the ordinary mean of prior observed purchase-week flavor shares for now:

    H_jt = sum_{s<t} share_js / number of prior observed purchase weeks

Set empty prehistory to zero. Missing calendar weeks do not enter the denominator. Current purchases affect subsequent choices only. This is not exponential smoothing or decay, and the papers above should not be cited as establishing this exact averaging specification.

The 0.9 weekly retention setting was provisional, not empirically established. A future decay exercise can profile retention while reestimating the demand coefficients at each candidate value, inspect likelihood curvature and multiple starts, and compare chronological holdout prediction. That is a proposed approach, not a result from the current data.

Changing cumulative stock to an average changes regressors by a time-varying denominator. Existing cumulative-stock coefficients do not become average-stock estimates by relabeling or one constant rescaling. Saved fits keep their original specification; no estimation rerun was performed for this change.
