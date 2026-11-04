# Optimal pricing when gamma is unknown and h0 is known

```bash
python3 supply/unknown_type.py
python3 supply/unknown_type.py --self-test
```

The firm **observes h0** and knows the distribution over the consumer's fixed
but **unobserved gamma**. Results are solved separately for h0=outside and
h0=inside. There is no distribution over h0 in this exercise.

| Prior | gamma=−1.25 | gamma=0 | gamma=+1.25 |
|---|---:|---:|---:|
| Favor negative | .5 | .25 | .25 |
| Favor zero | .25 | .5 | .25 |
| Favor positive | .25 | .25 | .5 |
| Equal | 1/3 | 1/3 | 1/3 |

For each row and observed h0, the firm solves for **one p1**, and **one p2 for
each observed h1**. It does not optimize separate prices by the true gamma.
The first choice informs the firm's belief about gamma; type remains unknown.
The four prior specifications are separate optimizations under uncertainty.

With s_g(p,h)=logit(beta+alpha*p+q(h)*g), q(j)=−1 and q(0)=1:

```
max_p1 sum_g w_g*s_g(p1,h0)*(p1-c)
 + delta * sum_h1 max_p2 sum_g w_g*Pr(h1|g,h0,p1)*s_g(p2,h1)*(p2-c).
```

The maximization over p2 is **outside the sum over gamma**: the firm selects
one price against the posterior mixture, not a separate optimal price for each
type and then an average. Terminal policies are reoptimized for every candidate
p1; this accounts for both state transitions and learning.
After h1=j, posterior weights are proportional to w_g*s_g(p1,h0); after h1=0,
they are proportional to w_g*(1-s_g(p1,h0)). Consumers use the current-period
logit choice rule, with no strategic delay. Defaults: beta=2, alpha=−1,
delta=.92, cost=0. Exact equal thirds avoid the .99 total from three .33s.

## Results

`results/unknown_type/optimal_policies.csv` has eight rows: four gamma distributions times
two known initial states. Columns directly report the prior probabilities,
p1, p2_after_inside, p2_after_outside, and optimal_profit.
`supply_expected_outcomes.csv` contains the same summary, and
`supply_history_outcomes.csv` supplies the type/choice accounting behind the
expectation. Every gamma faces the **same policy** for a given observable history.

Each prior gets one readable PNG/PDF page, `optimal_policy_*.png/.pdf`:

- Left: the probabilities over the three possible gamma values.
- Upper right: optimal p1 and the two h1-specific p2 prices for both known h0s.
- Lower right: maximized expected discounted profit for each known h0.

No known-type or commitment comparisons are plotted or included as output cases.
Checks cover common prices across true types, Bayesian beliefs, probability mass,
profit reconstruction, first-order conditions, degenerate-type limits, and a
denser optimization search. Revealing gamma is used only as a self-test bound.
No empirical data are accessed. The original supply script supplies numerical
helpers; final_simple provides the equivalent standalone pricing_model helpers.

## Additional supply_information line plots

```bash
python3 supply/supply_information.py
```

This adds the original horizontal three-panel layout **alongside** the four
prior-distribution/policy figures. The gamma support remains fixed. The x-axis
is the probability of one gamma, varying smoothly from 0 to 1; the other two
gamma values each receive half the remaining probability. A separate figure is
made for each focal gamma and each known h0. Equal thirds and probability .5
are explicitly included in the grid. Defaults use 41 equally spaced points,
plus the exact equal-thirds probability.

- Prices: unknown-gamma adaptive p1 and the two committed prices.
- Expected profit: revealing gamma before pricing, gamma unknown with observed
  h1, and gamma unknown with prices committed before purchases.
- Value of observing h1: adaptive unknown-gamma profit minus committed
  unknown-gamma profit. This includes history-dependent prices and type learning.

Every case conditions on **known h0**. The revealed-gamma and commitment curves
are benchmarks in these additional information plots; the original eight
unknown-gamma policies remain unchanged.

`results/unknown_type/supply_information.png` is the Pr(gamma=+1.25) sweep with known h0=outside.
The other five variants include focal type and known h0 in the filename.
All six have PDF versions. `supply_information_results.csv` records every plotted
number. A separate manifest records the settings. Existing plots are preserved.
