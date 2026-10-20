# TOY MODEL UPDATE

## Previous work

derivative of profit w.r.t $p_{rj1}$

$$
  0 = 1 + \alpha[\sum_{k\ne j}s_{rk1}(m_{rj1} - m_{rk1}) + \delta(w_r(j) - w_r(k))] \\
  \text{s.t.} \\
  m_{rj1} = p_{rj1} - c_j \\
  m_{rk1} = p_{rk1} - c_k
$$

derivative of profit w.r.t $p_{rk2}(j)$

$$
  0 = 1 + \alpha[\sum_{r\in R} s_{rk2}(j)(m_{rk2}(j) - m_{rl2}(j)] \\
$$

## Changes

1) Monopolist firm has complete info
2) Period 1 prices depend on period 2 prices (backwards induction)
3) Now dynamic optimal pricing problem with LOV

### Solved Prices

Switchers are of type
$r\in R \,$ such that there is a distribution of types $\gamma_r$. Firms price
to this type.

**Period 1:**
$$
p^*_{rj1} = p_{rk1} + c_j - c_k - \delta[s_{rk2}(j)(p_{rj2}(j) - c_j -
\frac{1}{\alpha s_{rj2}(j)}) - s_{rj2}(j)(p_{rj2}(j) - c_k -
\frac{1}{\alpha s_{rj2}(j)})]
$$

**Period 2:**
$$
p^*_{rk2}(j) = p_{rj2}(j) + c_k - c_j - \frac{1}{\alpha s_{rj2(j)}}
$$

### Intuition

* For high types, charge monopoly prices since they will switch anyways
* For low types induce to switch to higher margin product

### Want to know

1. Is there a point in the distribution of $\gamma$ when the firm will go from
   not targeting to targeting?
2. Setting a uniform price for a setting where the distribution is known to be
   50/50
3. Graph the distribution of profits as $r$ changes

## Corrections and implementation (2026-10-20)

The formulas above are retained as the original calculations; use the corrected
conditions below and in `updates/toy_model/toy_model.tex`.

* **Outside option:** the first-order conditions omitted its contribution.
  With one initial history, set $a_{rj}=p_{rj1}-c_j+\delta w_r(j)$. Then
  $0=1+\alpha[a_{rj}-\sum_{k=1}^J s_{rk1}a_{rk}]$.
  Equivalently the bracket is
  $s_{r01}a_{rj}+\sum_{k\ne j}s_{rk1}(a_{rj}-a_{rk})$.
  The continuation difference belongs inside the share-weighted sum.
* **Terminal prices:** at an interior optimum,
  $0=1+\alpha[m_{rk2}(j)-\sum_l s_{rl2}(j)m_{rl2}(j)]$.
  Hence inside margins are equal and
  $p_{rk2}(j)-p_{rl2}(j)=c_k-c_l$, with common margin
  $-1/(\alpha s_{r02}(j))$. The extra inverse-inside-share term in the
  original period-two formula is incorrect, invalidating the substitution
  used in the original period-one formula.
* **Initial history mixture:** the code pools initial histories. Define
  $S_{rj}=\sum_hq_hs_{rj}(h)$ and
  $D_{r,kj}=\sum_hq_h\alpha s_{rk}(h)(1\{k=j\}-s_{rj}(h))$.
  The correct period-one condition is
  $0=S_{rj}+\sum_kD_{r,kj}[m_{rk1}+\delta w_r(k)]$.
  A mixture of logits is generally not itself a single logit.
* **Other writeup corrections:** use $\pi_1+\delta\pi_2$; subtract a discount
  only once from utility; posterior derivatives are generally nonzero
  (remove the original `0 =`); include $N$ consistently in aggregate-profit
  derivatives; keep the alternative-product continuation value inside its
  sum. Price identities remain implicit because shares depend on prices.

### Answers and outputs

Run `python3 toy_model.py`. Files are written to `updates/toy_model/results/`:

1. **Newborn targeting cutoff:** `newborn_discount_probability.csv` sweeps
   $q=\Pr(L)$ from 0.01 to 0.99, with the 50/50 case at $q=0.5$.
   `newborn_discount_cutoff.png` graphs the choice. There is no fixed adoption
   fee or classifier. The firm knows incumbent types exogenously and constructs
   a regular menu from the known-H optimum. The offer replaces its period-two
   prices with known-L optimal prices, holding the first-period price fixed.
   This is a pure period-two discount; both menus remain fixed as $q$ varies.
   The newborn has no previous product, so their first purchase has no switching
   bonus and conveys no information about type. Type remains unobserved in
   period two in this experiment.

   Expected gain is $(1-q)\Delta_H+q\Delta_L$. Here
   $\Delta_H=-0.04484198$ and $\Delta_L=0.02410654$ per initial newborn.
   The cutoff is $q^*=-\Delta_H/(\Delta_L-\Delta_H)=0.650369$.
   **Do not offer this discount below a 65.04% probability of L; offer it above
   that probability.** At equality the firm is indifferent. Equivalently, do
   not discount if the probability of H exceeds about 34.96%. At 50/50 the
   discount is not profitable. These numbers depend on the specified menus.

   The known-H optimal first-period price is actually lower than the known-L
   price because the firm invests in the higher future-value consumer. Therefore
   the entire low-type dynamic policy cannot be called a discount from the H
   policy. Holding first-period prices fixed isolates the discount decision.
   `newborn_prices.csv` makes this comparison explicit.

   `newborn_discount_probability.csv` also reports the freely reoptimized common
   policy at each prior. It weakly outperforms either fixed menu; a cutoff between
   fixed offers is not a discontinuity in freely optimized prices. Varying the
   probability of *sending* a fixed offer, rather than the probability of L,
   gives an affine objective and an optimal send probability of 0 or 1.

   `population_mix.csv` and `profits.png` retain the original incumbent-history
   comparisons as benchmarks.
2. `prices_50_50.csv` contains common and targeted prices for every period and
   inside purchase history. Here uniform means common **across types**, with
   product and history variation allowed. Baseline common first-period prices
   are approximately **(2.28189, 2.10486)**. Common total profit is **0.97727336**,
   targeted profit is **0.99664374**, and the targeting gain is **0.01937038**
   per initial consumer (about **1.98%** of common profit).
3. `profit_by_gamma.csv` and the last panel of `profits.png` plot conditional
   expected profit as gamma changes. They compare prices optimized for each
   gamma with the fixed common policy optimized for the 50/50 population.
   These are profit curves, not probability densities. A continuous distribution
   of profits needs a continuous distribution over gamma; the present population
   has two point masses.

### Timing and scope

Consumers maximize current utility. The firm uses backward induction, optimizing
terminal prices at the posterior induced by first-period prices. Consumers
choosing the outside option in period one exit permanently, as in the original
code. Initial history is pooled, even when type is observed. Prices are net
transaction prices bounded by [1, 8]; no separate discount restriction is imposed.
Profits normalize N to one. Three optimization starts and checked solver status
provide numerical robustness, not a global optimality proof.

### Why the benchmark results stayed the same

The old code jointly optimized all prices, already incorporating the effect of
period-one demand on period-two profits. Its numerical objective did not use the
incorrect written first-order conditions. With freely chosen terminal prices,
the joint optimum has terminal prices optimal at each reached posterior, so
explicit backward induction reproduces the benchmark optimum. The original
50/50 results pool consumers who already have an inside purchase history. The
newborn experiment is reported separately and removes that initial history.
