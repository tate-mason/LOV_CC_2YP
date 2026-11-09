> Current workflow note: this is a theoretical binary mechanism exercise. The empirical three/five-category pipeline is documented in `../README.md` and `../demand/README.md`. Changing empirical flavor groups does not change these theoretical utilities. The calibrated empirical information exercise is in `demand/gamma_calibration.py`: fixed-belief offers, persistent-type revenue evaluation.

> final_simple edition: run commands below from final_simple. Editable defaults
> are near the top of each Python script. Supply also includes derivative figures;
> use `--skip-derivatives` to run only the three pricing cases.

# All supply cases in one runnable file

Run from the repository root:

```bash
python3 supply/supply_simulation.py
```

`supply_simulation.py` contains all solvers, calculations, checks and plotting
needed for the three cases. It does not import another model script or read any
empirical input. The default run completes all cases over the same gamma grid
and priors. It reports progress and writes CSVs plus PNG/PDF figures.

## Shared model

One inside product j and outside option 0; known fixed type `(alpha,beta,gamma)`.
The signed price coefficient satisfies `alpha<0`. The normalized inside utility,
conditional on the previous choice h, is

\[
 v(p,h)=\beta+\alpha p+q(h)\gamma,\quad q(j)=-1,\quad q(0)=1,
 \qquad s(p,h)=\operatorname{logit}^{-1}v(p,h).
\]

Positive gamma means symmetric switching/novelty preference; negative gamma means
persistence. This subtracts the raw outside switching payoff; it does not delete
that payoff. Consumers follow the current-period logit choice rule, without
strategic delay in anticipation of future prices. Type is fixed over periods.

Profit is

\[
 \Pi=s_1(p_1-c)+\delta\{s_1W_j+(1-s_1)W_0\}.
\]

Both continuation branches count. Defaults are beta=2, alpha=-1, delta=.92,
cost=0, gamma in [-3,3] at 61 points, and `pi=Pr(h0=j)` in {.25,.5,.75}.
With cost zero the reported profit also equals discounted revenue. `--cost`
allows another fixed marginal cost. Prices are unrestricted in the theoretical
model; first-period subsidies can occur.

## Case 1: known state, the base supply model

The firm observes h0 and type before choosing p1. After the first purchase, it
observes h1 and chooses its terminal price:

\[
 W_h=\max_{p_2}(p_2-c)s(p_2,h),\qquad
 0=s+\alpha s(1-s)(p_2-c).
\]

For each initial state,

\[
 \max_{p_1}\ s(p_1,h_0)(p_1-c)
 +\delta[s(p_1,h_0)W_j+(1-s(p_1,h_0))W_0].
\]

The first price has effective cost `c-delta*(Wj-W0)`. The code solves the
transformed monotone binary-logit FOC by bisection. Both optimal p1 values and
both terminal prices are exported. The two conditional optimal profits are also
exported. To compare with the two extensions, ex ante profit averages the
state-specific optimal values using the same prior pi. The baseline prices do
not depend on that prior, since h0 is observed.

## Case 2: initial state hidden, type known

The firm knows type and has prior pi, but cannot condition its first price on
h0. It observes h1 before setting p2. Aggregate initial demand is the mixture

\[
 A(p)=\pi s(p,j)+(1-\pi)s(p,0).
\]

The terminal prices and values remain the base model's Wj and W0. The common
first price maximizes

\[
 A(p_1)(p_1-c)+\delta[A(p_1)W_j+(1-A(p_1))W_0],
\]

with FOC

\[
 0=A(p_1)+A'(p_1)[p_1-c+\delta(W_j-W_0)].
\]

The mixture is across initial histories **within a fixed type**. It is not a
common price across gamma types. It is not correct to average the two utilities,
or to average state-specific optimal prices. In the Markov supply model, h0 is
irrelevant for continuation once the known-type consumer's h1 is observed.

## Case 3: commit to p1 and p2 before any purchases

The firm has the same type and initial prior but commits to two numbers before
either purchase; p2 cannot depend on h1. It maximizes

\[
 A(p_1)(p_1-c)+\delta(p_2-c)B(p_1,p_2),
\]
\[
 B(p_1,p_2)=A(p_1)s(p_2,j)+(1-A(p_1))s(p_2,0).
\]

The second-period state weights are endogenous to p1. Both prices are optimized
jointly, by profiling: reoptimize p1 at every candidate p2 and then maximize the
profile. The joint FOCs are

\[
 0=A+A'[p_1-c+\delta(p_2-c)(s(p_2,j)-s(p_2,0))],
\]
\[
 0=B+(p_2-c)[A\partial_p s(p_2,j)+(1-A)\partial_p s(p_2,0)].
\]

Expectations cover h0, endogenous first purchases and terminal logit choices.
There is no additional uncertainty about type. This case does not impose
`p1=p2`. At delta=0, the terminal price is unidentified by the objective; the
script uses a deterministic static reference price for reporting only.

## Prices, ratios and profit: exactly what is reported

For each case the history table has all four paths `(h0,h1)`:
`(0,0), (0,j), (j,0), (j,j)`. Initial-state probabilities are pi or 1-pi;
transition probabilities come from the case's own optimized first price.
Probabilities therefore differ between cases. The second-period taste shock
is integrated out.

| Column | Definition |
|---|---|
| `p1`, `p2` | Optimal offered prices on that path, including prices when a purchase does not occur |
| `price_ratio` | Path-specific p1/p2 |
| `path_probability` | Prior(h0) times Pr(h1\|h0,p1) |
| `conditional_profit` | `(p1-c)*1{h1=j} + delta*(p2-c)*s(p2,h1)` |
| `optimal_profit_given_h0` | Average conditional profit over h1 holding h0 fixed |

Path-conditional profit is not the firm's ex ante objective: it realizes the
first choice. The committed policy can have different conditional profits
across states even though its prices are state independent.

The summary table reports:

| Column | Definition |
|---|---|
| `expected_p1`, `expected_p2` | Offered prices averaged over each policy's joint distribution of h0,h1 |
| `ratio_of_expected_prices` | E[p1]/E[p2] |
| `expected_price_ratio` | E[p1/p2] under the joint path distribution |
| `optimal_profit` | Ex ante optimized discounted profit |

The two expected ratios are distinct objects; neither is a simple unweighted
mean of the four path ratios. Committed prices are fixed, so both ratios equal
p1/p2 for that case. These are ratios of offered prices, not prices conditional
on purchasing in both periods. Undefined zero-denominator ratios are exported
as missing values; default zero-cost terminal prices are strictly positive.

## Outputs

Default directory: `results/supply/`.

| File | Contents |
|---|---|
| `supply_solutions.csv` | One row per gamma/prior, all optimized policy prices and values, information gaps and FOC residuals |
| `supply_history_outcomes.csv` | Three cases times four histories for each gamma/prior |
| `supply_expected_outcomes.csv` | Three ex ante case summaries for each gamma/prior |
| `supply_cases_prior_*.png/.pdf` | Three columns (one per case), showing both periods' prices, path ratios, conditional and ex ante optimal profits |
| `supply_comparison_prior_*.png/.pdf` | Overlay of optimal profit, ratio of expected prices, expected price ratio across cases |
| `supply_history_profit_prior_*.png/.pdf` | All four path-conditional profits for each of the three cases, integrating only the terminal taste shock |
| `run_manifest.json` | Parameters, script hash and numerical residual summary |

All priors get their own figures. Figures and CSVs are regenerated in one run.
Existing files of the same names in the chosen output directory are overwritten.

## Checks and numerical interpretation

The ex ante information-value ordering must hold:

\[
 E V^{known}(h_0)\ge V^{hidden}\ge V^{committed}.
\]

The first gap values knowledge of the initial history. The second values
second-period pricing flexibility under the same initial prior. At gamma=0,
all values coincide and all price ratios equal one. At a degenerate initial
prior, known and hidden first-state information are equivalent. At delta=0,
adaptive and commitment values agree.

Every run checks path probability mass, reconstructs optimized profit from
path-conditional profits, and checks hidden and committed FOC residuals.
`--self-test` adds the limiting-case regressions. No empirical input is used.

The mixture objective can have multiple local maxima and optimal-price jumps.
Grid scans refine every detected local mode and compare boundaries; wider and
denser scans recheck the results. This is stronger than solving one FOC from one
initial guess, but is not a global optimality proof for arbitrary extreme
parameters. The default run tests all requested gamma/prior combinations. If
custom settings produce large residuals or binding numerical bounds, the script
stops rather than reporting the solution as verified.

For longer horizons, the known-state model admits backward induction on the
latest choice. With hidden histories use a belief over the unobserved state;
with full commitment optimize the entire price vector against the endogenous
state distribution. These remain separate from the empirical targeting story.
