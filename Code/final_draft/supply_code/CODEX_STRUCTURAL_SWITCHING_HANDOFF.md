# Codex handoff: refactor two-period structural pricing workflow

> **Purpose:** Bring the existing mathematical derivations, Python/SymPy project, numerical solver, tests, figures, and LaTeX exports into agreement with the **symmetric switching** model settled on October 30, 2026. Implement the changes in the existing repository rather than replacing it with an unrelated project. **Read this entire document before editing.**

## Your job / definition of done

1. Inspect repository layout, current formula conventions, tests, and plotting scripts. Preserve working interfaces where practical.
2. Replace the older *asymmetric* entry-only switching utility with symmetric switching across inside and outside choices. Keep the outside deterministic utility **normalized to zero after subtraction**, not by deleting its switching payoff.
3. Make both period-2 continuation histories explicit. Period 1 must integrate over the **two** realized period-1 states, not just the inside branch.
4. Derive all FOCs, optimized prices, derivatives, continuation values and ratios from **one canonical model**; no hand-coded outdated formulas in isolated scripts.
5. Add or refactor symbolic SymPy derivations, analytical numerical evaluators, independent finite-difference checks, regression tests, and reproducible plots.
6. Generate a clean LaTeX appendix with mathematical definitions, FOCs, derivations (including envelope theorem), conditional comparative statics, proof statements and assumptions. Do not assert unproven global signs.
7. Run the tests and reproducibility script. Report any existing modules/plots that are superseded and any discrepancies with previous results.

## Canonical economic model

- One inside product `j` and outside option `0`, periods `t ∈ {1,2}`, consumer type `r`. Firm knows `r` (type-specific pricing), conditional on given initial history `h0`. When implementing a pooled/non-discriminatory variant, treat it as a different model rather than silently combining types.
- Previous choice `x_{t-1} ∈ {j,0}`. Current realized choice `x_t ∈ {j,0}`. **Do not confuse a hypothetical switching indicator with a realized choice probability.**
- Primitive deterministic utilities before normalization:

  \[
  V_{rjt}^{\mathrm{raw}}=\beta_r+\alpha p_{rjt}+\gamma_r\mathbf{1}\{x_{t-1}=0\},\qquad
  V_{r0t}^{\mathrm{raw}}=\gamma_r\mathbf{1}\{x_{t-1}=j\}.
  \]

  Add independent Type-I extreme-value taste shocks to both alternatives. The switching preference applies to **either direction**: entering or exiting the inside product. Negative `gamma` = switching aversion; positive `gamma` = novelty preference.
- Subtract raw outside deterministic utility from both alternatives. Normalized outside deterministic utility = 0. Define the **signed previous-state indicator**

  \[
  q(h_{t-1})=\begin{cases}-1,&x_{t-1}=j\\+1,&x_{t-1}=0.\end{cases}
  \]

  The normalized *inside-versus-outside utility difference* is

  \[
  v_{rjt}(h_{t-1})=\beta_r+\alpha p_{rjt}(h_{t-1})+q(h_{t-1})\gamma_r.
  \]

- Shares, holding history fixed:

  \[
  s_{rjt}=\frac{e^{v_{rjt}}}{1+e^{v_{rjt}}},\quad s_{r0t}=1-s_{rjt},\quad
  \partial_p s_{rjt}=\alpha s_{rjt}s_{r0t},\quad
  (\partial_{\gamma_r}s_{rjt})_{p}=q(h_{t-1})s_{rjt}s_{r0t}.
  \]

  Total derivative along optimal pricing is

  \[
  \frac{ds_{rjt}^*}{d\gamma_r}=s_{rjt}^*s_{r0t}^*\left(q(h_{t-1})+\alpha\frac{dp_{rjt}^*}{d\gamma_r}\right).
  \]

- Assumptions: `alpha < 0`, `0 < delta < 1`, marginal cost `c_j` fixed, fully interior binary logit shares. Prices **not constrained by cost** (allow introductory subsidies). Each continuation value is conditioned on a realized discrete state, not directly on the numerical period-1 price. No extra dependence of continuation preference, price, or costs on `p1`. State type `r` does not evolve.

## Backward induction and key algebra

### Terminal period

  \[
  V_{r2}(a;\gamma_r)=\max_{p_{rj2}(a)} s_{rj2}(a)[p_{rj2}(a)-c_j],\qquad a\in\{j,0\}.
  \]

  \[
  0=s_{rj2}+\alpha s_{rj2}s_{r02}(p_{rj2}-c_j),\quad
  \boxed{p_{rj2}^*(a)=c_j-\frac{1}{\alpha s_{r02}(a)}}.
  \]

  This is an implicit pricing equation; resolve with a root finder or optimization routine. With the envelope theorem:

  \[
  \boxed{\frac{dV_{r2}(a)}{d\gamma_r}=-\frac{q(a)s_{rj2}^*(a)}{\alpha}},\qquad
  \boxed{\frac{dp_{rj2}^*(a)}{d\gamma_r}=-\frac{q(a)s_{rj2}^*(a)}{\alpha}}.
  \]

  In particular, for `u=s_{rj2}(j)` and `v=s_{rj2}(0)`, `q(j)=-1`, `q(0)=1`:

  \[
  V_2'(j)=u/\alpha,\quad V_2'(0)=-v/\alpha,\quad
  \boxed{\Delta V_2'=(u+v)/\alpha},\qquad \Delta V_2=V_2(j)-V_2(0).
  \]

  The *total* equilibrium share derivative for a fixed continuation state is
  \(s_{rj2}'(a)=q(a)s_{rj2}(a)[1-s_{rj2}(a)]^2\).

### First period: **both continuation branches**

  \[
  \Pi_r(p_1;h_0)=s_{rj1}(h_0)(p_1-c_j)+\delta\left[s_{rj1}(h_0)V_{r2}(j)+(1-s_{rj1}(h_0))V_{r2}(0)\right].
  \]

  Equivalently, \(\Pi_r=\delta V_2(0)+s_{rj1}[p_1-c_j+\delta\Delta V_2]\).

  **Price dependence:** at fixed `h0`, `V2(j)` and `V2(0)` do *not* depend directly on `p1`; first-period price only changes the probabilities attached to the states. Thus

  \[
  \frac{\partial}{\partial p_1}\left\{\delta[s_1V_2(j)+(1-s_1)V_2(0)]\right\}
  =\delta(\partial_{p_1}s_1)[V_2(j)-V_2(0)].
  \]

  \[
  0=s_1+\alpha s_1(1-s_1)[p_1-c_j+\delta\Delta V_2],\quad
  \boxed{p_{rj1}^*(h_0)=c_j-\frac{1}{\alpha s_{r01}(h_0)}-\delta\Delta V_{r2}}.
  \]

  The `static FOC component` \(c_j-1/[\alpha s_{r01}(h_0)]\) uses the **share at the actual dynamic optimum**, and is NOT the separately optimized static-monopoly price. For a true static-versus-dynamic counterfactual, solve a second optimization with `delta=0`.

  \[
  \boxed{\Pi_r^*(h_0)=s_1^*(p_1^*-c_j)+\delta[s_1^*V_2(j)+(1-s_1^*)V_2(0)]}.
  \]

### Comparative statics: symmetric switching

  Set \(q=q(h_0)\), \(s=s_{rj1}^*(h_0)\), \(r_0=1-s\), \(u=s_{rj2}^*(j)\), \(v=s_{rj2}^*(0)\), \(B=u+v\).

  \[
  \boxed{\frac{dp_1^*}{d\gamma_r}=-\frac{q s+\delta r_0(u+v)}{\alpha}}.
  \]

  \[
  \boxed{\frac{d\Pi_r^*}{d\gamma_r}=\frac{-q s+\delta(su-r_0v)}{\alpha}}.
  \]

  **Do not reuse** the older, entry-only expression \(-[z_1 s+\delta s_0s_2(0)]/\alpha\). It came from a different utility specification.

  For the **analytical second derivative** (avoids noisy finite-differencing of optimized prices):

  \[
  s'=s r_0^2(q-\delta B),\qquad
  u'=-u(1-u)^2,\qquad v'=v(1-v)^2,\qquad B'=u'+v'.
  \]

  \[
  \boxed{\frac{d^2p_1^*}{d\gamma_r^2}=-\frac{s r_0^2(q-\delta B)^2+\delta r_0 B'}{\alpha}}.
  \]

  Derive each expression in the symbolic code, then verify it against independent central differences of the fully reoptimized model. `dprofit/dgamma` may have different signs by initial history; **do not claim monotonicity for all states**.

### Price ratios: define histories explicitly

  - State-matched (descriptive, **not** necessarily a realized single consumer path): \(R_j=p_1^*(h_0=j)/p_2^*(h_1=j)\), \(R_0=p_1^*(h_0=0)/p_2^*(h_1=0)\).
  - **Induce-then-harvest**: \(R_{0\to j}=p_1^*(h_0=0)/p_2^*(h_1=j)\), conditional on acquisition in period 1. This is NOT the expectation over states.
  - For any path-specific ratio, use the quotient rule:

  \[
  \boxed{\frac{d}{d\gamma}\left(\frac{p_1}{p_2}\right)=\frac{p_2p_1'-p_1p_2'}{p_2^2}}.
  \]

  - Do not replace a ratio of expectations with an expected ratio or vice versa. If an **ex-ante** ratio is wanted, formally define the object first.

## Repository migration plan

Expected existing layout is roughly `model/`, `pricing/`, `comparative_statics/`, `checks/`, `export/`, and `run_all.py`; discover the real layout before editing. Suggested mapping:

| Area | Change |
|---|---|
| `model/symbols.py` | Define `beta_r`, `gamma_r`, `alpha`, `delta`, `c_j`, state symbols; put assumptions on alpha/delta in checks. |
| `model/utility.py` | Introduce raw utilities for BOTH alternatives, normalization, and signed `q(history)`. Deprecate ambiguous 0/1 `z` from final formulas. |
| `model/shares.py` | Implement conditional logit shares; direct and total partial derivatives with correct state signs. |
| `model/profit.py` | Separate `profit2`, `V2(j)`, `V2(0)`, `profit1` including **both** branches and optimized `Pi1`. |
| `pricing/period2_foc.py`, `period2_price.py` | Derive period-2 FOC and implicit optimal price; evaluate separately for `j`, `0`. |
| `pricing/period1_foc.py`, `period1_price.py` | Derive period-1 FOC; continuation values constant w.r.t. `p1` conditional on states. |
| `comparative_statics/gamma_*` | Rebuild `dp2/dgamma`, `dV2/dgamma`, `dp1/dgamma`, `d2p1/dgamma2`, `dPi/dgamma`, path-specific price ratio derivatives. |
| `checks/` | Add symbolic identity checks, finite-difference checks, state enumeration and parameter regression tests. |
| `export/` | Generate TeX from verified SymPy expressions and include definitions/assumptions and evidence checks. |
| `figures/` | Both periods' prices, price-derivative panels, induce/harvest ratio, decomposition, continuation difference. |
| `run_all.py` | Run packages via `python -m ...` or package imports; don't execute package modules as bare paths when imports require repo root. |

**Avoid relying on incomplete older scripts** (some previous copies of `gamma_derivative_plots_all_again.py` and `period1_price_decomposition.py` only contained solver definitions and no executable plot generation). Use the complete runnable implementation below as the tested reference and refactor into your existing modules.

## Numerical benchmark and required figures

Benchmark parameter set: \(\beta=2,\alpha=-1,\delta=0.92,c_j=0,\gamma\in[-3,3]\). Unconstrained first-period prices (subsidies allowed). Sample at least 401 gamma values and verify the FOCs/residuals. Plot, at minimum:

1. Both optimal `p1` curves by `x0`, both optimal `p2` curves by `x1`.
2. `dp1/dgamma` and **analytic** `d2p1/dgamma2` for both initial states.
3. `dPi*/dgamma` for both initial states.
4. Both state-matched `d(p1/p2)/dgamma` and induce-then-harvest `d[p1(0)/p2(j)]/dgamma`.
5. For each initial state: `p1`, `c-1/(alpha*s01)` evaluated at dynamic optimum, and `-delta*DeltaV`; verify summation.
6. `DeltaV2` and the *true counterfactual static monopoly price* if desired, clearly distinct from FOC-component decomposition.

## Complete runnable numerical reference implementation

Save the following as `structural_switching_reference.py`; execute from repo root with `python structural_switching_reference.py`. Dependencies: `numpy scipy matplotlib`. It writes `outputs/*.png` and `outputs/model_results.csv`. This is the **authoritative computational reference for the migration**; split it into modules as you prefer.

```python
"""Two-period symmetric switching model: solve, differentiate, verify, plot.

Run: python structural_switching_reference.py
Requires: numpy, scipy, matplotlib.  Outputs to ./outputs.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

BETA, ALPHA, DELTA, COST = 2.0, -1.0, 0.92, 0.0
GAMMAS = np.linspace(-3.0, 3.0, 601)
OUTPUT = Path('outputs')


def expit_safe(x):
    from scipy.special import expit
    return expit(x)


def inside_share(price, gamma, state, beta=BETA, alpha=ALPHA):
    """state='j': existing customer, state='0': outside previously."""
    q = -1 if state == 'j' else +1 if state == '0' else None
    if q is None:
        raise ValueError("state must be 'j' or '0'")
    return expit_safe(beta + q * gamma + alpha * price)


def p2_opt(gamma, state):
    """Unique finite price solving p=c-1/[alpha*(1-s)] (interior)."""
    f = lambda p: p - COST + 1 / (ALPHA * (1 - inside_share(p, gamma, state)))
    return brentq(f, -30.0, 40.0, xtol=1e-13)


def evaluate(gamma, initial_state):
    p2j, p20 = p2_opt(gamma, 'j'), p2_opt(gamma, '0')
    u, v = inside_share(p2j, gamma, 'j'), inside_share(p20, gamma, '0')
    Vj, V0 = (p2j-COST)*u, (p20-COST)*v
    dV = Vj - V0
    q = -1 if initial_state == 'j' else 1
    f = lambda p: p-COST+1/(ALPHA*(1-inside_share(p, gamma, initial_state)))+DELTA*dV
    p1 = brentq(f, -30.0, 40.0, xtol=1e-13)
    s = inside_share(p1, gamma, initial_state)
    r = 1-s
    pi = s*(p1-COST)+DELTA*(s*Vj+r*V0)
    static_piece = COST - 1/(ALPHA*r)
    dynamic_piece = -DELTA*dV
    # Envelope derivatives: Vj'=u/alpha, V0'=-v/alpha.
    d_pi = (-q*s + DELTA*(s*u-r*v))/ALPHA
    dp1 = -(q*s + DELTA*r*(u+v))/ALPHA
    # ds/dgamma = s*r*(q + alpha*dp1) = s*r^2*(q - delta*(u+v)).
    ds = s*r*r*(q-DELTA*(u+v))
    dB = -u*(1-u)**2 + v*(1-v)**2
    d2p1 = -(s*r*r*(q-DELTA*(u+v))**2 + DELTA*r*dB)/ALPHA
    dp2j = -(-1)*u/ALPHA
    dp20 = -(+1)*v/ALPHA
    matched_p2 = p2j if initial_state == 'j' else p20
    matched_dp2 = dp2j if initial_state == 'j' else dp20
    d_matched_ratio = (dp1*matched_p2-p1*matched_dp2)/matched_p2**2
    d_induce_ratio = (dp1*p2j - p1*dp2j)/p2j**2 if initial_state == '0' else np.nan
    assert np.isclose(p1, static_piece+dynamic_piece, atol=1e-9)
    return dict(gamma=gamma, initial_state=initial_state, p1=p1, p2j=p2j,
                p20=p20, profit=pi, V2j=Vj, V20=V0, DeltaV=dV,
                static_component=static_piece, continuation_adjustment=dynamic_piece,
                dp1=dp1, d2p1=d2p1, dprofit=d_pi,
                d_matched_ratio=d_matched_ratio, d_induce_ratio=d_induce_ratio)


def verify():
    h = 1e-4
    for g in [-2.4, -1.2, 0, 0.7, 2.6]:
        for state in ['j', '0']:
            result = evaluate(g, state)
            plus, minus = evaluate(g+h, state), evaluate(g-h, state)
            for value, key in [('dp1', 'p1'), ('dprofit', 'profit')]:
                numerical = (plus[key]-minus[key])/(2*h)
                assert np.isclose(result[value], numerical, atol=2e-7), (g,state,value,result[value],numerical)
            numerical2 = (plus['dp1']-minus['dp1'])/(2*h)
            assert np.isclose(result['d2p1'], numerical2, atol=2e-7), (g,state,'d2p1')
            numerical_ratio = (plus['p1']/ (plus['p2j'] if state=='j' else plus['p20']) -
                               minus['p1']/(minus['p2j'] if state=='j' else minus['p20']))/(2*h)
            assert np.isclose(result['d_matched_ratio'], numerical_ratio, atol=2e-7)
            if state == '0':
                numerical_ih = (plus['p1']/plus['p2j'] - minus['p1']/minus['p2j'])/(2*h)
                assert np.isclose(result['d_induce_ratio'], numerical_ih, atol=2e-7)
    print('All analytical derivative checks passed.')


def plot_series(rows, key, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for state,label in [('j',r'Initially inside ($x_0=j$)'),('0',r'Initially outside ($x_0=0$)')]:
        ax.plot(GAMMAS, [r[key] for r in rows[state]], label=label, linewidth=2)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xlabel=r'$\gamma$', ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT/filename, dpi=180)
    plt.close(fig)


def main():
    verify()
    OUTPUT.mkdir(exist_ok=True)
    rows = {state:[evaluate(float(g),state) for g in GAMMAS] for state in ['j','0']}
    with (OUTPUT/'model_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows['j'][0]))
        writer.writeheader()
        for state in ['j','0']:
            writer.writerows(rows[state])
    plot_series(rows, 'p1', r'$p_1^*$', 'price1.png', 'Optimal period-1 prices')
    plot_series(rows, 'dp1', r'$dp_1^*/d\gamma$', 'dp1.png', 'First derivative of period-1 price')
    plot_series(rows, 'd2p1', r'$d^2p_1^*/d\gamma^2$', 'd2p1.png', 'Second derivative of period-1 price (analytic)')
    plot_series(rows, 'dprofit', r'$d\Pi^*/d\gamma$', 'dprofit.png', 'Optimized profit derivative')
    plot_series(rows, 'd_matched_ratio', r'$d(p_1/p_2)/d\gamma$', 'matched_price_ratio.png', 'State-matched price-ratio derivatives')
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    ax.plot(GAMMAS, [r['d_induce_ratio'] for r in rows['0']], linewidth=2)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set(xlabel=r'$\gamma$', ylabel=r'$d[p_1(0)/p_2(j)]/d\gamma$', title='Induce-then-harvest price-ratio derivative')
    fig.tight_layout();fig.savefig(OUTPUT/'induce_harvest_ratio.png',dpi=180);plt.close(fig)
    for state in ['j','0']:
        fig,ax=plt.subplots(figsize=(8.6,5.2))
        for key,label in [('p1',r'$p_1^*$'),('static_component','Static FOC component'),('continuation_adjustment',r'$-\delta\Delta V_2$')]:
            ax.plot(GAMMAS,[r[key] for r in rows[state]],label=label,linewidth=2)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(xlabel=r'$\gamma$',ylabel='Price / component',title=f'First-period decomposition: initial state {state}')
        ax.legend();fig.tight_layout();fig.savefig(OUTPUT/f'decomposition_{state}.png',dpi=180);plt.close(fig)
    print('Saved outputs to', OUTPUT.resolve())

if __name__ == '__main__':
    main()

```

## Acceptance tests and caveats

- Run `python structural_switching_reference.py` from the project root and confirm **all analytical derivative checks passed**. Inspect all plots and validate no kinks from imposed `p >= c` bounds.
- Implement full SymPy checks: verify `diff(profit1, p1)` equals the stated FOC after substituting logit shares; check both continuation branches; compare algebraic analytic derivative identities; compare terminal and initial state cases.
- The **period-1 price** is implicitly defined because `s01` depends on that same price. Never treat the RHS share as exogenous when computing parameter derivatives.
- When differentiating `V2(a)` with respect to `gamma`, use the envelope theorem, holding *period-2 optimal price fixed inside its partial derivative*. When differentiating it with respect to `p1`, it is constant given a fixed continuation state.
- Preserve strict distinction between `q(h)` (signed **potential choice** effect), the realized choice `x_t`, `s_jt` (choice probability), and the potential switching indicator for each option.
- If a consumer type mixture, posterior updating, advertising, multi-product choice, price commitment, or price-dependent future utility is present in repository, **do not pretend this one-type two-choice reference directly solves that extension**. Keep original extensions behind explicit interfaces and update them separately with additional derivatives.
- Numerical solver uses bracket `[-30,40]` valid for the benchmark only; increase or dynamically bracket for broad parameter sweeps. All derivatives are conditional on each discrete initial state.
- In previous exploratory plots the second derivative was estimated by twice taking finite differences of optimizer output, which produced noise. The code above implements the analytical formula and checks it independently.
- **No invented proof:** when a comparative-static sign is not global, state it as conditional or show a numerical counterexample. Report model assumptions explicitly beside all propositions.

## Suggested Codex execution prompt

> Read `CODEX_STRUCTURAL_SWITCHING_HANDOFF.md` and refactor the current repository to implement its canonical symmetric-switching two-period model, formulas, tests, computational solver and figures. First inspect existing modules and list which old expressions are invalid, then make minimally disruptive edits. Create reproducible SymPy derivation modules and LaTeX export, and run all validation checks. Use the embedded complete Python script as the numerical ground truth. Explain any extensions of the repository not handled by this canonical benchmark instead of silently overwriting them.
