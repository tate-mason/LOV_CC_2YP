"""Supply mathematics: symbols, utilities, FOCs, and comparative statics.

This is the single source of the theoretical derivatives. supply_simulation.py
imports these formulas when it evaluates and plots them. This file never reads
empirical data. Running it directly prints equations; it does not run a simulation.
"""
from pathlib import Path
import numpy as np
import sympy as sp

# Names like p1, gamma and sj1 below are mathematical SYMBOLS, not estimated numbers.
# sp.diff(expression, gamma) differentiates an expression with respect to gamma.
# sp.simplify(expression) rearranges the expression without changing its meaning.
# To choose numeric parameters, edit supply_simulation.py instead.

# === 1. Symbols, signed histories, utility, and choice probabilities === #



alpha, beta, gamma, delta, c = sp.symbols("alpha beta gamma delta c", real=True)

p1, p2, p2j, p20 = sp.symbols("p_1 p_2 p_{2j} p_{20}", real=True)

q, q2 = sp.symbols("q q_2", real=True)

sj1, sj2, u, v = sp.symbols("s s_2 u v", positive=True)

s01, s02 = 1-sj1, 1-sj2

Vj, V0 = sp.symbols("V_{2j} V_{20}", real=True)

h0, h1 = sp.symbols("h_0 h_1")

p1_star_sym, p2_star_sym = sp.symbols("p_1^* p_2^*")

z1, z2 = q, q2

def signed_state(state):
    if state == "j":
        return -1
    if state == "0":
        return 1
    raise ValueError("state must be 'j' or '0'")

def raw_utilities(price, signed_history):
    return (beta + alpha*price + gamma*(1+signed_history)/2,
            gamma*(1-signed_history)/2)

raw_inside1, raw_outside1 = raw_utilities(p1, q)

raw_inside2, raw_outside2 = raw_utilities(p2, q2)

U1 = sp.expand(raw_inside1 - raw_outside1)

U2 = sp.expand(raw_inside2 - raw_outside2)

sj1_expr = 1/(1+sp.exp(-U1))

sj2_expr = 1/(1+sp.exp(-U2))

s01_expr, s02_expr = 1-sj1_expr, 1-sj2_expr

dsj1_dp1, dsj2_dp2 = alpha*sj1*s01, alpha*sj2*s02

ds01_dp1, ds02_dp2 = -dsj1_dp1, -dsj2_dp2

dsj1_dgamma = q*sj1*s01

dsj2_dgamma_direct = q2*sj2*s02

dsj2_dutility = sj2*s02



# === 2. Profit, first-order conditions, and prices === #



pi1 = sj1*(p1-c)

profit2 = sj2*(p2-c)

pi2 = sj1*Vj + (1-sj1)*V0

Pi = pi1 + delta*pi2

DeltaV = Vj-V0

foc1_raw = sj1 + alpha*sj1*s01*(p1-c) + delta*alpha*sj1*s01*DeltaV

foc1_collected = sj1 + alpha*sj1*s01*(p1-c+delta*DeltaV)

foc1_eq = sp.Eq(foc1_collected, 0)

foc2 = sj2 + alpha*sj2*s02*(p2-c)

foc2_eq = sp.Eq(foc2, 0)

p1_star = c-1/(alpha*s01)-delta*(Vj-V0)

p1_star_eq = sp.Eq(p1_star_sym, p1_star)

p2_star = c-1/(alpha*s02)

p2_star_eq = sp.Eq(p2_star_sym, p2_star)

assumptions = {"alpha": "<0", "delta": "0<delta<1", "prices": "unconstrained real",
               "cost": "fixed", "shares": "strictly interior", "type": "fixed and observed",
               "history": "realized j or 0; both continuation states integrated"}



# === 3. Implicit derivatives and the envelope theorem === #



# Differentiate the terminal pricing FOC while allowing the equilibrium purchase
# probability to change.
def price2_derivative(theta, share=sj2, history=q2, price=p2):
    """Implicit derivative of the terminal optimal price with respect to theta."""
    # Only one of these direct effects is nonzero for a given parameter theta.
    intercept_effect = sp.diff(beta, theta)
    price_coefficient_effect = price * sp.diff(alpha, theta)
    history_effect = history * sp.diff(gamma, theta)
    direct_utility_effect = intercept_effect + price_coefficient_effect + history_effect

    cost_effect = (1 - share) * sp.diff(c, theta)
    alpha_effect = sp.diff(alpha, theta) / alpha**2
    demand_effect = -share * direct_utility_effect / alpha
    return sp.simplify(cost_effect + alpha_effect + demand_effect)

def value2_derivative(theta, share, history, price):
    """Terminal value derivative from the envelope theorem."""
    direct_utility_effect = (sp.diff(beta, theta)
                             + price * sp.diff(alpha, theta)
                             + history * sp.diff(gamma, theta))
    demand_effect = -share * direct_utility_effect / alpha
    cost_effect = -share * sp.diff(c, theta)
    return demand_effect + cost_effect

# Differentiate the initial pricing FOC, including how the terminal values change.
# Histories stay fixed when taking a derivative.
def price1_derivative(theta):
    """Implicit derivative of the initial price, including both future branches."""
    direct_utility_effect = (sp.diff(beta, theta)
                             + p1 * sp.diff(alpha, theta)
                             + q * sp.diff(gamma, theta))
    inside_value_effect = value2_derivative(theta, u, -1, p2j)
    outside_value_effect = value2_derivative(theta, v, 1, p20)
    value_difference_effect = inside_value_effect - outside_value_effect

    cost_effect = s01 * sp.diff(c, theta)
    alpha_effect = sp.diff(alpha, theta) / alpha**2
    current_demand_effect = -sj1 * direct_utility_effect / alpha
    continuation_effect = -s01 * (sp.diff(delta, theta) * (Vj - V0)
                                  + delta * value_difference_effect)
    return sp.simplify(cost_effect + alpha_effect + current_demand_effect
                       + continuation_effect)

# Apply the chain rule: prices, shares, and terminal values all change with the parameter.
def total_derivative(expr, theta):
    """Apply the chain rule to an expression evaluated at optimal prices."""
    initial_price_effect = price1_derivative(theta)
    terminal_inside_price_effect = price2_derivative(theta, u, -1, p2j)
    terminal_outside_price_effect = price2_derivative(theta, v, 1, p20)

    # logit share derivative = share*(1-share)*utility derivative.
    initial_share_effect = sj1 * s01 * (
        sp.diff(beta, theta) + p1 * sp.diff(alpha, theta)
        + q * sp.diff(gamma, theta) + alpha * initial_price_effect)
    terminal_inside_share_effect = u * (1 - u) * (
        sp.diff(beta, theta) + p2j * sp.diff(alpha, theta)
        - sp.diff(gamma, theta) + alpha * terminal_inside_price_effect)
    terminal_outside_share_effect = v * (1 - v) * (
        sp.diff(beta, theta) + p20 * sp.diff(alpha, theta)
        + sp.diff(gamma, theta) + alpha * terminal_outside_price_effect)

    # Each entry says how an endogenous variable changes when theta changes.
    changes = {
        p1: initial_price_effect,
        p2j: terminal_inside_price_effect,
        p20: terminal_outside_price_effect,
        sj1: initial_share_effect,
        u: terminal_inside_share_effect,
        v: terminal_outside_share_effect,
        Vj: value2_derivative(theta, u, -1, p2j),
        V0: value2_derivative(theta, v, 1, p20),
    }
    derivative = sp.diff(expr, theta)
    for variable, variable_change in changes.items():
        derivative += sp.diff(expr, variable) * variable_change
    return sp.factor(derivative)

dp1_dgamma = price1_derivative(gamma)

dp2_dgamma = price2_derivative(gamma)

dVj_dgamma, dV0_dgamma = value2_derivative(gamma,u,-1,p2j), value2_derivative(gamma,v,1,p20)

ds_dgamma = sp.factor(sj1*s01*(q+alpha*dp1_dgamma))

du_dgamma, dv_dgamma = -u*(1-u)**2, v*(1-v)**2

d2p1_dgamma2 = total_derivative(dp1_dgamma,gamma)

def terminal_total_derivative(expr, theta):
    """The same chain rule, restricted to a single terminal history."""
    price_effect = price2_derivative(theta)
    utility_effect = (sp.diff(beta, theta) + p2 * sp.diff(alpha, theta)
                      + q2 * sp.diff(gamma, theta) + alpha * price_effect)
    share_effect = sj2 * s02 * utility_effect
    derivative = (sp.diff(expr, theta)
                  + sp.diff(expr, p2) * price_effect
                  + sp.diff(expr, sj2) * share_effect)
    return sp.factor(derivative)

def profit_derivative(theta):
    """Envelope derivative of optimized profit, including both terminal values."""
    direct_utility_effect = (sp.diff(beta, theta) + p1 * sp.diff(alpha, theta)
                             + q * sp.diff(gamma, theta))
    current_demand_effect = direct_utility_effect * sj1 * s01 * (p1 - c + delta * (Vj - V0))
    current_cost_effect = -sj1 * sp.diff(c, theta)
    discount_effect = sp.diff(delta, theta) * (sj1 * Vj + s01 * V0)
    future_value_effect = delta * (
        sj1 * value2_derivative(theta, u, -1, p2j)
        + s01 * value2_derivative(theta, v, 1, p20))
    derivative = current_demand_effect + current_cost_effect + discount_effect + future_value_effect
    # Substitute the initial FOC to simplify the envelope expression.
    optimal_initial_price = c - 1 / (alpha * s01) - delta * (Vj - V0)
    return sp.simplify(derivative.subs(p1, optimal_initial_price))

d2p2_dgamma2 = terminal_total_derivative(dp2_dgamma,gamma)

ds2_dgamma = sp.factor(sj2*s02*(q2+alpha*dp2_dgamma))

dp2dbeta_dgamma = terminal_total_derivative(price2_derivative(beta),gamma)

dPi_dgamma = profit_derivative(gamma)

dPi_dbeta = profit_derivative(beta)

ratio_derivative = (p2*dp1_dgamma-p1*dp2_dgamma)/p2**2

matched_j = ratio_derivative.subs({q:-1,p2:p2j,sj2:u,q2:-1})

matched_0 = ratio_derivative.subs({q:1,p2:p20,sj2:v,q2:1})

induce_harvest = ratio_derivative.subs({q:1,p2:p2j,sj2:u,q2:-1})





# === 4. Display equations and optionally write the mathematical appendix === #

def display_equation(lhs, expression):
    """Keep the mathematical right-hand side sourced directly from SymPy."""
    return '\\[\n' + lhs + ' = ' + sp.latex(expression) + '\n\\]\n'

def show(title, expr):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    sp.pprint(expr)

def make_latex_appendix(output, parameters, gammas, evaluate_model):
    TEX_OUTPUT = Path(output)
    if parameters != dict(beta=2.0, alpha=-1.0, delta=0.92, c=0.0) or not np.array_equal(gammas, np.linspace(-3, 3, 601)):
        raise ValueError("The written appendix describes the original benchmark. Set make_appendix=False for custom calibrations.")
    TEX_OUTPUT.mkdir(parents=True, exist_ok=True)
    equations='\\section{Symbolically derived equations}\n'
    equations += r'Here $q_2=q(x_1)$, $s_2=s_2(x_1)$ and $p_2=p_2(x_1)$ describe a fixed terminal history. Price equations and comparative-static right-hand sides use equilibrium shares; the profit and FOCs use shares at the candidate prices.'+'\n'
    for lhs,expression in [(r'v_1',U1),(r'v_2',U2),(r'\Pi',Pi),
                           (r'F_1',foc1_collected),(r'F_2',foc2),
                           (r'p_1^*',p1_star),(r'p_2^*',p2_star),
                           (r'\frac{dp_1^*}{d\gamma}',dp1_dgamma),
                           (r'\frac{dp_2^*}{d\gamma}',dp2_dgamma),
                           (r'\frac{d\Pi^*}{d\gamma}',dPi_dgamma)]:
        equations+=display_equation(lhs,expression)
    (TEX_OUTPUT/'sympy_equations.tex').write_text(equations)
    document=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,graphicx,booktabs,array}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newcommand{\dd}{\mathrm{d}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{Two-period pricing with symmetric switching preferences}
\author{}\date{October 30, 2026}
\begin{document}\maketitle
\section{Model, histories, and normalization}
One firm sells one inside product $j$ against outside option $0$ in periods
$t=1,2$. Type $r$ is fixed and observed by the firm, which prices separately for
that type. All formulas below are conditional on $r$; its subscript and the
fixed product index are suppressed. There is no pooling of heterogeneous types.
The initial realized choice is $x_0\in\{j,0\}$. The second-period price can depend
on the realized first-period choice $x_1$. Independent Type-I extreme-value
shocks enter both alternatives in each period.

Before normalization the deterministic utilities are
\[
 V_{jt}^{\rm raw}=\beta+\alpha p_t+\gamma\mathbf1\{x_{t-1}=0\},
 \qquad V_{0t}^{\rm raw}=\gamma\mathbf1\{x_{t-1}=j\}.
\]
The payoff to switching applies in either direction. Subtracting the raw outside
utility from both alternatives gives normalized outside utility zero and
\[
 v_t(h)=\beta+\alpha p_t(h)+q(h)\gamma,\qquad
 q(j)=-1,\quad q(0)=+1.
\]
Normalization retains the outside switching payoff through the signed inside
utility difference; it does not delete that payoff. Negative $\gamma$ denotes
switching aversion and positive $\gamma$ denotes novelty preference.
\[
 s_t(h)=\frac{e^{v_t(h)}}{1+e^{v_t(h)}},\quad s_{0t}(h)=1-s_t(h),\quad
 \partial_p s_t=\alpha s_t(1-s_t),\quad
 (\partial_\gamma s_t)_p=q(h)s_t(1-s_t).
\]
A realized state $x_t$, a hypothetical switching indicator, and the probability
$s_t$ are distinct objects. Every derivative below holds the discrete history fixed.

\textbf{Assumptions.} $\alpha<0$, $0<\delta<1$, fixed marginal cost $c$, finite
parameters, and interior binary-logit shares. Prices are unrestricted real
numbers: introductory subsidies are allowed. Continuation values are conditional
on realized states and have no direct dependence on $p_1$. Type does not evolve.
The case $\delta=0$ is used only as a static counterfactual.

\begin{center}\begin{tabular}{ll}\toprule
Notation & Meaning\\\midrule
$q=q(x_0)$ & Signed initial-state coefficient\\
$s=s_1^*(x_0),\quad r_0=1-s$ & First-period equilibrium shares\\
$u=s_2^*(j),\quad v=s_2^*(0)$ & Inside shares in the two continuation states\\
$V_j=V_2(j),\quad V_0=V_2(0)$ & Optimized continuation profits\\
$\Delta V=V_j-V_0,\quad B=u+v$ & Continuation difference and sum of terminal shares\\
$\Pi^*$ & Expected discounted profit per initial consumer\\\bottomrule
\end{tabular}\end{center}

\section{Backward induction and optimality}
For each realized first-period choice $a\in\{j,0\}$, solve
\[
 V_2(a)=\max_{p_2(a)}s_2(a)[p_2(a)-c].
\]
Its first-order condition is
\[
 0=s_2+\alpha s_2(1-s_2)(p_2-c),\qquad
 p_2^*(a)=c-\frac1{\alpha[1-s_2^*(a)]}.
\]
This is an implicit equation: the share depends on the same price being solved.
Both $V_j$ and $V_0$ must be computed before period 1.

First-period profit integrates over both possible realized histories:
\begin{align*}
 \Pi(p_1;x_0)&=s_1(p_1-c)+\delta[s_1V_j+(1-s_1)V_0]\\
 &=\delta V_0+s_1[p_1-c+\delta\Delta V].
\end{align*}
Because the branch values do not depend directly on $p_1$,
\[
 \partial_{p_1}\{\delta[s_1V_j+(1-s_1)V_0]\}
 =\delta\alpha s_1(1-s_1)(V_j-V_0).
\]
Consequently,
\[
 0=s_1+\alpha s_1(1-s_1)[p_1-c+\delta\Delta V],\qquad
 p_1^*=c-\frac1{\alpha(1-s)}-\delta\Delta V.
\]
The static FOC component $c-1/[\alpha(1-s)]$ uses the share at the dynamic
optimum. It is not the counterfactual static optimum, which must be solved
separately with $\delta=0$.

\begin{proposition}[Unique optimal prices]
Under the stated assumptions each continuation price and the first-period price
has a unique finite global optimum, with no lower bound imposed on $p_1$.
\end{proposition}
\begin{proof}
Both price problems reduce, up to a constant, to maximizing
$s(p)(p-k)$ with $s(p)=\operatorname{logit}^{-1}(b+\alpha p)$.
Use $k=c$ in period 2 and $k=c-\delta\Delta V$ in period 1.
The sign of the profit derivative is the sign of
\[
 H(p)=1+e^{b+\alpha p}+\alpha(p-k),\quad
 H'(p)=\alpha(1+e^{b+\alpha p})<0.
\]
$H$ ranges from positive infinity to negative infinity, so its root is unique
and the objective increases before it and decreases afterward. The price may
lie below physical cost $c$ in period 1 when continuation incentives justify it.
\end{proof}

\section{Envelope derivatives and conditional comparative statics}
Differentiating each continuation value using the envelope theorem gives
\[
 V_2'(a)=q(a)s_2(1-s_2)(p_2^*-c)=-\frac{q(a)s_2}{\alpha}.
\]
Implicit differentiation of the terminal FOC gives the same expression for the
optimal price derivative:
\[
 p_2^{*\prime}(a)=-\frac{q(a)s_2}{\alpha},\quad
 V_j'=u/\alpha,\quad V_0'=-v/\alpha,\quad
 \Delta V'=(u+v)/\alpha.
\]
In particular, $p_2^{*\prime}(j)<0$ and $p_2^{*\prime}(0)>0$.
The total equilibrium terminal-share derivative is
$s_2'=q(a)s_2(1-s_2)^2$, including the optimal-price response.

For the first-period implicit pricing equation let
$F=p_1-c+1/[\alpha(1-s_1)]+\delta\Delta V$. At fixed history,
\[
 F_{p_1}=\frac1{r_0},\qquad
 F_\gamma=\frac{qs}{\alpha r_0}+\frac{\delta B}{\alpha},\qquad
 \boxed{p_1^{*\prime}=-\frac{qs+\delta r_0 B}{\alpha}}.
\]
The share therefore changes according to
\[
 s'=sr_0(q+\alpha p_1^{*\prime})=sr_0^2(q-\delta B).
\]
The first-period envelope theorem retains both continuation branches:
\begin{align*}
 \Pi^{*\prime}
 &=qsr_0[p_1^*-c+\delta\Delta V]+\delta[sV_j'+r_0V_0']\\
 &=\boxed{\frac{-qs+\delta(su-r_0v)}{\alpha}}.
\end{align*}
The term $\delta V_0'$ must not be dropped. The envelope partial derivative holds
the optimizing price fixed inside the partial derivative; it does not hold the
continuation value fixed as $\gamma$ varies.

For the analytical second price derivative define
\[
 u'=-u(1-u)^2,\quad v'=v(1-v)^2,\quad B'=u'+v'.
\]
Differentiating the first derivative and substituting $s'$ gives
\[
 \boxed{p_1^{*\prime\prime}
 =-\frac{sr_0^2(q-\delta B)^2+\delta r_0B'}{\alpha}}.
\]
This expression is used analytically in checks, rather than obtained from noisy
twice-differenced optimized prices. No second-derivative figure is included.

\begin{proposition}[Conditional signs]
For an initially outside consumer, $p_1^{*\prime}>0$ and $\Pi^{*\prime}>0$.
For an initially inside consumer,
\[
 p_1^{*\prime}>0\ \Longleftrightarrow\ \delta r_0(u+v)>s,\qquad
 \Pi^{*\prime}>0\ \Longleftrightarrow\ \delta r_0v>s(1+\delta u).
\]
Equalities give zero derivatives; reversed inequalities give negative derivatives.
\end{proposition}
\begin{proof}
For $q=1$, the price numerator $s+\delta r_0B$ is positive and $-1/\alpha>0$.
The profit numerator is $-s(1-\delta u)-\delta r_0v<0$ because $\delta u<1$;
dividing by $\alpha<0$ makes the derivative positive. Substituting $q=-1$
gives the two stated thresholds. These statements do not establish a common
monotonicity sign for both histories or a global sign for the price ratios.
\end{proof}

\section{Price ratios: three distinct conditioning statements}
The state-matched ratios are descriptive comparisons:
\[
 R_j=\frac{p_1^*(x_0=j)}{p_2^*(x_1=j)},\qquad
 R_0=\frac{p_1^*(x_0=0)}{p_2^*(x_1=0)}.
\]
They do not necessarily describe one realized consumer path. The acquisition
comparison is instead
\[
 R_{0\to j}=\frac{p_1^*(x_0=0)}{p_2^*(x_1=j)},
\]
conditional on acquisition in period 1. It is not an expectation over histories.
For each definition use its own numerator, denominator and state derivatives:
\[
 \boxed{R'=\frac{p_2^*p_1^{*\prime}-p_1^*p_2^{*\prime}}{(p_2^*)^2}}.
\]
Terminal prices exceed $c$; the benchmark $c=0$ therefore has strictly positive
ratio denominators. For other calibrations a ratio is only defined when its
denominator is nonzero. No expected ratio or ratio of expectations is substituted.

\section{Numerical benchmark and verification}
Every plotted point reoptimizes both terminal branches and period 1:
\[
 \beta=2,\qquad\alpha=-1,\qquad\delta=0.92,\qquad c=0,\qquad
 \gamma\in[-3,3]\ \text{on a 601-point grid}.
\]
The calibration is illustrative, not estimated. SymPy verifies raw-utility
normalization, both branch weights, the profit-derived FOCs, implicit price
derivatives and the envelope identities. Numerical checks compare all 1,202
benchmark state/grid solutions with the handoff's independent reference.
Central differences reoptimize the full model at nearby parameters; an
independent scalar profit optimizer checks optimality. Additional calibrations
check derivatives and an introductory-subsidy case. Numerical evidence is not
a proof of unqualified global comparative-static signs.

\subsection*{Selected optimized values}
\begin{center}\small\begin{tabular}{crrrrrr}\toprule
$x_0$ & $\gamma$ & $p_1^*$ & $p_2^*(j)$ & $p_2^*(0)$ & $\Pi^*$ & $\Pi^{*\prime}$\\\midrule
'''
    for state in ['j','0']:
        for g in [-3,-1,0,1,3]:
            r=evaluate_model(g,state)
            document+=f"${state}$ & {g} & {r['p1']:.3f} & {r['p2j']:.3f} & {r['p20']:.3f} & {r['profit']:.3f} & {r['dprofit']:.3f}"+r'\\'+'\n'
    document+=r'''\bottomrule\end{tabular}\end{center}
\clearpage
\input{sympy_equations.tex}
'''
    specs=[
        ('matched_price_ratio_derivative','State-matched price-ratio derivatives',
         r'$R_j=p_1^*(j)/p_2^*(j)$ and $R_0=p_1^*(0)/p_2^*(0)$. Each curve holds its stated histories fixed and uses the analytical quotient rule.'),
        ('induce_harvest_price_ratio_derivative','Induce-then-harvest price-ratio derivative',
         r'$R_{0\to j}=p_1^*(0)/p_2^*(j)$, conditional on a first-period acquisition. This is a path-specific comparison, not an ex-ante expectation.'),
        ('profit_by_gamma','Optimized profit levels',
         r'Expected discounted profit includes both realized continuation histories. The two curves condition on the initial realized state.'),
        ('gamma_profit_by_gamma','Optimized profit derivative',
         r'Total derivatives use the envelope theorem and both continuation-value derivatives. Their signs depend on the initial history and the stated thresholds.'),
        ('optimal_prices_by_gamma','Optimal prices in both periods',
         r'Left: period-1 prices for both initial states. Right: period-2 prices for both realized first-period states. Prices are unconstrained; all shares adjust.'),
        ('gamma_p1_by_gamma','Optimal period-1 price derivative',
         r'Analytical total derivatives, with both continuation branches reoptimized. Initial-state signs are interpreted using the proved conditional statements.'),
    ]
    for name,title,caption in specs:
        if not (TEX_OUTPUT/'figures'/f'{name}.pdf').exists():
            raise FileNotFoundError(name)
        document+='\\clearpage\n\\section*{'+title+'}\n'
        document+='\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\linewidth]{figures/'+name+'.pdf}\n'
        document+='\\caption{'+caption+r' Benchmark: $\beta=2$, $\alpha=-1$, $\delta=0.92$, $c=0$; $\gamma\in[-3,3]$.}'+'\n\\end{figure}\n'
    document+='\\end{document}\n'
    (TEX_OUTPUT/'comparative_statics_figures.tex').write_text(document)
    print('Generated verified symbolic equations and appendix with six requested figures.')

# === 5. Print the main symbolic results === #

if __name__ == "__main__":
    show("Inside utility in period 1", U1)
    show("Inside utility in period 2", U2)
    show("Expected discounted profit", Pi)
    show("Period 1 first-order condition", foc1_collected)
    show("Terminal first-order condition", foc2)
    show("Period 1 optimal price", p1_star)
    show("Terminal optimal price", p2_star)
    show("dp1/dgamma", dp1_dgamma)
    show("d2p1/dgamma2", d2p1_dgamma2)
    show("dprofit/dgamma", dPi_dgamma)
    show("Price-ratio derivative", ratio_derivative)
