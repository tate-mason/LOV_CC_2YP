"""Create a minimal LaTeX document containing every figure and SymPy formula.

Run after make_comparative_static_figures.py, then compile using:
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error writeup_math_graphs/comparative_statics_figures.tex
"""
from pathlib import Path
import sympy as sp

from export.latex_export import display_equation
from make_comparative_static_figures import DERIVATIVES
from model.symbols import p1, p2, sj1, sj2, s01, s02, delta
from model.utility import U1, U2
from model.shares import sj1_expr, sj2_expr
from model.profit import Pi
from pricing.period1_foc import foc1_collected
from pricing.period2_foc import foc2
from pricing.period1_price import p1_star
from pricing.period2_price import p2_star
from comparative_statics.gamma_p1p2 import ratio_derivative

ROOT = Path(__file__).resolve().parent / 'writeup_math_graphs'
S1, S2 = sp.symbols('s_1 s_2')
PRESENTATION_SYMBOLS = {sj1: S1, sj2: S2, s01: 1-S1, s02: 1-S2}


def economics_expression(expression):
    """Suppress the fixed product index and write outside shares as 1-s_t."""
    return sp.sympify(expression).xreplace(PRESENTATION_SYMBOLS)


def equation(lhs, expression):
    return display_equation(lhs, economics_expression(expression))


def main():
    ROOT.mkdir(exist_ok=True)
    # This file is regenerated from the symbolic model, never from copied TeX.
    model_equations = [
        (r'v_1', U1), (r'v_2', U2),
        (r's_1', sj1_expr), (r's_2', sj2_expr),
        (r'\Pi', Pi),
    ]
    equations = '\\section*{Model and pricing conditions}\n'
    equations += ''.join(equation(*item) for item in model_equations)
    equations += '\\subsection*{First-order conditions for both prices}\n'
    equations += equation(r'F_1 = \frac{\partial\Pi}{\partial p_1}', foc1_collected)
    equations += equation(r'\frac{\partial\Pi}{\partial p_2} = \delta s_1 F_2', delta*sj1*foc2)
    equations += equation(r'F_2', foc2)
    equations += r'\[F_1=F_2=0\]' + '\n'
    equations += equation(r'0 = \frac{F_1}{s_1}', sp.simplify(foc1_collected/sj1))
    equations += equation(r'0 = \frac{F_2}{s_2}', sp.simplify(foc2/sj2))
    equations += equation(r'p_2^*', p2_star)
    equations += equation(r'p_1^*', p1_star)
    (ROOT/'sympy_equations.tex').write_text(equations)

    pages = [
        ('profit_by_gamma', 'Optimized expected profit', r'\Pi^*', Pi),
        ('price_ratio_by_gamma', 'Optimal price ratio', r'R^*', p1/p2),
        ('price_ratio_derivative_cases', 'Price-ratio derivative in gamma',
         r'\frac{\mathrm{d}R^*}{\mathrm{d}\gamma}', ratio_derivative),
        ('p1_by_gamma', 'Optimal period-1 price', r'p_1^*', p1_star),
        ('p2_by_gamma', 'Optimal period-2 price', r'p_2^*', p2_star),
    ]
    for key,(expr,label,name) in DERIVATIVES.items():
        # Matplotlib labels already contain valid LaTeX math delimiters.
        pages.append((name, label, label.strip('$'), expr))
    pages.extend([
        ('period1_foc', 'Period-1 pricing condition', r'F_1', foc1_collected),
        ('period2_foc', 'Period-2 pricing condition', r'F_2', foc2),
    ])
    missing = [name for name,_,_,_ in pages if not (ROOT/'figures'/f'{name}.pdf').exists()]
    if missing:
        raise FileNotFoundError('Run make_comparative_static_figures.py first; missing: '+', '.join(missing))
    document = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
\begin{center}\Large Pricing conditions and comparative statics\end{center}
\section*{Notation and definitions}
We study one product and suppress its fixed index $j$: $p_{jt}$ becomes $p_t$
and $s_{jt}$ becomes $s_t$. The outside-option share $s_{0t}$ is
written as $1-s_t$. $v_t$ is systematic component of utility.

\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}p{0.19\linewidth}p{0.76\linewidth}@{}}
\hline
Symbol & Definition \\
\hline
$t\in\{1,2\}$ & Time period. \\
$j$ & Fixed product index, suppressed in displayed formulas. \\
$p_t$, $p_t^*$ & Price and optimal price in period $t$, respectively. \\
$c\geq0$ & Constant marginal cost, common to both periods. \\
$v_t$ & Systematic utility from purchasing in period $t$; outside utility is zero. \\
$s_t\in(0,1)$ & Logit purchase probability. In period 2 this is conditional on a period-1 purchase. \\
$1-s_t$ & Probability of choosing the outside option in period $t$. \\
$\alpha<0$ & Price coefficient in utility; $-\alpha>0$ is price sensitivity. \\
$\beta$ & Baseline utility or demand intercept. \\
$\gamma$ & Utility coefficient on the history indicator, called switching preference in the figures. \\
$z_t\in\{0,1\}$ & Fixed, observed history indicator that shifts period-$t$ utility by $\gamma z_t$. \\
$\delta\in(0,1)$ & Discount factor on period-2 profit. \\
$\Pi$, $\Pi^*$ & Expected discounted profit per initial customer, and its optimized value. \\
$F_1$, $F_2$ & Period-1 profit derivative and normalized period-2 profit derivative; their roots give the pricing FOCs. \\
$R^*$ & Optimal price ratio $p_1^*/p_2^*$. \\
$\partial$, $\mathrm d$ & Partial derivatives hold the other price fixed; total derivatives allow both optimal prices and shares to adjust. \\
\hline
\end{tabular}

An asterisk denotes evaluation at the optimum. In all derivative formulas,
unstarred prices and shares on the right-hand side are evaluated at that optimum.

\[
\alpha=-2,\quad \beta=-1,\quad \delta=0.7,\quad c=1,\quad
(z_1,z_2)\in\{(0,0),(1,0),(0,1)\},\quad \gamma\in[0,3].
\]
\clearpage
\input{sympy_equations.tex}
'''
    for name,title,lhs,expr in pages:
        document += '\\clearpage\n\\section*{'+title+'}\n'
        if name == 'beta_p1_derivative_by_gamma':
            document += ('\\[\\resizebox{\\linewidth}{!}{$\\displaystyle '
                         +lhs+' = '+sp.latex(economics_expression(expr))+'$}\\]\n')
        else:
            document += equation(lhs,expr)
        document += ('\\begin{figure}[htbp]\n\\centering\n'
                     '\\includegraphics[width=\\linewidth]{figures/'+name+'.pdf}\n'
                     '\\end{figure}\n')
    document += '\\end{document}\n'
    (ROOT/'comparative_statics_figures.tex').write_text(document)
    print(f'Exported SymPy equations and LaTeX document with {len(pages)} figures.')


if __name__ == '__main__':
    main()
