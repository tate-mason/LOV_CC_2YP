"""Advisor packet from saved theoretical/aggregate pricing results only."""
from pathlib import Path
from statistics import NormalDist
import sys
import json
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'advisor-pricing-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demand'))
from binary_welfare import evaluate_policy
from quintile_theory import solve_distribution


def plain_case(root):
    source=root/'separate_experiments_50281568/novelty_only/point_estimates.csv'
    p=pd.read_csv(source).set_index('parameter').estimate
    beta=float(p.inside_constant+p.beta_plain)
    alpha=-float(np.exp(p.mu_log_minus_alpha+.5*p.sd_log_minus_alpha**2))
    x,w=np.polynomial.legendre.leggauss(21)
    probability=.05+.9*((np.arange(5)[:,None]+(x+1)/2)/5).ravel()
    gamma=np.array([NormalDist(p.mean_novelty,p.sd_novelty).inv_cdf(float(u)) for u in probability])
    mass=np.tile(w/10,5);bins=np.repeat(np.arange(1,6),21)
    policies=[];summary=[];diagnostics=[]
    for h0 in ['j','0']:
        uniform=solve_distribution(gamma,mass,h0,alpha,beta,1.)
        diagnostics.append(dict(h0=h0,regime='Uniform',quintile='population',**uniform))
        for k in range(1,6):
            selected=bins==k;conditional=mass[selected]/.2
            informed=solve_distribution(gamma[selected],conditional,h0,alpha,beta,1.)
            diagnostics.append(dict(h0=h0,regime='Informed',quintile=str(k),**informed))
            for regime,policy in [('Uniform',uniform),('Informed',informed)]:
                outcomes=evaluate_policy(gamma[selected],h0,alpha,beta,1.,
                    policy['p1'],policy['p2j'],policy['p20'])
                policies.append(dict(h0=h0,regime=regime,quintile=k,flavor='plain',
                    p1=policy['p1'],p2_after_inside=policy['p2j'],p2_after_outside=policy['p20']))
                summary.append(dict(h0=h0,regime=regime,quintile=k,
                    **{metric:float(conditional@outcomes[metric])
                       for metric in ['consumer_surplus','revenue','total_welfare']}))
    folder=root/'trimmed_flavor_gamma/no_subsidies/plain_trimmed_05_95'
    folder.mkdir(exist_ok=True)
    policies=pd.DataFrame(policies);summary=pd.DataFrame(summary);diagnostics=pd.DataFrame(diagnostics)
    assert (policies[['p1','p2_after_inside','p2_after_outside']]>=0).all().all()
    assert diagnostics.kkt_residual.max()<1e-6
    np.testing.assert_allclose(diagnostics.objective,diagnostics.reported_revenue,atol=1e-10)
    policies.to_csv(folder/'price_policies.csv',index=False)
    summary.to_csv(folder/'quintile_outcomes.csv',index=False)
    diagnostics.to_csv(folder/'solver_diagnostics.csv',index=False)
    (folder/'settings.json').write_text(json.dumps(dict(source=str(source),
        specification='plain-specific trimmed flavor switching',alpha=alpha,beta_plain=beta,
        gamma_mean=float(p.mean_novelty),gamma_original_sd=float(p.sd_novelty),
        retained_percentiles=[5,95],price_floor=0.,learning=False,
        inside='plain yogurt',outside='not plain: berry, other yogurt, or no yogurt',
        pooled_yogurt_inclusive_value=False,demand_refitted=False),indent=2)+'\n')
    return policies,summary


def plot_prices(data, destination):
    fig,axes=plt.subplots(2,2,figsize=(10,6.1),sharey=True,layout='constrained')
    decisions=[('j','p1','Period 1: initially inside'),('0','p1','Period 1: initially outside'),
               ('j','p2_after_inside','Period 2: after inside purchase'),
               ('j','p2_after_outside','Period 2: after outside choice')]
    for ax,(h0,key,title) in zip(axes.ravel(),decisions):
        for regime,offset,color,marker in [('Uniform',-.08,'#444444','D'),('Informed',.08,'#2E6896','o')]:
            part=data[(data.h0==h0)&(data.regime==regime)].sort_values('quintile')
            assert len(part)==5
            ax.scatter(part.quintile+offset,part[key],s=42,color=color,marker=marker,label=regime,zorder=3)
        ax.set(title=title,xticks=range(1,6),xticklabels=[f'Q{k}' for k in range(1,6)],
               xlabel='Gamma quintile',ylabel='Optimized price ($)')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,frameon=False)
        ax.axhline(0,color='black',linewidth=.5,zorder=0)
    maximum=float(data[['p1','p2_after_inside','p2_after_outside']].max().max())
    axes[0,0].set_ylim(0,max(maximum*1.08,.1))
    fig.savefig(destination,bbox_inches='tight');plt.close(fig)


def plot_welfare(data, destination):
    fig,axes=plt.subplots(2,3,figsize=(11,6.3),layout='constrained')
    for row_axes,h0 in zip(axes,['j','0']):
        for ax,(metric,label) in zip(row_axes,[('consumer_surplus','CS'),
                ('revenue','Revenue'),('total_welfare','TW')]):
            values=[]
            for k in range(1,6):
                part=data[(data.h0==h0)&(data.quintile==k)].set_index('regime')
                baseline=float(part.loc['Uniform',metric])
                assert baseline != 0
                values.append(100*(float(part.loc['Informed',metric])-baseline)/abs(baseline))
            ax.axhline(0,color='#444444',linewidth=.8,label='Uniform')
            ax.bar(range(1,6),values,color='#2E6896',label='Informed')
            ax.set(title=label+' / initially '+('inside' if h0=='j' else 'outside'),
                   xticks=range(1,6),xticklabels=[f'Q{k}' for k in range(1,6)],
                   xlabel='Gamma quintile',ylabel='Change / |Uniform baseline| (%)')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=8,frameon=False)
    fig.savefig(destination,bbox_inches='tight');plt.close(fig)


def build(root):
    root=Path(root)
    output=root/'trimmed_flavor_gamma'
    theory=output/'no_subsidies/01_pure_theory'
    assets=output/'advisor_assets';assets.mkdir(exist_ok=True)
    policy,outcomes=plain_case(root)

    # Theory nodes are numerical integration support, not household records.
    nodes=pd.read_csv(theory/'type_results.csv')
    metrics=['p1','p2j','p20']
    weighted=nodes.copy();weighted[metrics]=weighted[metrics].mul(weighted.weight,axis=0)
    grouped=weighted.groupby(['h0','regime','group'])[metrics+['weight']].sum()
    grouped[metrics]=grouped[metrics].div(grouped.weight,axis=0)
    data=grouped.reset_index()
    data['quintile']=data.group.str.split('_').str[-1].astype(int)
    data['regime']=data.regime.map({'common':'Uniform','informed':'Informed'})
    data=data.rename(columns={'p2j':'p2_after_inside','p20':'p2_after_outside'})
    plot_prices(data,assets/'theory_prices.pdf')
    welfare_metrics=['consumer_surplus','revenue','total_welfare']
    welfare=nodes.copy()
    welfare[welfare_metrics]=welfare[welfare_metrics].mul(welfare.weight,axis=0)
    welfare=welfare.groupby(['h0','regime','group'])[welfare_metrics+['weight']].sum()
    welfare[welfare_metrics]=welfare[welfare_metrics].div(welfare.weight,axis=0)
    welfare=welfare.reset_index()
    welfare['quintile']=welfare.group.str.split('_').str[-1].astype(int)
    welfare['regime']=welfare.regime.map({'common':'Uniform','informed':'Informed'})
    plot_welfare(welfare,assets/'theory_welfare.pdf')
    plot_prices(policy,assets/'plain_prices.pdf')
    plot_welfare(outcomes,assets/'plain_welfare.pdf')
    selection=[('0',1,'consumer_surplus','CS'),('0',1,'total_welfare','TW'),
               ('j',1,'consumer_surplus','CS'),('j',1,'total_welfare','TW'),
               ('0',5,'consumer_surplus','CS'),('0',5,'total_welfare','TW')]
    rows=[];export=[]
    for h0,k,metric,label in selection:
        part=outcomes[(outcomes.h0==h0)&(outcomes.quintile==k)].set_index('regime')
        u,i=float(part.loc['Uniform',metric]),float(part.loc['Informed',metric])
        delta=i-u;pct=100*delta/abs(u)
        state='Outside' if h0=='0' else 'Inside'
        rows.append(f'{state} & Q{k} & {label} & {u:.5f} & {i:.5f} & {delta:+.5f} & {pct:+,.1f}'+r'\% \\')
        export.append(dict(initial_state=state,quintile=k,metric=label,uniform=u,informed=i,
                           change=delta,change_pct=pct))
    pd.DataFrame(export).to_csv(output/'advisor_welfare_table.csv',index=False)
    table='\n'.join(rows[:2]+[r'\addlinespace']+rows[2:4]+[r'\addlinespace']+rows[4:])
    settings=json.loads((theory/'settings.json').read_text())
    assert settings['learning'] is False
    tex=r'''\documentclass[11pt]{article}
\usepackage[letterpaper,landscape,margin=0.55in]{geometry}
\usepackage{graphicx,booktabs,amsmath,array,fancyhdr,xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\newcommand{\titleline}[1]{{\Large\bfseries #1}\par\vspace{3pt}}
\begin{document}
\titleline{Theory benchmark: optimized prices}
\small
Normal gamma: mean 0, SD 1.25; $\beta=2$, $\alpha=-1$, zero marginal cost, $\delta=1$.
Uniform observes history but not gamma; Informed observes its quintile. Neither learns.
All offered prices are constrained to be nonnegative.
\begin{center}
\includegraphics[width=9.5in,height=5.4in,keepaspectratio]{advisor_assets/theory_prices.pdf}
\end{center}
Each quintile has 20\% probability. Points are optimized offers at the indicated decision,
without averaging over future purchase histories. The informed firm knows the quintile, not exact gamma.
\newpage
\titleline{Theory benchmark: welfare and revenue}
\small
Percentage changes are Informed minus Uniform, divided by the absolute Uniform baseline.
CS is consumer surplus; TW is total welfare.
\begin{center}
\includegraphics[width=9.8in,height=5.95in,keepaspectratio]{advisor_assets/theory_welfare.pdf}
\end{center}
\newpage
\titleline{Plain prices: trimmed, flavor switching specification}
\small
Gamma is trimmed at the 5th and 95th percentiles, with the retained distribution renormalized.
All offered prices are constrained to be nonnegative.
\begin{center}
\includegraphics[width=9.5in,height=5.35in,keepaspectratio]{advisor_assets/plain_prices.pdf}
\end{center}
Inside means plain yogurt; outside means no plain yogurt (berry, other yogurt, or no yogurt).
Q1--Q5 each contain 20\% of the retained distribution (18\% of original mass).
\newpage
\titleline{Plain welfare: trimmed, flavor switching specification}
\small
Percentage changes are Informed minus Uniform, divided by the absolute Uniform baseline.
CS is consumer surplus; TW is total welfare.
\begin{center}
\includegraphics[width=9.8in,height=5.95in,keepaspectratio]{advisor_assets/plain_welfare.pdf}
\end{center}
\newpage
\titleline{Welfare outliers: dollar levels and percentage changes}
\small
Trimmed, flavor switching specification. Values are expected dollars per consumer
over the two modeled shopping weeks, conditional on the stated initial state and quintile.
CS is consumer surplus; TW is total welfare (CS plus revenue at zero cost).
\vspace{10pt}
\begin{center}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{lllrrrr}
\toprule
Initial state & Quintile & Metric & Uniform (\$) & Informed (\$) & Change (\$) & Change (\%)\\
\midrule
TABLE_ROWS
\bottomrule
\end{tabular}
\end{center}
\end{document}
'''
    (output/'advisor_comparison.tex').write_text(tex.replace('TABLE_ROWS',table))
    print(output/'advisor_comparison.tex')


if __name__=='__main__':
    build(Path(__file__).resolve().parents[3])
