"""Quantify the unchanged binary theory using aggregate demand estimates only.

No panels, histories, observed-price records, or empirical I/O. Only common and
informed pricing; habit is held at its reference level, not advanced over time.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import os

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, wrightomega

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_calibration import known_prices, uncertainty

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'quantified-theory-mpl'))


def compare(gamma, weights, alpha, beta, delta, h0):
    """Original theory policies; evaluate both using the same persistent gamma."""
    informed = [known_prices(g, h0, alpha, beta, delta) for g in gamma]
    common = uncertainty(gamma, weights, h0, alpha, beta, delta)
    p1 = common['expected_p1']
    s1 = expit(beta+alpha*p1+(-gamma if h0 == 'j' else gamma))
    s2j = expit(beta+alpha*common['p2j']-gamma)
    s20 = expit(beta+alpha*common['p20']+gamma)
    revenue = p1*s1+delta*(s1*common['p2j']*s2j+(1-s1)*common['p20']*s20)
    known_revenue = np.array([r['profit'] for r in informed])
    if np.any(known_revenue+1e-7 < revenue):
        raise AssertionError('Informed optimum must dominate each type’s common-policy revenue.')
    return informed, common, revenue


def run(estimates, output, nodes=9, delta=1.):
    # This explicitly named file contains parameter estimates, never research records.
    table = pd.read_csv(estimates)
    if set(table.columns)-{'parameter', 'estimate', 'at_bound'}:
        raise ValueError('Expected aggregate parameter-estimate columns only.')
    if table.parameter.duplicated().any():
        raise ValueError('Duplicate parameter names.')
    p = table.set_index('parameter').estimate.to_dict()
    if not np.isfinite(list(p.values())).all() or p['sd_novelty'] <= 0:
        raise ValueError('Invalid parameter estimates.')
    alpha = -float(np.exp(p['mu_log_minus_alpha']+.5*p['sd_log_minus_alpha']**2))
    x, weights = hermgauss(nodes)
    gamma = p['mean_novelty']+np.sqrt(2)*p['sd_novelty']*x
    weights /= np.sqrt(np.pi)
    labels = np.where(x < -.6744897501960817/np.sqrt(2), 'low',
                      np.where(x > .6744897501960817/np.sqrt(2), 'high', 'middle'))
    if not all(np.any(labels == group) for group in ['low', 'middle', 'high']):
        raise ValueError('Quadrature must represent all three display groups.')
    priors = [('Fitted distribution', weights)]
    for favored in ['low', 'middle', 'high']:
        w = weights.copy()
        for group in ['low', 'middle', 'high']:
            mask = labels == group
            w[mask] *= (.5 if group == favored else .25)/weights[mask].sum()
        priors.append((favored.title()+' more likely', w))
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model='original binary theory with aggregate estimated inputs',
        utility='beta_flavor + alpha*p + gamma*(1{previous not focal}-1{previous focal})',
        alpha=alpha, alpha_mapping='negative population mean magnitude: -exp(mu_log_minus_alpha+sd_log_minus_alpha^2/2)',
        beta_mapping='inside_constant + beta_flavor; other is reference flavor',
        habit='held fixed at reference stock zero; no habit dynamics',
        outside='theory outside utility normalized to zero; no composite outside reconstruction',
        information='informed knows gamma; common knows distribution; fixed beliefs in both histories',
        evaluation='same persistent gamma in both periods; actual revenue, not subjective common objective',
        delta=delta, cost=0., nodes=nodes, gamma_mean=p['mean_novelty'], gamma_sd=p['sd_novelty'],
        observed_price_comparison=False, demand_refitted=False,
        source=str(estimates.resolve()), source_sha256=hashlib.sha256(estimates.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    all_totals = []
    for flavor in ['plain', 'berry', 'other']:
        beta = p['inside_constant']+p.get('beta_'+flavor, 0.)
        folder = output/flavor
        folder.mkdir(exist_ok=True)
        rows, totals = [], []
        for scenario, w in priors:
            for h0 in ['j', '0']:
                informed, common, common_revenue = compare(gamma, w, alpha, beta, delta, h0)
                for k, g in enumerate(gamma):
                    for regime in ['informed', 'common']:
                        s1 = expit(beta+alpha*common['expected_p1']+(-g if h0 == 'j' else g))
                        values = informed[k] if regime == 'informed' else dict(
                            expected_p1=common['expected_p1'],
                            expected_p2=s1*common['p2j']+(1-s1)*common['p20'], profit=common_revenue[k])
                        rows.append(dict(scenario=scenario, h0=h0, regime=regime, gamma=g,
                            group=labels[k], weight=w[k], p1=values['expected_p1'],
                            expected_p2=values['expected_p2'], revenue=values['profit']))
                for regime in ['informed', 'common']:
                    selected = [r for r in rows if r['scenario']==scenario and r['h0']==h0 and r['regime']==regime]
                    totals.append(dict(flavor=flavor, scenario=scenario, h0=h0, regime=regime,
                        **{key: sum(r['weight']*r[key] for r in selected) for key in ['p1','expected_p2','revenue']}))
        results, summary = pd.DataFrame(rows), pd.DataFrame(totals)
        results.to_csv(folder/'type_results.csv', index=False)
        summary.to_csv(folder/'comparison.csv', index=False)
        pd.DataFrame(dict(gamma=gamma, weight=weights, group=labels)).to_csv(folder/'gamma_distribution.csv', index=False)
        (folder/'settings.json').write_text(json.dumps(dict(metadata, flavor=flavor, beta=beta), indent=2)+'\n')
        figures(results, summary, alpha, beta, delta, gamma, folder)
        lines = [r'\begin{tabular}{llrrr}\toprule',r'Initial state & Distribution & Informed & Common & Gain\\\midrule']
        for scenario, _ in priors:
            for h0 in ['j','0']:
                r = summary[(summary.scenario==scenario)&(summary.h0==h0)].set_index('regime').revenue
                lines.append(f'${h0}$ & {scenario} & {r.informed:.4f} & {r.common:.4f} & {r.informed-r.common:.4f} '+r'\\')
        (folder/'revenue_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
        all_totals.extend(totals)
    pd.DataFrame(all_totals).to_csv(output/'all_flavors_comparison.csv', index=False)
    (output/'README.md').write_text('Only common versus informed pricing in the original binary theory.\n'
        'Inputs are aggregate parameter estimates. No observed-price comparison or demand refit.\n'
        'Habit is held at zero reference stock. Flavor intercepts are relative to the theory outside option.\n'
        'These are quantified theory results, not household counterfactual predictions.\n')
    return pd.DataFrame(all_totals)


def figures(results, summary, alpha, beta, delta, support, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig, name):
        for ext in ['pdf','png']:
            fig.savefig(output/(name+'.'+ext), bbox_inches='tight', dpi=170)
        plt.close(fig)
    grid = np.unique(np.r_[np.linspace(min(-3.,support.min()), max(3.,support.max()),301),support])
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for h0, color in [('j','#2E6896'),('0','#B35C21')]:
        axes[0].plot(grid,[known_prices(g,h0,alpha,beta,delta)['expected_p1'] for g in grid],color=color,
                     label='Initially '+('inside' if h0=='j' else 'outside'))
        z=beta+(-grid if h0=='j' else grid)
        axes[1].plot(grid,-(1+wrightomega(z-1))/alpha,color=color,
                     label='After '+('inside' if h0=='j' else 'outside'))
    for ax,title in zip(axes,['Optimal period-1 prices','Optimal period-2 prices']):
        ax.set(xlabel=r'$\gamma$',ylabel='Price ($)',title=title);ax.legend();ax.grid(alpha=.2)
    save(fig,'optimal_prices_by_gamma')
    base=results[results.scenario=='Fitted distribution']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
    for ax,h0 in zip(axes,['j','0']):
        selected=base[base.h0==h0]
        for group,color in [('low','#2E6896'),('middle','#208475'),('high','#B35C21')]:
            r=selected[(selected.regime=='informed')&(selected.group==group)]
            ax.plot([1,2],np.average(r[['p1','expected_p2']],axis=0,weights=r.weight),marker='o',color=color,label=group.title()+' group (informed)')
        r=selected[selected.regime=='common']
        ax.plot([1,2],np.average(r[['p1','expected_p2']],axis=0,weights=r.weight),marker='D',ls='--',color='black',label='Common prices')
        ax.set(title='Initially '+('inside' if h0=='j' else 'outside'),xticks=[1,2],xticklabels=[r'$p_1$',r'$\mathbb{E}[p_2]$'],ylabel='Price ($)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'information_price_paths_comparison')
    scenarios=list(summary.scenario.unique())
    for gain,stem in [(False,'information_revenues'),(True,'information_value')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
        for ax,h0 in zip(axes,['j','0']):
            r=summary[summary.h0==h0].pivot(index='scenario',columns='regime',values='revenue').reindex(scenarios)
            x=np.arange(len(r))
            if gain: ax.bar(x,r.informed-r.common,label='Informed minus common')
            else:
                ax.bar(x-.18,r.common,width=.36,label='Common')
                ax.bar(x+.18,r.informed,width=.36,label='Informed')
            ax.set(title='Initially '+('inside' if h0=='j' else 'outside'),xticks=x,
                   xticklabels=['Fitted','Low-heavy','Middle-heavy','High-heavy'],ylabel='Two-period revenue ($)')
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        save(fig,stem)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    totals=run(args.estimates,args.output)
    print(totals[totals.scenario=='Fitted distribution'].to_string(index=False))
