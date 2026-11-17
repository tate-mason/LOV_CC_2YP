"""Quantify percentile-bin pricing using aggregate demand estimates only.

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
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_calibration import known_prices, uncertainty
from binary_welfare import evaluate_policy
from welfare_reporting import write_welfare_report, prior_labels, plot_discrimination_percent
from plot_supply_bins import plot_bins

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


def discrete_gamma(mean, sd, bins=5):
    """Equal-probability normal bins, represented by conditional bin means."""
    if bins < 2 or not np.isfinite([mean, sd]).all() or sd <= 0:
        raise ValueError('Need at least two bins and a positive finite gamma SD.')
    edges = norm.ppf(np.linspace(0., 1., bins+1))
    weights = np.full(bins, 1/bins)
    gamma = mean + sd * np.diff(-norm.pdf(edges)) / weights
    return gamma, weights, np.array([f'bin_{k+1}' for k in range(bins)])


def gamma_bin_distribution(mean, sd, bins=5, nodes=9):
    """Integrate the fitted normal within percentile bins; no mean substitution.

    Gauss-Legendre integration over probability space gives exactly 1/bins
    probability per bin, including the unbounded first and last bins.
    """
    if bins < 2 or nodes < 2 or not np.isfinite([mean, sd]).all() or sd <= 0:
        raise ValueError('Need >=2 bins, >=2 nodes per bin, and positive finite SD.')
    x, w = np.polynomial.legendre.leggauss(nodes)
    probabilities = (np.arange(bins)[:, None] + (x+1)/2)/bins
    gamma = mean + sd*norm.ppf(probabilities.ravel())
    weights = np.tile(w/(2*bins), bins)
    labels = np.repeat([f'bin_{k+1}' for k in range(bins)], nodes)
    return gamma, weights, labels


def compare_bins(gamma, weights, labels, alpha, beta, delta, h0):
    """One policy per observed bin, integrating all gamma nodes in that bin.

    Preserve the theory's fixed conditional beliefs after either history.
    Bin-level objective gains need not be gains for every gamma within a bin.
    """
    common = uncertainty(gamma, weights, h0, alpha, beta, delta)
    policies = {}
    for group in dict.fromkeys(labels):
        mask = labels == group
        conditional = weights[mask]/weights[mask].sum()
        policy = uncertainty(gamma[mask], conditional, h0, alpha, beta, delta)
        outcome = evaluate_policy(gamma[mask], h0, alpha, beta, delta,
                                  policy['expected_p1'], policy['p2j'], policy['p20'])
        baseline = evaluate_policy(gamma[mask], h0, alpha, beta, delta,
                                    common['expected_p1'], common['p2j'], common['p20'])
        if conditional @ (outcome['profit']-baseline['profit']) < -1e-7:
            raise AssertionError('Bin optimum must dominate common pricing within each bin.')
        policies[group] = policy
    return [policies[group] for group in labels], common


def run(estimates, output, nodes=21, delta=1., bins=5):
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
    if bins is None:
        x, weights = hermgauss(nodes)
        gamma = p['mean_novelty']+np.sqrt(2)*p['sd_novelty']*x
        weights /= np.sqrt(np.pi)
        labels = np.where(x < -.6744897501960817/np.sqrt(2), 'low',
                          np.where(x > .6744897501960817/np.sqrt(2), 'high', 'middle'))
    else:
        gamma, weights, labels = gamma_bin_distribution(p['mean_novelty'], p['sd_novelty'], bins, nodes)
    priors = [('Fitted distribution', weights)]
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model='original binary theory with aggregate estimated inputs',
        utility='beta_flavor + alpha*p + gamma*(1{previous not focal}-1{previous focal})',
        alpha=alpha, alpha_mapping='negative population mean magnitude: -exp(mu_log_minus_alpha+sd_log_minus_alpha^2/2)',
        beta_mapping='inside_constant + beta_flavor; other is reference flavor',
        habit='held fixed at reference stock zero; no habit dynamics',
        outside='relative utility subtracts V0=gamma*1{h=j}; welfare restores this theory utility level',
        welfare='expected experienced logit utility / (-alpha); common shock constant omitted; total=CS+profit',
        welfare_discount='same delta for consumer utility and firm profit; myopic consumer choices',
        information=('informed knows percentile bin, not gamma; common knows population distribution; '
                     'fixed conditional beliefs in both histories') if bins else
                    'informed knows gamma; common knows distribution; fixed beliefs in both histories',
        evaluation='same persistent gamma in both periods; actual revenue, not subjective common objective',
        delta=delta, cost=0., nodes=len(gamma), bins=bins,
        nodes_per_bin=nodes if bins else None,
        discretization='Gauss-Legendre within equal-probability normal bins' if bins else 'Gauss-Hermite',
        gamma_mean=p['mean_novelty'], gamma_sd=p['sd_novelty'],
        observed_price_comparison=False, demand_refitted=False,
        source=str(estimates.resolve()), source_sha256=hashlib.sha256(estimates.read_bytes()).hexdigest(),
        welfare_sha256=hashlib.sha256(Path(__file__).with_name('binary_welfare.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    all_totals = []
    for flavor in ['plain', 'berry', 'other']:
        beta = p['inside_constant']+p.get('beta_'+flavor, 0.)
        folder = output/flavor
        folder.mkdir(exist_ok=True)
        rows, totals = [], []
        for scenario, w in priors:
            for h0 in ['j', '0']:
                if bins is None:
                    informed, common, _ = compare(gamma, w, alpha, beta, delta, h0)
                else:
                    informed, common = compare_bins(gamma, w, labels, alpha, beta, delta, h0)
                for k, g in enumerate(gamma):
                    for regime in ['informed', 'common']:
                        policy = informed[k] if regime == 'informed' else common
                        terminal = -(1+wrightomega(beta+np.array([-g,g])-1))/alpha
                        p2j = policy.get('p2j', terminal[0])
                        p20 = policy.get('p20', terminal[1])
                        outcome = evaluate_policy(g, h0, alpha, beta, delta,
                                                  policy['expected_p1'], p2j, p20)
                        rows.append(dict(scenario=scenario, h0=h0, regime=regime, gamma=g,
                            group=labels[k], weight=w[k], **{key:float(value) for key,value in outcome.items()}))
                for regime in ['informed', 'common']:
                    selected = [r for r in rows if r['scenario']==scenario and r['h0']==h0 and r['regime']==regime]
                    totals.append(dict(flavor=flavor, scenario=scenario, h0=h0, regime=regime,
                        **{key: sum(r['weight']*r[key] for r in selected) for key in outcome}))
        results, summary = pd.DataFrame(rows), pd.DataFrame(totals)
        results.to_csv(folder/'type_results.csv', index=False)
        if bins is not None:
            policies = results[results.regime == 'informed'][
                ['scenario','h0','group','p1','p2j','p20']].drop_duplicates()
            policies.to_csv(folder/'bin_policies.csv', index=False)
            boundaries = norm.ppf(np.linspace(0., 1., bins+1),
                                  loc=p['mean_novelty'], scale=p['sd_novelty'])
            pd.DataFrame(dict(group=[f'bin_{k+1}' for k in range(bins)],
                percentile_lower=np.arange(bins)*100/bins,
                percentile_upper=np.arange(1,bins+1)*100/bins,
                gamma_lower=boundaries[:-1], gamma_upper=boundaries[1:],
                weight=np.full(bins, 1/bins))).to_csv(folder/'bin_definitions.csv', index=False)
        summary.to_csv(folder/'comparison.csv', index=False)
        pd.DataFrame(dict(gamma=gamma, weight=weights, group=labels)).to_csv(folder/'gamma_distribution.csv', index=False)
        (folder/'settings.json').write_text(json.dumps(dict(metadata, flavor=flavor, beta=beta), indent=2)+'\n')
        figures(results, summary, alpha, beta, delta, gamma, folder)
        if bins is not None:
            plot_bins(results, folder)
        write_welfare_report(results, summary, folder)
        plot_discrimination_percent(summary, folder)
        lines = [r'\begin{tabular}{llrrr}\toprule',r'Initial state & Distribution & Informed & Uniform & Gain\\\midrule']
        for scenario, _ in priors:
            for h0 in ['j','0']:
                r = summary[(summary.scenario==scenario)&(summary.h0==h0)].set_index('regime').revenue
                label = prior_labels([scenario])[0]
                lines.append(f'${h0}$ & {label} & {r.informed:.4f} & {r.common:.4f} & {r.informed-r.common:.4f} '+r'\\')
        (folder/'revenue_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
        all_totals.extend(totals)
    pd.DataFrame(all_totals).to_csv(output/'all_flavors_comparison.csv', index=False)
    (output/'README.md').write_text('Common versus informed pricing, consumer surplus and total welfare in the original binary theory.\n'
        'Inputs are aggregate parameter estimates. No observed-price comparison or demand refit.\n'
        'Habit is held at zero reference stock. Flavor intercepts are relative to the theory outside option.\n'
        'Welfare restores the theory outside switching utility gamma*1{h=j}; zero-outside sensitivity also saved.\n'
        'Bin pricing integrates multiple gamma nodes per percentile bin; no bin-mean substitution.\n'
        'The informed regime observes the bin; prices may still depend on purchase history.\n'
        'These are quantified theory results, not household counterfactual predictions.\n')
    return pd.DataFrame(all_totals)


def figures(results, summary, alpha, beta, delta, support, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    def save(fig, name):
        for ext in ['pdf','png']:
            fig.savefig(output/(name+'.'+ext), bbox_inches='tight', dpi=170)
        plt.close(fig)
    base=results[results.scenario=='Fitted distribution']
    # Compare weighted bin means, using the same bin in the uniform baseline.
    metrics = ['p1', 'p2j', 'p20', 'expected_p2', 'consumer_surplus', 'revenue']
    weighted = base.copy()
    weighted[metrics] = weighted[metrics].mul(weighted.weight, axis=0)
    grouped = weighted.groupby(['h0','regime','group'], sort=False)[metrics+['weight']].sum()
    grouped[metrics] = grouped[metrics].div(grouped.weight, axis=0)
    uniform = grouped.xs('common',level='regime')[metrics]
    informed_bins = grouped.xs('informed',level='regime')[metrics]
    if (uniform.abs() <= 1e-12).any().any():
        raise ValueError('Percentage plots require nonzero uniform-pricing baselines.')
    percent = 100*(informed_bins-uniform)/uniform.abs()
    percent.to_csv(output/'bin_changes_percent.csv')
    def bin_labels(groups):
        if all(str(g).startswith('bin_') for g in groups):
            n = len(groups)
            return [f'{100*k/n:g}–{100*(k+1)/n:g}%' for k in range(n)]
        return [str(g).title() for g in groups]
    for measures, stem in [([('p1','Period 1'),('p2j','Period 2: inside history'),
                            ('p20','Period 2: outside history')], 'optimal_prices_by_gamma'),
                           ([('p1','Period 1'),('expected_p2','Expected period 2')],
                            'information_price_paths_comparison')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True,layout='constrained')
        width = .8/len(measures)
        for ax,h0 in zip(axes,['j','0']):
            data = percent.xs(h0,level='h0')
            x = np.arange(len(data))
            for k,(metric,label) in enumerate(measures):
                ax.bar(x+(k-(len(measures)-1)/2)*width,data[metric],width=width*.95,label=label)
            ax.set(title='Initially '+('inside' if h0=='j' else 'outside'),
                   xticks=x,xticklabels=bin_labels(data.index),
                   xlabel='Gamma percentile bin',ylabel='Price change from uniform pricing (%)')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.axhline(0.,color='black',linewidth=.7)
            ax.grid(False)
            ax.legend(fontsize=8)
        save(fig,stem)
    # Compare identical persistent types, integrating each policy's own histories.
    keys = ['scenario', 'h0', 'gamma', 'group']
    common = results[results.regime == 'common'].set_index(keys)
    informed = results[results.regime == 'informed'].set_index(keys)
    changes = informed[['consumer_surplus', 'total_welfare', 'p1', 'expected_p2']] - common[['consumer_surplus', 'total_welfare', 'p1', 'expected_p2']]
    changes['weight'] = informed.weight
    changes['consumer_winner'] = changes.consumer_surplus > 1e-9
    changes.reset_index().to_csv(output/'type_welfare_changes.csv', index=False)
    grouped_changes = changes.reset_index()
    for metric in ['consumer_surplus', 'total_welfare', 'p1', 'expected_p2']:
        grouped_changes[metric] *= grouped_changes.weight
    grouped_changes['consumer_winner_share'] = grouped_changes.consumer_winner*grouped_changes.weight
    metrics = ['consumer_surplus','total_welfare','p1','expected_p2','consumer_winner_share']
    by_bin = grouped_changes.groupby(['scenario','h0','group'],sort=False)[metrics+['weight']].sum()
    by_bin[metrics] = by_bin[metrics].div(by_bin.weight, axis=0)
    by_bin.to_csv(output/'bin_welfare_changes.csv')
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
    for ax, h0 in zip(axes, ['j', '0']):
        r = percent.xs(h0,level='h0')
        change = r.consumer_surplus
        ax.bar(np.arange(len(r)), change, color=np.where(change > 1e-9, '#208475', '#B35C21'))
        ax.axhline(0., color='black', linewidth=.7)
        ax.set(title='Initially '+('inside' if h0=='j' else 'outside'),
               ylabel='Consumer surplus change from uniform pricing (%)', xlabel='Gamma percentile bin',
               xticks=np.arange(len(r)), xticklabels=bin_labels(r.index))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(False)
    save(fig, 'consumer_surplus_by_type')
    scenarios=list(summary.scenario.unique())
    for gain,stem in [(False,'information_revenues'),(True,'information_value')]:
        if gain and len(scenarios) == 1:
            r = summary.pivot(index='h0',columns='regime',values='revenue').reindex(['j','0'])
            if (r.common <= 0).any():
                raise ValueError('Percentage revenue gains require positive uniform revenue.')
            fig,ax=plt.subplots(figsize=(6.5,4.2),layout='constrained')
            ax.bar([0,1],100*(r.informed/r.common-1),color=['#2E6896','#B35C21'],width=.55)
            ax.set(title='Value of bin information',xticks=[0,1],
                   xticklabels=['Initially inside','Initially outside'],
                   ylabel='Revenue gain over uniform pricing (%)')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.axhline(0.,color='black',linewidth=.7)
            ax.grid(False)
            save(fig,stem)
            continue
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
        for ax,h0 in zip(axes,['j','0']):
            r=summary[summary.h0==h0].pivot(index='scenario',columns='regime',values='revenue').reindex(scenarios)
            x=np.arange(len(r))
            if gain:
                if (r.common <= 0).any():
                    raise ValueError('Percentage revenue gains require positive uniform revenue.')
                ax.bar(x,100*(r.informed/r.common-1))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            else:
                if (r.common <= 0).any():
                    raise ValueError('Percentage revenue comparisons require positive uniform revenue.')
                ax.bar(x-.18,np.full(len(r),100.),width=.36,label='Uniform')
                ax.bar(x+.18,100*r.informed/r.common,width=.36,label='Bin pricing')
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.set(title='Initially '+('inside' if h0=='j' else 'outside'),xticks=x,
                   xticklabels=prior_labels(scenarios),
                   ylabel='Revenue gain over uniform pricing (%)' if gain else 'Revenue relative to uniform pricing (100%)')
            if not gain:
                ax.legend(fontsize=8)
            ax.grid(False)
        save(fig,stem)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bins',type=int,default=5,help='Equal-probability gamma types (default: 5)')
    parser.add_argument('--nodes',type=int,default=21,help='Integration nodes per bin (default: 21)')
    parser.add_argument('--quadrature',action='store_true',help='Use the previous nine-node approximation')
    args=parser.parse_args()
    totals=run(args.estimates,args.output,nodes=args.nodes,bins=None if args.quadrature else args.bins)
    print(totals.loc[totals.scenario=='Fitted distribution',
        ['flavor','h0','regime','revenue','consumer_surplus','total_welfare','quantity']].to_string(index=False))
