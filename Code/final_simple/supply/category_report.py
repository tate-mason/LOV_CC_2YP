"""Results-only category-pricing figures and tables; no household input."""
from pathlib import Path
from statistics import NormalDist
import json
import os
import tempfile

import numpy as np
import pandas as pd

from category_pricing import solve, outcomes

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'category-pricing-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


def percentage(new, baseline):
    return np.nan if abs(baseline) <= 1e-12 else 100*(new-baseline)/abs(baseline)


def build(output, audit, previous_results=None):
    output = Path(output)
    table = pd.read_csv(output/'point_estimates.csv')
    if set(table)-{'parameter','estimate','at_bound'}:
        raise ValueError('Expected population estimates only.')
    p = table.set_index('parameter').estimate
    alpha = -float(np.exp(p.mu_log_minus_alpha+.5*p.sd_log_minus_alpha**2))
    beta = p.inside_constant+np.array([0.,p.beta_berry,p.beta_plain])
    flavor_names = ['other','berry','plain']
    x,w = np.polynomial.legendre.leggauss(21)
    probability = ((np.arange(5)[:,None]+(x+1)/2)/5).ravel()
    gamma = np.array([NormalDist(p.mean_novelty,p.sd_novelty).inv_cdf(float(u)) for u in probability])
    mass = np.tile(w/10,5)
    bins = np.repeat(np.arange(1,6),21)
    initial_total = audit['initial_inside']+audit['initial_outside']
    if initial_total != audit['fitted_households'] or initial_total <= 0:
        raise ValueError('Initial counts do not match the fitted sample.')
    state_weights = {'j':audit['initial_inside']/initial_total,'0':audit['initial_outside']/initial_total}
    all_summaries = []
    all_policy = []
    for mode,mode_title in [('common_price','One common yogurt price'),
                            ('separate_flavor_prices','Separate flavor prices')]:
        folder = output/mode;folder.mkdir(exist_ok=True)
        policy_rows, summary, checks = [],[],[]
        for h0 in ['j','0']:
            uniform = solve(gamma,mass,h0,alpha,beta,mode=mode)
            checks.append(dict(h0=h0,regime='Uniform',group='population',
                objective=uniform['objective'],revenue=uniform['reported_revenue'],
                residual=uniform['flavor_first_order_residual']))
            for k in range(1,6):
                selected = bins==k
                conditional = mass[selected]/mass[selected].sum()
                informed = solve(gamma[selected],conditional,h0,alpha,beta,mode=mode)
                checks.append(dict(h0=h0,regime='Informed',group=f'Q{k}',
                    objective=informed['objective'],revenue=informed['reported_revenue'],
                    residual=informed['flavor_first_order_residual']))
                for regime,policy in [('Uniform',uniform),('Informed',informed)]:
                    result = outcomes(gamma[selected],h0,alpha,beta,1.,
                        policy['p1'],policy['p2j'],policy['p20'])
                    summary.append(dict(mode=mode,h0=h0,regime=regime,quintile=k,
                        population_mass=.2,
                        **{metric:float(conditional@value) for metric,value in result.items()}))
                    for j,flavor in enumerate(flavor_names):
                        policy_rows.append(dict(mode=mode,h0=h0,regime=regime,quintile=k,
                            flavor=flavor,p1=policy['p1'][j],p2_after_yogurt=policy['p2j'][j],
                            p2_after_no_yogurt=policy['p20'][j]))
        policy = pd.DataFrame(policy_rows)
        data = pd.DataFrame(summary)
        check = pd.DataFrame(checks)
        np.testing.assert_allclose(check.objective,check.revenue,atol=1e-10)
        if check.residual.abs().max()>1e-6:
            raise ValueError('Flavor-price first-order check failed.')
        check.to_csv(folder/'solver_diagnostics.csv',index=False)
        policy.to_csv(folder/'price_policies.csv',index=False)
        data.to_csv(folder/'quintile_outcomes.csv',index=False)
        all_summaries.append(data);all_policy.append(policy)
        changes=[]
        metrics=[('consumer_surplus','CS'),('revenue','Revenue'),('total_welfare','Total welfare')]
        for (h0,k),part in data.groupby(['h0','quintile']):
            r=part.set_index('regime')
            changes.append(dict(h0=h0,quintile=k,**{
                metric+'_pct_change':percentage(r.loc['Informed',metric],r.loc['Uniform',metric])
                for metric,_ in metrics}))
        changes=pd.DataFrame(changes)
        changes.to_csv(folder/'welfare_revenue_percentage_changes.csv',index=False)
        with PdfPages(folder/'pricing_results.pdf') as packet:
            plot_flavors=['other'] if mode=='common_price' else flavor_names
            for flavor in plot_flavors:
                fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True,layout='constrained')
                fig.suptitle(mode_title+(' / '+flavor if mode!='common_price' else ''),fontsize=15)
                for ax,(h0,key,label) in zip(axes.ravel(),[
                        ('j','p1','Period 1: initially yogurt'),('0','p1','Period 1: initially no yogurt'),
                        ('j','p2_after_yogurt','Period 2: after yogurt'),
                        ('j','p2_after_no_yogurt','Period 2: after no yogurt')]):
                    for regime,color,offset,marker in [('Uniform','#555555',-.08,'D'),('Informed','#2E6896',.08,'o')]:
                        part=policy[(policy.h0==h0)&(policy.flavor==flavor)&(policy.regime==regime)].sort_values('quintile')
                        ax.scatter(part.quintile+offset,part[key],color=color,marker=marker,label=regime)
                    ax.set(title=label,xticks=range(1,6),xticklabels=[f'Q{k}' for k in range(1,6)],
                           xlabel='Category-switching gamma quintile',ylabel='Optimized price ($)')
                    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
                fig.supxlabel('Observed shopping weeks, including no yogurt. No learning; fixed empirical alpha and flavor tastes.',fontsize=9)
                stem='prices' if mode=='common_price' else 'prices_'+flavor
                fig.savefig(folder/(stem+'.png'),dpi=180,bbox_inches='tight')
                packet.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            fig.suptitle(mode_title+' — Informed versus Uniform',fontsize=15)
            for row_axes,h0 in zip(axes,['j','0']):
                part=changes[changes.h0==h0].sort_values('quintile')
                for ax,(metric,label) in zip(row_axes,metrics):
                    ax.axhline(0,color='#555555',label='Uniform',linewidth=.8)
                    ax.bar(part.quintile,part[metric+'_pct_change'],color='#2E6896',label='Informed')
                    ax.set(title=label+' / initially '+('yogurt' if h0=='j' else 'no yogurt'),
                           xticks=range(1,6),xlabel='Gamma quintile',ylabel='Change / |Uniform baseline| (%)')
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100));ax.legend(fontsize=8)
            packet.savefig(fig);fig.savefig(folder/'welfare_revenue_changes.png',dpi=180,bbox_inches='tight');plt.close(fig)
        combined=[]
        for regime,part in data.groupby('regime'):
            weights=part.h0.map(state_weights).to_numpy()*.2
            combined.append(dict(regime=regime,**{metric:float(weights@part[metric]) for metric,_ in metrics}))
        combined=pd.DataFrame(combined).set_index('regime')
        combined.to_csv(folder/'observed_initialization_outcomes.csv')
        pd.DataFrame([dict(metric=metric,change_pct=percentage(combined.loc['Informed',metric],
            combined.loc['Uniform',metric])) for metric,_ in metrics]).to_csv(
                folder/'observed_initialization_percentage_changes.csv',index=False)
        contributions=data[['h0','regime','quintile','revenue']].copy()
        contributions['weighted_revenue']=.2*contributions.revenue
        contributions['share_of_total_revenue_pct']=100*contributions.weighted_revenue/contributions.groupby(
            ['h0','regime']).weighted_revenue.transform('sum')
        contributions.to_csv(folder/'revenue_contributions.csv',index=False)
    common,separate=all_policy
    maximum_difference = 0.
    for key in ['p1','p2_after_yogurt','p2_after_no_yogurt']:
        np.testing.assert_allclose(common[key],separate[key],atol=1e-9)
        maximum_difference = max(maximum_difference,float(np.max(np.abs(common[key]-separate[key]))))
    pd.DataFrame([dict(price_structure='common and separate',maximum_price_difference=maximum_difference,
        reason='one owner; fixed common alpha; zero equal costs; category-only transitions')]).to_csv(
            output/'price_structure_comparison.csv',index=False)
    if previous_results is not None:
        previous_results=Path(previous_results)
        old=pd.read_csv(previous_results/'observed_initialization_percentage_changes.csv')
        if {'household_code','household_id'} & set(old):
            raise ValueError('Expected aggregate old results only.')
        old.to_csv(output/'previous_flavor_model_percentage_changes.csv',index=False)
    (output/'pricing_settings.json').write_text(json.dumps(dict(alpha=alpha,beta_flavor=beta.tolist(),
        gamma_mean=float(p.mean_novelty),gamma_sd=float(p.sd_novelty),delta=1.,cost=0.,
        learning=False,initial_state_counts=state_weights,one_owner_of_all_flavors=True,
        observation_unit='observed shopping week, including no yogurt',
        price_structure_equality='exact common-markup reduction; independent flavor FOCs checked'),indent=2)+'\n')
    (output/'README.md').write_text(
        '# Category switching with flavor tastes\n\n'
        'Newly estimated yogurt-versus-no-yogurt switching on observed shopping weeks, including zeros. '
        'Initial state is the first actual shopping week. Both priced periods are further observed '
        'shopping occasions, not necessarily adjacent calendar weeks. Future choices are endogenous.\n\n'
        'Open common_price/pricing_results.pdf and separate_flavor_prices/pricing_results.pdf. '
        'Tables are saved separately in those folders. Uniform does not observe type; Informed knows '
        'its quintile. Neither learns after purchase. Alpha and flavor beta are fixed in pricing.\n\n'
        'Under one owner, equal zero costs, a common alpha and category-only histories, the '
        'separate-price optimum has equal flavor prices. Flavor constants affect category demand '
        'through log-sum-exp. This equality is not evidence that flavor tastes were omitted.\n\n'
        'Welfare/revenue changes use 100*(Informed-Uniform)/abs(Uniform); zero baselines are undefined. '
        'Revenue-contribution tables report shares as separately requested.\n\n'
        'parameter_comparison.csv compares coefficients to the earlier flavor-switching fit. '
        'Gamma now describes category switching, so the two estimates have different meanings. '
        'Any previous flavor-seller results also have different product scope and are not a '
        'matched causal counterfactual. No empirical records or household identifiers are exported.\n')
