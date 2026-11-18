"""Separate advisor packet: unchanged policies, zero-outside welfare primitives."""
from pathlib import Path
from statistics import NormalDist
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demand'))
from zero_outside_welfare import evaluate_zero_outside
from build_advisor_packet import plot_welfare


def evaluate_case(policies,mean,sd,alpha,beta,quantiles,folder):
    x,w=np.polynomial.legendre.leggauss(21)
    lo,hi=quantiles
    probability=lo+(hi-lo)*((np.arange(5)[:,None]+(x+1)/2)/5).ravel()
    gamma=np.array([NormalDist(mean,sd).inv_cdf(float(u)) for u in probability])
    bins=np.repeat(np.arange(1,6),21);weights=np.tile(w/10,5)
    rows=[]
    for r in policies.itertuples():
        selected=bins==r.quintile;conditional=weights[selected]/.2
        result=evaluate_zero_outside(gamma[selected],r.h0,alpha,beta,1.,r.p1,
                                     r.p2_after_inside,r.p2_after_outside)
        metrics=['consumer_surplus','total_welfare','revenue','profit','quantity1','quantity2',
                 'consumer_surplus1','consumer_surplus2','original_consumer_surplus',
                 'original_total_welfare','original_consumer_surplus1','original_consumer_surplus2']
        rows.append(dict(h0=r.h0,regime=r.regime,quintile=r.quintile,
            **{key:float(conditional@result[key]) for key in metrics}))
    data=pd.DataFrame(rows)
    assert (data.consumer_surplus>=0).all()
    folder.mkdir(parents=True,exist_ok=True)
    policies.to_csv(folder/'price_policies.csv',index=False)
    data.to_csv(folder/'quintile_outcomes.csv',index=False)
    comparisons=[]
    for (h0,k),part in data.groupby(['h0','quintile']):
        r=part.set_index('regime')
        for metric in ['consumer_surplus','total_welfare']:
            old_u=float(r.loc['Uniform','original_'+metric]);old_i=float(r.loc['Informed','original_'+metric])
            new_u=float(r.loc['Uniform',metric]);new_i=float(r.loc['Informed',metric])
            comparisons.append(dict(h0=h0,quintile=k,metric=metric,
                original_uniform=old_u,original_informed=old_i,original_change=old_i-old_u,
                zero_outside_uniform=new_u,zero_outside_informed=new_i,zero_outside_change=new_i-new_u,
                original_change_pct=100*(old_i-old_u)/abs(old_u) if old_u else np.nan,
                zero_outside_change_pct=100*(new_i-new_u)/abs(new_u) if new_u else np.nan))
    pd.DataFrame(comparisons).to_csv(folder/'welfare_normalization_comparison.csv',index=False)
    return data


def build(root):
    root=Path(root);base=root/'trimmed_flavor_gamma';output=base/'zero_outside'
    assets=output/'assets';assets.mkdir(parents=True,exist_ok=True)
    plain_source=base/'no_subsidies/plain_trimmed_05_95'
    settings=json.loads((plain_source/'settings.json').read_text())
    source=plain_source/'price_policies.csv'
    fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
    plain_policies=pd.read_csv(source,dtype={'h0':str})
    plain=evaluate_case(plain_policies,settings['gamma_mean'],settings['gamma_original_sd'],
        settings['alpha'],settings['beta_plain'],(.05,.95),output/'plain')
    original=pd.read_csv(plain_source/'quintile_outcomes.csv',dtype={'h0':str})
    np.testing.assert_allclose(plain.revenue,original.revenue,atol=1e-12)
    np.testing.assert_allclose(plain.original_consumer_surplus,original.consumer_surplus,atol=1e-12)
    np.testing.assert_allclose(plain.original_total_welfare,original.total_welfare,atol=1e-12)
    theory_source=base/'no_subsidies/01_pure_theory'
    theory_settings=json.loads((theory_source/'settings.json').read_text())
    diagnostics=pd.read_csv(theory_source/'solver_diagnostics.csv',dtype={'h0':str})
    rows=[]
    for h0 in ['j','0']:
        for regime,code in [('Uniform','common'),('Informed','informed')]:
            for k in range(1,6):
                group='population' if code=='common' else f'bin_{k}'
                row=diagnostics[(diagnostics.h0==h0)&(diagnostics.regime==code)&(diagnostics.group==group)].iloc[0]
                rows.append(dict(h0=h0,regime=regime,quintile=k,p1=row.p1,
                    p2_after_inside=row.p2j,p2_after_outside=row.p20))
    theory_policies=pd.DataFrame(rows)
    theory=evaluate_case(theory_policies,theory_settings['mean_gamma'],theory_settings['sd_gamma'],
        theory_settings['alpha'],theory_settings['beta'],(0.,1.),output/'theory')
    plot_welfare(theory,assets/'theory_welfare.pdf')
    plot_welfare(plain,assets/'plain_welfare.pdf')
    selections=[('0',1,'consumer_surplus','CS'),('0',1,'total_welfare','TW'),
                ('j',1,'consumer_surplus','CS'),('j',1,'total_welfare','TW'),
                ('0',5,'consumer_surplus','CS'),('0',5,'total_welfare','TW')]
    rows=[];table=[]
    for h0,k,metric,label in selections:
        r=plain[(plain.h0==h0)&(plain.quintile==k)].set_index('regime')
        u=float(r.loc['Uniform',metric]);i=float(r.loc['Informed',metric]);change=i-u
        pct=100*change/abs(u)
        state='Outside' if h0=='0' else 'Inside'
        rows.append(f'{state} & Q{k} & {label} & {u:.5f} & {i:.5f} & {change:+.5f} & {pct:+,.1f}'+r'\% \\')
        table.append(dict(initial_state=state,quintile=k,metric=label,uniform=u,informed=i,change=change,change_pct=pct))
    pd.DataFrame(table).to_csv(output/'advisor_welfare_table.csv',index=False)
    table_rows='\n'.join(rows[:2]+[r'\addlinespace']+rows[2:4]+[r'\addlinespace']+rows[4:])
    text=(base/'advisor_comparison.tex').read_text()
    text=text.replace('advisor_assets/theory_welfare.pdf','zero_outside/assets/theory_welfare.pdf')
    text=text.replace('advisor_assets/plain_welfare.pdf','zero_outside/assets/plain_welfare.pdf')
    text=text.replace('CS is consumer surplus; TW is total welfare.',
                      'CS uses zero outside utility; TW is CS plus revenue.')
    text=text.replace('CS is consumer surplus; TW is total welfare (CS plus revenue at zero cost).',
                      'CS uses zero outside utility; TW is CS plus revenue at zero cost.')
    text=re.sub(r'(\\midrule\n).*?(\\bottomrule)',lambda m:m.group(1)+table_rows+'\n'+m.group(2),text,flags=re.S)
    math=Path(__file__).with_name('zero_outside_math.tex').read_text()
    text=text.replace(r'\end{document}',math)
    target=base/'advisor_comparison_zero_outside.tex'
    target.write_text(text)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==fingerprint
    provenance=dict(welfare='history-dependent WTP; zero deterministic outside utility',
        original_price_policy_sha256=fingerprint,prices_reoptimized=False,
        choices_changed=False,revenue_changed=False,demand_refitted=False,
        learning=False,price_floor=0.,source_plain_settings=settings,
        output='separate version; original advisor_comparison.pdf unchanged')
    (output/'specification.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(target)
    comparison=pd.read_csv(output/'plain/welfare_normalization_comparison.csv',dtype={'h0':str})
    print(comparison[(comparison.h0=='0')&(comparison.quintile==1)].to_string(index=False))


if __name__=='__main__':
    build(Path(__file__).resolve().parents[3])
