"""Re-solve entry-only demand using estimates and theoretical integration only."""
from pathlib import Path
from statistics import NormalDist
import json
import re
import numpy as np
import pandas as pd
from entry_switching_model import solve,evaluate
from build_advisor_packet import plot_prices,plot_welfare


def case(folder,mean,sd,alpha,beta,trim):
    x,w=np.polynomial.legendre.leggauss(21)
    u=trim[0]+(trim[1]-trim[0])*((np.arange(5)[:,None]+(x+1)/2)/5).ravel()
    g=np.array([NormalDist(mean,sd).inv_cdf(float(p)) for p in u])
    weights=np.tile(w/10,5);bins=np.repeat(np.arange(1,6),21)
    prices=[];outcomes=[];diagnostics=[]
    for h in ['j','0']:
        uniform=solve(g,weights,h,alpha,beta)
        for regime in ['Uniform','Informed']:
            for k in range(1,6):
                mask=bins==k;wk=weights[mask]/.2
                policy=uniform if regime=='Uniform' else solve(g[mask],wk,h,alpha,beta)
                key=dict(h0=h,regime=regime,quintile=k)
                prices.append(dict(**key,p1=policy['p1'],p2_after_inside=policy['p2j'],p2_after_outside=policy['p20']))
                diagnostics.append(dict(**key,**policy))
                result=evaluate(g[mask],h,alpha,beta,1.,policy['p1'],policy['p2j'],policy['p20'])
                outcomes.append(dict(**key,**{name:float(wk@v) for name,v in result.items()}))
    folder.mkdir(parents=True,exist_ok=True)
    p=pd.DataFrame(prices);o=pd.DataFrame(outcomes);d=pd.DataFrame(diagnostics)
    assert p.p2_after_inside.nunique()==1
    assert p[['p1','p2_after_inside','p2_after_outside']].min().min()>=0
    assert d.kkt_residual.max()<1e-6
    p.to_csv(folder/'price_policies.csv',index=False);o.to_csv(folder/'quintile_outcomes.csv',index=False)
    d.to_csv(folder/'solver_diagnostics.csv',index=False)
    (folder/'settings.json').write_text(json.dumps(dict(mean_gamma=mean,sd_gamma=sd,alpha=alpha,beta=beta,
        retained_quantiles=trim,s_j=0,s_0=1,outside_utility=0,learning=False,price_floor=0,
        delta=1,demand_refitted=False,initial_history='conditional plain versus not plain',
        distribution_across_initial_histories='identical',integration_nodes_per_quintile=21),indent=2)+'\n')
    return p,o


def build(root):
    base=Path(root)/'trimmed_flavor_gamma';out=base/'entry_switching';assets=out/'assets'
    assets.mkdir(parents=True,exist_ok=True)
    s=json.loads((base/'no_subsidies/plain_trimmed_05_95/settings.json').read_text())
    plain_p,plain=case(out/'plain',s['gamma_mean'],s['gamma_original_sd'],s['alpha'],s['beta_plain'],(.05,.95))
    theory_p,theory=case(out/'theory',0.,1.25,-1.,2.,(0.,1.))
    for label,p,o in [('plain',plain_p,plain),('theory',theory_p,theory)]:
        plot_prices(p,assets/f'{label}_prices.pdf');plot_welfare(o,assets/f'{label}_welfare.pdf')
    table=[];rows=[]
    for h,k in [('0',1),('j',1),('0',5)]:
        for metric,name in [('consumer_surplus','CS'),('total_welfare','TW')]:
            r=plain[(plain.h0==h)&(plain.quintile==k)].set_index('regime')
            u=float(r.loc['Uniform',metric]);i=float(r.loc['Informed',metric]);pct=100*(i-u)/abs(u) if u else np.nan
            state='Outside' if h=='0' else 'Inside'
            table.append(dict(initial_state=state,quintile=k,metric=name,uniform=u,informed=i,change=i-u,change_pct=pct))
            rows.append(f'{state} & Q{k} & {name} & {u:.5f} & {i:.5f} & {i-u:+.5f} & {pct:+,.1f}'+r'\% \\')
    pd.DataFrame(table).to_csv(out/'advisor_welfare_table.csv',index=False)
    tex=(base/'advisor_comparison.tex').read_text().replace('advisor_assets/','entry_switching/assets/')
    tex=tex.replace('All offered prices are constrained to be nonnegative.',
        r'All offered prices are nonnegative. Switching term: $s_j=0$, $s_0=1$; outside utility is zero.')
    tex=re.sub(r'(\\midrule\n).*?(\\bottomrule)',lambda m:m.group(1)+'\n'.join(rows)+'\n'+m.group(2),tex,flags=re.S)
    math=Path(__file__).with_name('entry_switching_math.tex').read_text()
    tex=tex.replace(r'\end{document}',math+'\n'+r'\end{document}')
    (base/'advisor_comparison_entry_switching.tex').write_text(tex)

if __name__=='__main__':
    build(Path(__file__).resolve().parents[3])
