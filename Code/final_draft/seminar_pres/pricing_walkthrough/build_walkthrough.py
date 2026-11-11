"""Build an auditable code/math PDF from source and aggregate outputs only."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0,str(ROOT/'Code/final_simple'))
from demand.binary_information_pricing_model import solve_binary
SNAP = HERE/'source_snapshot'
SNAP.mkdir(exist_ok=True)

def esc(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')

files = {
 'binary_model': 'Code/final_simple/demand/binary_information_pricing_model.py',
 'runner': 'Code/final_simple/demand/information_pricing.py',
 'optimizer': 'Code/final_simple/demand/information_pricing_model.py',
 'history': 'Code/final_simple/demand/history.py',
 'posterior': 'Code/final_simple/demand/split_pricing.py',
 'data': 'Code/final_simple/demand/estimation.py',
 'categories': 'Code/final_simple/flavor_specs.py',
 'policy': 'Code/privacy/server_data_policy.py',
 'plots': 'Code/final_simple/demand/plot_information_pricing.py',
 'refresh': 'Code/final_draft/seminar_pres/refresh_information_pricing.py',
 'theory': 'Code/final_draft/writeup/1108_writeup/theory/gamma_uncertainty.py',
 'theory_helpers': 'Code/final_simple/supply/pricing_model.py',
 'tests': 'Code/final_simple/checks/check_binary_information_pricing.py',
 'launcher': 'Code/final_simple/server/run_information_pricing.sh',
}
manifest = {}
for key,name in files.items():
    source=ROOT/name
    snapshot=SNAP/(key+source.suffix)
    shutil.copyfile(source,snapshot)
    manifest[key]=dict(path=name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),lines=len(source.read_text().splitlines()))
(SNAP/'current_source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')

meta=json.loads((ROOT/'new_pricing_res/information_pricing_summary.json').read_text())
parameter_file=ROOT/'Results/pricing_no_disc/no_control_fit/point_estimates.csv'
assert hashlib.sha256(parameter_file.read_bytes()).hexdigest()==meta['parameters_sha256']
with parameter_file.open() as f: params={r['parameter']:float(r['estimate']) for r in csv.DictReader(f)}
rows=[r'\begin{longtable}{lr}\toprule Parameter & Saved estimate\\\midrule']
for name,value in params.items():rows.append(esc(name)+f' & {value:.9f} '+r'\\')
rows += [r'\bottomrule\end{longtable}']
(HERE/'parameter_table.tex').write_text('\n'.join(rows)+'\n')
g=np.array(meta['gamma_nodes']);w=np.array(meta['gamma_weights']);mu=params['mean_novelty'];sd=params['sd_novelty']
labels=np.where(g<mu-.6744897501960817*sd,'low',np.where(g>mu+.6744897501960817*sd,'high','middle'))
rows=[r'\begin{longtable}{rrrl}\toprule Node & $\gamma_r$ & Weight & Display group\\\midrule']
for k,(value,weight,label) in enumerate(zip(g,w,labels),1):rows.append(f'{k} & {value:.6f} & {weight:.8f} & {label} '+r'\\')
rows += [r'\bottomrule\end{longtable}']
(HERE/'quadrature_table.tex').write_text('\n'.join(rows)+'\n')
rows=[r'\begin{tabular}{lrr}\toprule Group & Numerical mass & Mean $\gamma$\\\midrule']
for name in ['low','middle','high']:
 mask=labels==name;mass=w[mask].sum();mean=float(w[mask]@g[mask]/mass)
 rows.append(f'{name.title()} & {mass:.6f} & {mean:.6f} '+r'\\')
rows += [r'\bottomrule\end{tabular}']
(HERE/'group_table.tex').write_text('\n'.join(rows)+'\n')

# A transparent mathematical example, not an empirical household or panel.
example=solve_binary(prices=np.array([2.]),available=np.ones(1,bool),beta=np.array([2.]),stock=np.array([.5]),alpha=1.,habit=.25,gamma=np.array([-1.,1.]),weights=np.ones(2)/2,initial_inside=False,focal=0)
rows=[r'\begin{tabular}{lrrrr}\toprule & $\gamma$ & $p_1$ & $p_2(j)$ & $p_2(0)$\\\midrule']
for k,gamma in enumerate(example['gamma']):
 rows.append(f'Full information & {gamma:.0f} & {example["full"]["p1"][k]:.6f} & {example["p2_full"][k,0]:.6f} & {example["p2_full"][k,1]:.6f} '+r'\\')
rows.append(f'Limited information & hidden & {example["limited"]["p1"][0]:.6f} & {example["p2_limited"][0]:.6f} & {example["p2_limited"][1]:.6f} '+r'\\')
rows += [r'\bottomrule\end{tabular}'];(HERE/'example_prices.tex').write_text('\n'.join(rows)+'\n')
(HERE/'example_values.tex').write_text(
 f'Full-information expected revenue is {example["full_revenue"]:.6f}. '
 f'Limited-information realized expected revenue is {example["limited_revenue"]:.6f}. '
 f'The limited firm\'s subjective optimized value is {example["limited_objective"]:.6f}. '
 f'The reported information gain is {example["gain"]:.6f}.\n')
p1=example['limited']['p1'];s1=example['limited']['state_probability'][:,0]
posterior_j=.5*s1/s1.mean();posterior_0=.5*(1-s1)/(1-s1.mean())
(HERE/'example_selection.tex').write_text(
 f'Under the common first price, the low-type purchase probability is {s1[0]:.6f} '
 f'and the high-type purchase probability is {s1[1]:.6f}. '
 f'The actual low-type share among purchasers is {posterior_j[0]:.6f}, '
 f'and among nonpurchasers it is {posterior_0[0]:.6f}; '
 'the restricted firm nevertheless uses probability 0.5 in both states.\n')

# Complete sources for the new pipeline, plus the exact reused dependency functions.
appendix=[]
def listing(key,title,functions=None,language='Python'):
    suffix=(ROOT/files[key]).suffix
    path='source_snapshot/'+key+suffix
    appendix.extend([r'\clearpage',r'\section{'+esc(title)+'}',
      r'\noindent Source: \path{'+files[key]+r'}.\par',
      r'\noindent SHA-256: {\scriptsize\path{'+manifest[key]['sha256']+r'}}.\par\medskip'])
    if functions is None:
        appendix.append(r'\lstinputlisting[language='+language+']{'+path+'}')
    else:
        source=(ROOT/files[key]).read_text();tree=ast.parse(source)
        for name in functions:
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            first=min([node.lineno]+[d.lineno for d in node.decorator_list])
            appendix.extend([r'\subsection{\texttt{'+esc(name)+'}}',
              r'\lstinputlisting[firstline='+str(first)+',lastline='+str(node.end_lineno)+',firstnumber='+str(first)+']{'+path+'}'])
listing('binary_model','Complete binary solver')
listing('optimizer','Reused mixture-price optimizer',['_terminal_common'])
listing('runner','Complete server orchestration and aggregation')
listing('history','History rules: only saved cumulative history is accepted by the new binary runner')
listing('posterior','Reused posterior and starting-context functions',['logsumexp','persistent_draws','posterior','forecast_context'])
listing('data','Saved-data reconstruction dependencies',['normalized_upc_expr','read_flavors','prepare_data','check_dma','switch_away_event','add_stocks'])
listing('categories','Flavor ordering and classification definitions')
listing('plots','Complete aggregate figure generation')
listing('refresh','Complete aggregate import into the presentation')
listing('launcher','Complete server submission launcher',language='bash')
listing('policy','Complete server data policy')
listing('theory','Complete binary theoretical information exercise')
listing('theory_helpers','Theoretical numerical helper functions',['expit','golden_maximum','static_price'])
listing('tests','Complete binary regression and mathematical tests')
appendix.extend([r'\clearpage',r'\section{Previous full-basket solver: the source of the earlier results}',
 r'This is preserved for comparison, not called by the corrected binary runner. Its full \texttt{solve\_environment} enumeration explains why the earlier output was not the requested binary exercise.',
 r'\lstinputlisting{source_snapshot/previous_information_pricing_model.py}'])
(HERE/'source_appendix.tex').write_text('\n'.join(appendix)+'\n')
rows=[r'\begin{longtable}{p{.22\textwidth}rp{.60\textwidth}}\toprule Snapshot & Lines & Source\\\midrule']
for key,value in manifest.items():rows.append(esc(key)+f' & {value["lines"]} & '+r'\path{'+value['path']+'} '+r'\\')
rows += [r'\bottomrule\end{longtable}'];(HERE/'source_index.tex').write_text('\n'.join(rows)+'\n')
print('Prepared parameter/quadrature tables, mathematical example, source snapshots and complete source appendix.')
