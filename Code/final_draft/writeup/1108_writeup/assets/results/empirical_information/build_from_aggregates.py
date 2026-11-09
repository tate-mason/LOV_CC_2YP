"""Rebuild presentation charts from aggregate exported information-friction results only."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/empirical-results-mpl')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if p.name=='LOV_CC_2YP')
src=root/'res_pricing1108'
asset=Path(__file__).resolve().parents[1]
def table(path,caption,label,columns,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 text='\\begin{table}[htbp]\\centering\\small\n\\caption{'+caption+'}\n\\label{tab:'+label+'}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+columns+'}\\toprule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n'
 path.write_text(text)
 return text


# Replot aggregate information results using consistent persistent-type revenue.
for flavor in ['plain','berry']:
 target=asset/'empirical_information'/flavor;target.mkdir(parents=True,exist_ok=True)
 frame=pd.read_csv(src/'pricing'/flavor/'information_friction_aggregate.csv')
 scenarios=['Equal types','Low type more likely','Medium type more likely','High type more likely'];ticks=['Equal','More low types','More middle types','More high types'];x=np.arange(4)
 for metric,name,ylabel in [('persistent_type_revenue','revenues','Expected discounted revenue'),('gap','information_value','Revenue gain from knowing novelty type')]:
  fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
  for ax,h in zip(axes,['0','j']):
   v=frame[frame.h0.astype(str)==h].set_index(['scenario','case'])
   known=np.array([v.loc[(sc,'known'),'persistent_type_revenue'] for sc in scenarios]);unknown=np.array([v.loc[(sc,'unknown'),'persistent_type_revenue'] for sc in scenarios])
   assert np.all(known>=unknown-1e-7)
   if metric=='gap':ax.bar(x,known-unknown,color='#386cb0')
   else:
    ax.bar(x-.18,known,.36,label='Known novelty type',color='#386cb0');ax.bar(x+.18,unknown,.36,label='Fixed-belief offers',color='#e08032')
   ax.set_xticks(x,ticks,rotation=18);ax.set_title('Initially '+('outside' if h=='0' else 'inside'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  axes[0].set_ylabel(ylabel)
  if metric!='gap':fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
  fig.savefig(target/(name+'.pdf'));plt.close(fig)
 for name in ['first_prices','terminal_prices']:
  shutil.copy2(src/'pricing'/flavor/('information_friction_'+name+'.pdf'),target/(name+'.pdf'))
 rows=[r'Prior & Initial state & Known & Fixed-belief policy & Difference\\\midrule']
 for sc in scenarios:
  for h in ['0','j']:
   v=frame[(frame.scenario==sc)&(frame.h0.astype(str)==h)].set_index('case');k=v.loc['known','persistent_type_revenue'];u=v.loc['unknown','persistent_type_revenue']
   rows.append(f"{sc} & ${h}$ & {k:,.2f} & {u:,.2f} & {k-u:.2f} \\\\")
 table(target/'results.tex',flavor.capitalize()+': revenue evaluated under persistent consumer types','empirical-information-'+flavor,'llrrr',rows)
