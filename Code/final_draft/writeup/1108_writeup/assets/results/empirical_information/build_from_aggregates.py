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


# Presentation reads only the recovered aggregate result table.
import sys
import shutil
sys.path.insert(0,str(root/'Code'))
from pricing_presentation import information_figures
for flavor in ['plain','berry']:
 target=asset/'empirical_information'/flavor;target.mkdir(parents=True,exist_ok=True)
 frame=pd.read_csv(src/'pricing'/flavor/'information_friction_aggregate.csv')
 information_figures(frame,target,delta=None)
 for old,new in [('profits','revenues'),('information_value','information_value'),
                 ('price_paths','first_prices'),('price_paths','terminal_prices')]:
  for extension in ['pdf','png']:
   shutil.copyfile(target/f'information_friction_{old}.{extension}', target/f'{new}.{extension}')
 rows=[r'Prior & Initial state & Personalized pricing & Common pricing & Difference\\\midrule']
 for scenario in frame.scenario.unique():
  for state in ['0','j']:
   values=frame[(frame.scenario==scenario)&(frame.h0.astype(str)==state)].set_index('case')
   personal=values.loc['known','persistent_type_revenue'];common=values.loc['unknown','persistent_type_revenue']
   rows.append(f"{scenario} & ${state}$ & {personal:,.2f} & {common:,.2f} & {personal-common:.2f} " + r"\\")
 table(target/'results.tex',flavor.capitalize()+': pricing comparison','empirical-information-'+flavor,'llrrr',rows)
