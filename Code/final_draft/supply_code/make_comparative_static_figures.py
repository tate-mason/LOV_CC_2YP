"""Re-solve both FOCs at each gamma; save reproducible figures and source data."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.numerical import solve

OUT = Path('figures')
PARAMETERS = dict(alpha=-2., beta=-1., delta=.7, c=1.)
CASES = {'(0,0)':(0,0), '(1,0)':(1,0), '(0,1)':(0,1)}


def main():
    OUT.mkdir(exist_ok=True)
    grid=np.linspace(0,3,201)
    curves={}
    rows=[]
    for label,(z1,z2) in CASES.items():
        result=np.array([solve(**PARAMETERS,gamma=g,z1=z1,z2=z2) for g in grid])
        p1,p2,s1,s2,profit=result.T
        a,d=PARAMETERS['alpha'],PARAMETERS['delta']
        derivative=(-p2*z1*s1+z2*s2*(p1+d*(1-s1)*p2))/(a*p2**2)
        curves[label]=(p1,p2,s1,s2,profit,derivative)
        rows.extend(zip([label]*len(grid),grid,p1,p2,s1,s2,profit,p1/p2,derivative))
    with (OUT/'solutions.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['history','gamma','p1','p2','s1','s2','profit','ratio','ratio_derivative'])
        writer.writerows(rows)
    specs=[('profit_by_gamma',4,'Optimized expected profit'),
           ('price_ratio_by_gamma',None,'Optimal price ratio p1/p2'),
           ('price_ratio_derivative_cases',5,'d(p1/p2)/d gamma')]
    for name,index,ylabel in specs:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for label,values in curves.items():
            y=values[0]/values[1] if index is None else values[index]
            ax.plot(grid,y,label=label,linewidth=2)
        if index==5:
            ax.axhline(0,color='black',linewidth=.7)
        ax.set(xlabel='Switching preference gamma',ylabel=ylabel)
        ax.set_title('Solutions of both logit pricing FOCs')
        ax.legend(title='(z1,z2)')
        ax.grid(alpha=.2)
        fig.text(.5,.01,'alpha=-2, beta=-1, delta=0.7, c=1; shares adjust with prices',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ('png','pdf'):
            fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    print(f'Saved three figures (PNG/PDF) and solutions.csv in {OUT}/')

if __name__=='__main__':
    main()
