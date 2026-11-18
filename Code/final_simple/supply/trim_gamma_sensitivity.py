"""Distribution sensitivity from population estimates and aggregate initial counts."""
from pathlib import Path
from statistics import NormalDist
import argparse
import hashlib
import json
import math
import shutil

import numpy as np
import pandas as pd

from category_report import build
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


def run(old_estimates, binary_results, output):
    old_estimates, binary_results, output = map(Path,[old_estimates,binary_results,output])
    if output.exists():
        raise FileExistsError('Use a fresh output directory; preserve existing results.')
    parameters = pd.read_csv(old_estimates)
    if set(parameters)-{'parameter','estimate','at_bound'}:
        raise ValueError('Expected population parameter estimates only.')
    audit=json.loads((binary_results/'sample_audit.json').read_text())
    if not {'initial_inside','initial_outside','fitted_households'} <= set(audit):
        raise ValueError('Expected aggregate observed-week initial counts.')
    fit=json.loads((old_estimates.parent/'fit.json').read_text())
    if fit.get('spec') != 'novelty_only' or fit.get('habit') is not False or not fit.get('converged'):
        raise ValueError('Expected the old converged no-habit flavor-switching fit.')
    output.mkdir(parents=True)
    summary=[]
    settings=[]
    scenarios=[('old_uncut','Old gamma: full distribution',old_estimates,(0.,1.)),
               ('old_trimmed_05_95','Old gamma: middle 90%',old_estimates,(.05,.95)),
               ('binary_reference','New binary gamma',binary_results/'point_estimates.csv',(0.,1.))]
    for name,label,estimates,quantiles in scenarios:
        folder=output/name;folder.mkdir()
        shutil.copyfile(estimates,folder/'point_estimates.csv')
        build(folder,audit,gamma_quantiles=quantiles,label=label)
        p=pd.read_csv(estimates).set_index('parameter').estimate
        lower,upper=quantiles
        mu,sd=float(p.mean_novelty),float(p.sd_novelty)
        normal=NormalDist(mu,sd)
        lo=normal.inv_cdf(lower) if lower else None
        hi=normal.inv_cdf(upper) if upper<1 else None
        a=NormalDist().inv_cdf(lower) if lower else -math.inf
        b=NormalDist().inv_cdf(upper) if upper<1 else math.inf
        phi=lambda z:0. if not math.isfinite(z) else math.exp(-z*z/2)/math.sqrt(2*math.pi)
        mass=upper-lower
        shift=(phi(a)-phi(b))/mass
        variance=1+((a*phi(a) if math.isfinite(a) else 0.)-
                    (b*phi(b) if math.isfinite(b) else 0.))/mass-shift**2
        provenance=dict(scenario=name,label=label,source=str(estimates.resolve()),
            parameter_sha256=hashlib.sha256(estimates.read_bytes()).hexdigest(),
            gamma_source='flavor-history estimate used as category-history calibration'
                if name.startswith('old_') else 'new category-history estimate',
            source_mean=mu,source_sd=sd,retained_mean=mu+sd*shift,retained_sd=sd*math.sqrt(variance),
            lower_percentile=100*lower,upper_percentile=100*upper,
            gamma_lower_cutoff=lo,gamma_upper_cutoff=hi,retained_original_mass=mass,
            tails='removed and renormalized, not capped' if lower else 'unmodified',
            quintiles='five equal-probability groups of retained distribution',
            initialization='same aggregate observed-shopping-week counts from binary_est for all scenarios',
            included_households=audit['fitted_households'],
            actual_households_removed=False,demand_refitted=False,learning=False,
            price_floor=0.,subsidies_allowed=False,
            other_coefficients='old fit, identical across old_uncut and old_trimmed_05_95'
                if name.startswith('old_') else 'binary fit; reference only, not the isolated trim comparison')
        (folder/'sensitivity_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        settings.append(provenance)
        for mode in ['common_price','separate_flavor_prices']:
            changes=pd.read_csv(folder/mode/'observed_initialization_percentage_changes.csv')
            for row in changes.itertuples():
                summary.append(dict(scenario=name,label=label,mode=mode,metric=row.metric,
                                    informed_vs_uniform_pct=row.change_pct))
    pd.DataFrame(summary).to_csv(output/'sensitivity_comparison.csv',index=False)
    pd.DataFrame([{key:p[key] for key in ['scenario','source_mean','source_sd','retained_mean',
        'retained_sd','gamma_lower_cutoff','gamma_upper_cutoff','retained_original_mass']}
        for p in settings]).to_csv(output/'distribution_comparison.csv',index=False)
    with PdfPages(output/'comparison.pdf') as packet:
        fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
        grid=np.linspace(-7,6,1600)
        colors=['#777777','#2E6896','#208475']
        for p,color in zip(settings,colors):
            density=np.exp(-.5*((grid-p['source_mean'])/p['source_sd'])**2)/(p['source_sd']*np.sqrt(2*np.pi))
            if p['gamma_lower_cutoff'] is not None:
                density=np.where((grid>=p['gamma_lower_cutoff'])&(grid<=p['gamma_upper_cutoff']),
                                 density/p['retained_original_mass'],0.)
            ax.plot(grid,density,label=p['label'],color=color,linewidth=2)
        ax.set(title='Gamma distribution sensitivity',xlabel='Gamma',ylabel='Density')
        ax.legend();ax.grid(alpha=.2);packet.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
        data=pd.DataFrame(summary)
        data=data[data['mode']=='common_price']
        metrics=['consumer_surplus','revenue','total_welfare']
        for offset,p,color in zip([-.25,0,.25],settings,colors):
            values=data[data.scenario==p['scenario']].set_index('metric').loc[metrics,'informed_vs_uniform_pct']
            ax.bar(np.arange(3)+offset,values,width=.23,label=p['label'],color=color)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(title='Informed versus Uniform, observed initial-state weights',xticks=range(3),
               xticklabels=['Consumer surplus','Revenue','Total welfare'],
               ylabel='Change / |Uniform baseline| (%)')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100));ax.legend();ax.grid(axis='y',alpha=.2)
        packet.savefig(fig);plt.close(fig)
    (output/'README.md').write_text(
        '# Old flavor-switching gamma: 5th–95th percentile sensitivity\n\n'
        'Open comparison.pdf for distribution and aggregate information-gain comparisons. '
        'Open old_trimmed_05_95/common_price/pricing_results.pdf for the new pricing and welfare figures. '
        'Separate flavor-price outputs have their own folder under each scenario.\n\n'
        'The primary comparison is old_uncut versus old_trimmed_05_95: alpha, all flavor betas, '
        'initial-state weights, no-learning policy and solver are identical. Only the gamma '
        'distribution changes. Both apply the old flavor-history coefficient as a category-history '
        'calibration; no demand fit is rerun. binary_reference uses all new binary-fit coefficients '
        'and is a separate reference, not the isolated effect of trimming.\n\n'
        'Trimming discards the old distribution below its 5th percentile and above its 95th '
        'percentile and renormalizes the middle 90%. It does not cap observations. Each new '
        'quintile has 20% of retained mass (18% of original mass). All scenarios use the same '
        'aggregate first-observed-shopping-week counts from binary_est. No individual households '
        'or research records are read, removed, or reclassified locally.\n')
    print(pd.DataFrame(summary).query("mode == 'common_price'").to_string(index=False))
    print('Created '+str(output/'comparison.pdf'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-estimates',type=Path,required=True)
    parser.add_argument('--binary-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.old_estimates,args.binary_results,args.output)
