"""Build a standalone comparison packet from theory and aggregate fit settings."""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from quintile_theory import run
from plot_supply_bins import plot_bins


def build(output):
    root = Path(__file__).resolve().parents[3]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exercises = []
    theory = output/'01_pure_theory'
    run(theory, individual=False)
    exercises.append(('Pure theory: normal gamma, mean 0, SD 1.25', theory))
    sources = [
        ('earlier_fit', root/'Code/final_simple/results/quantified_theory_five_bins_integrated'),
        ('observed_week_128', root/'Code/final_draft/seminar_pres/assets/quantified_theory_observed_128')]
    for fit, source in sources:
        for flavor in ['plain', 'berry', 'other']:
            folder = output/f'{len(exercises)+1:02d}_{fit}_{flavor}'
            settings = json.loads((source/flavor/'settings.json').read_text())
            if fit == 'earlier_fit':
                folder.mkdir(parents=True, exist_ok=True)
                results = pd.read_csv(source/flavor/'type_results.csv')
                results.to_csv(folder/'type_results.csv', index=False)
                plot_bins(results, folder, individual=False)
                shutil.copyfile(source/flavor/'settings.json', folder/'settings.json')
            else:
                results = run(folder, mean=settings['gamma_mean'], sd=settings['gamma_sd'],
                    beta=settings['beta'], alpha=settings['alpha'], delta=settings['delta'],
                    exercise=f'Applied {fit}, {flavor}; aggregate fitted inputs', individual=False)
            (folder/'source_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
            exercises.append((f'Applied / {fit.replace("_", " ")} / {flavor}', folder))

    # Verify accounting and exactly one common offer per history in every exercise.
    ledger = []
    for title, folder in exercises:
        data = pd.read_csv(folder/'type_results.csv')
        np.testing.assert_allclose(data.groupby(['h0','regime','group']).weight.sum(), .2)
        np.testing.assert_allclose(data.total_welfare, data.consumer_surplus+data.revenue)
        np.testing.assert_allclose(data.expected_p2,
            data.quantity1*data.p2j+(1-data.quantity1)*data.p20)
        for _, common in data[data.regime == 'common'].groupby('h0'):
            assert all(common[key].nunique() == 1 for key in ['p1','p2j','p20'])
        ledger.append(dict(exercise=title, directory=folder.name, quintiles=5,
                           population_mass_per_quintile=.2, checks='passed'))
    pd.DataFrame(ledger).to_csv(output/'exercise_index.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    lines = ['# Supply exercise review', '',
        'Open **supply_review.pdf** for the complete packet. Each exercise has two pages: '
        'overlaid expected price paths, then CS/revenue/TS across all quintiles.', '',
        'Every quintile contains 20% of its population. Five informed paths and exactly one '
        'population-uniform path appear in each initial-history panel. The firm knows the '
        'quintile, with uncertainty within it. Uniform offers can depend on purchase history; '
        'their displayed E[p2] is averaged over the full population.', '',
        'Welfare bars are per-consumer conditional quintile means under each policy. '
        'TS = CS + revenue at zero cost. Gamma persists across periods. '
        'Beliefs remain fixed within the observed bin or full population.', '',
        'Pure theory assumes normal gamma (mean 0, SD 1.25), beta 2, alpha -1, delta 1. '
        'Applied exercises use the saved aggregate estimates. The corrected observed-week '
        'fit is recomputed with quintiles to match the earlier fit; no demand model is refitted.', '',
        'The old three-point-prior and quadrature display versions are superseded here by '
        'the matching quintile versions. Each folder includes PNG/PDF figures, numerical '
        'outcomes, uniform population paths, settings and source provenance.', '']
    with PdfPages(output/'supply_review.pdf') as packet:
        for title, folder in exercises:
            lines += [f'## {title}', '']
            for stem, label in [('expected_price_paths_by_bin','Price paths'),
                                ('welfare_all_bins','CS, revenue and TS')]:
                relative = f'{folder.name}/{stem}.png'
                lines += [f'![{label}]({relative})', '']
                fig, ax = plt.subplots(figsize=(13.5, 8.5))
                ax.imshow(plt.imread(folder/(stem+'.png')))
                ax.axis('off')
                fig.suptitle(title+' — '+label, fontsize=15)
                fig.tight_layout()
                packet.savefig(fig)
                plt.close(fig)
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Created {len(exercises)} exercises, two matching figures each: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
