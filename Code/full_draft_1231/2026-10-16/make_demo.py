"""Generate synthetic purchase data, a complete menu, and explicit price failures."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

import demand_core as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='demo_data')
    parser.add_argument('--households', type=int, default=30)
    args = parser.parse_args()
    if args.households < 2:
        parser.error('At least two households required.')
    folder = core.local_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        parser.error('Choose a fresh directory.')
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(219)
    flavors = ['PLAIN', 'STRAWBERRY', 'VANILLA']
    weeks = pd.date_range('2024-01-06', periods=5, freq='7D')
    prices = rng.uniform(.8, 2.2, (len(weeks), 3))
    menu = [{'store_code_uc': 1, 'week_end': str(w.date()), 'flavor': f,
             'upc': j+1, 'price': prices[t, j]} for t, w in enumerate(weeks) for j, f in enumerate(flavors)]
    purchases = []
    for h in range(args.households):
        beta = np.r_[0, rng.normal([.4, -.2], [.8, .8])]
        stock = np.zeros(3)
        for t, week in enumerate(weeks):
            stock *= .8 if t else 1
            counts = rng.multinomial(5, softmax(beta - .7*prices[t] - .5*np.log1p(stock)))
            for j, q in enumerate(counts):
                if q:
                    purchases.append({'household_code': h, 'store_code_uc': 1,
                                      'week_end': str(week.date()), 'flavor': flavors[j],
                                      'upc': j+1, 'quantity': q, 'price': prices[t, j],
                                      'line_spend': q*prices[t, j]})
            stock += counts
    clean = pd.DataFrame(purchases)
    clean.to_csv(folder / 'purchases.csv', index=False)
    pd.DataFrame(menu).to_csv(folder / 'menu.csv', index=False)
    dirty = clean.copy()
    dirty['price'] = dirty.price.astype(object)
    dirty.loc[:4, 'price'] = [None, 'bad', 0, -1, np.inf]
    dirty.to_csv(folder / 'price_failures.csv', index=False)
    (folder / 'truth.json').write_text('{"gamma": 0.5, "alpha": -0.7, "taste_sd": [0.8, 0.8], "delta": 0.8}\n')
    print(f'Synthetic data saved to {folder}')


if __name__ == '__main__':
    main()
