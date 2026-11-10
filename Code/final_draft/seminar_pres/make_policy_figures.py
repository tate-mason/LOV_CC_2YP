"""Server-only policy figures from aggregate pricing tables."""
from pathlib import Path
import os
import sys
import argparse
_policy_root = next(p for p in Path(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only
os.environ.setdefault('MPLCONFIGDIR', '/tmp/seminar-policy-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

@server_only
def main(output):
    for flavor in ['plain', 'berry']:
        folder = output / flavor
        table = pd.read_csv(folder / 'targeted_revenue_by_switching_group.csv')
        sys.path.insert(0, str(_policy_root / 'Code'))
        from pricing_presentation import price_paths, POLICY_LABELS
        table = table.loc[table.policy != 'no_coupon'].copy()
        table['policy'] = table.policy.replace(POLICY_LABELS)
        price_paths(table, folder, flavor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/scratch/dtm63837/Kilts_Panel/plain_berry_pricing_ex'))
    main(parser.parse_args().output)
