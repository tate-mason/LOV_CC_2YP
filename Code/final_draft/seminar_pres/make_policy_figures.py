"""Server-only policy figures from aggregate pricing tables."""
from pathlib import Path
import sys
import argparse
_policy_root = next(p for p in Path(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only
sys.path.insert(0, str(_policy_root / 'Code'))
from pricing_presentation import displayed_policies, price_paths, pricing_comparison_figures
import pandas as pd

@server_only
def main(output):
    for flavor in ['plain', 'berry']:
        folder = output / flavor
        table = pd.read_csv(folder / 'targeted_revenue_by_switching_group.csv')
        price_paths(displayed_policies(table), folder, flavor)
        pricing_comparison_figures(folder, folder, flavor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/scratch/dtm63837/Kilts_Panel/plain_berry_pricing_ex'))
    main(parser.parse_args().output)
