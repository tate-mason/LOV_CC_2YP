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
        selected = table.loc[(table.policy == 'personalized_dynamic') & table.group.isin(['low', 'high'])].set_index('group').reindex(['low', 'high'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ['Low switching\n(bottom quartile)', 'High switching\n(top quartile)']
        for group, color, label in zip(['low', 'high'], ['#3575AD', '#D17925'], labels):
            row = selected.loc[group]
            axes[0].plot([1, 2], [100*row.discount1, 100*row.expected_discount2], 'o-', color=color, label=label.replace('\n', ' '))
        axes[0].set(xticks=[1, 2], xticklabels=['Initial offer', 'Expected terminal offer'], ylabel='Mean discount (% of posted price)', title=f'{flavor.title()}: offer path by switching type')
        axes[0].legend(fontsize=8)
        gains = selected.revenue_change / selected.eligible_households
        axes[1].bar(labels, gains, color=['#3575AD', '#D17925'])
        axes[1].set(ylabel='Discounted revenue gain per eligible household ($)', title='Gain relative to no coupons')
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.text(.5, .02, 'Terminal offers average over purchase histories. Type groups also differ in price sensitivity and habit.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, 1))
        for ext in ['pdf', 'png']:
            fig.savefig(folder / f'pricing_mechanism.{ext}', dpi=180)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/scratch/dtm63837/Kilts_Panel/plain_berry_pricing_ex'))
    main(parser.parse_args().output)
