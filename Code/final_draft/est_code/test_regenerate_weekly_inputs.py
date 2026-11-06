"""Recovery control flow tests with generated fixtures only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import regenerate_weekly_inputs as recovery


class RecoveryTests(unittest.TestCase):
    def check_recovery(self, mismatch):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fit = dict(model='heterogeneous_history', households=1, weeks=2, nll=3., settings=dict(
                sample='/server/synthetic.parquet', first_year=2006, last_year=2015,
                max_households=None, capacity=7, no_price_control=False,
                household_dma_column='dma_cd', habit_weight=.7, draws=4, seed=306))
            (directory/'fit.json').write_text(json.dumps(fit))
            names = recovery.parameter_names('heterogeneous_history', True, True)
            pd.DataFrame(dict(parameter=names, estimate=np.ones(len(names)))).to_csv(directory/'point_estimates.csv', index=False)
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            frame = pd.DataFrame({'synthetic_row': [1, 2]})
            data = dict(n_households=1, outside=True, control=True)
            with patch.object(recovery, 'prepare_data', return_value=frame) as prepare, \
                    patch.object(recovery, 'add_price_controls', return_value=(frame, None)) as controls, \
                    patch.object(recovery, 'make_arrays', return_value=data), \
                    patch.object(recovery, 'household_loglike', return_value=np.array([-4. if mismatch else -3.])):
                if mismatch:
                    (directory/'weekly_inputs.csv').write_text('existing export')
                    with self.assertRaisesRegex(ValueError, 'likelihood'):
                        recovery.regenerate(directory)
                    self.assertEqual((directory/'weekly_inputs.csv').read_text(), 'existing export')
                    self.assertFalse((directory/'weekly_inputs_regeneration.json').exists())
                else:
                    report = recovery.regenerate(directory)
                    self.assertFalse(report['demand_refitted'])
                    self.assertEqual(len(pd.read_csv(directory/'weekly_inputs.csv')), 2)
                controls.assert_called_once()
                self.assertEqual(prepare.call_args.args[0].sample, Path('/server/synthetic.parquet'))
            for name, content in before.items():
                self.assertEqual((directory/name).read_bytes(), content)

    def test_recovery_preserves_parameters(self):
        self.check_recovery(False)

    def test_mismatch_preserves_existing_export(self):
        self.check_recovery(True)


if __name__ == '__main__':
    unittest.main()
