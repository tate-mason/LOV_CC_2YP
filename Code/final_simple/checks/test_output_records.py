"""Verify output records using captured DataFrames instead of reading CSV files."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from demand import estimation
from checks import test_est
from checks.test_price_control import synthetic_panel


class OutputRecordChecks(unittest.TestCase):
    def test_bootstrap_output_matches_captured_replicates(self):
        saved = {}
        def capture(table, filename, *args, **kwargs):
            saved[Path(filename).name] = table.copy()
        def estimate(sample, initial):
            theta = np.array([sample.cf_other.std(), sample.cf_berry.std()])
            return OptimizeResult(x=theta, fun=1., success=True, message='synthetic')
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(seed=306, bootstrap_reps=20,
                                       no_price_control=False, output=Path(directory))
            with patch.object(pd.DataFrame, 'to_csv', capture), \
                 patch.object(estimation, 'add_price_controls', wraps=estimation.add_price_controls) as first_stage, \
                 contextlib.redirect_stdout(io.StringIO()):
                table, report = estimation.bootstrap_standard_errors(
                    synthetic_panel(), settings, estimate, np.zeros(2), ['a', 'b'])
            self.assertEqual(first_stage.call_count, 20)
            self.assertEqual(report['usable'], 20)
            np.testing.assert_allclose(table.std_error,
                                       saved['bootstrap_draws.csv'][['a', 'b']].std(ddof=1))
            self.assertIn('covariance.csv', saved)

    def test_mixture_fit_writes_plain_text_and_compatible_metadata(self):
        saved = {}
        def capture(table, filename, *args, **kwargs):
            saved[Path(filename).name] = table.copy()
        data = estimation.make_arrays(estimation.add_stocks(test_est.EstimationChecks().frame()), 7)
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(model='mixture', draws=8, seed=306, starts=2,
                                       maxiter=3, output=Path(directory), se_method='none', capacity=7)
            with patch.object(pd.DataFrame, 'to_csv', capture), contextlib.redirect_stdout(io.StringIO()):
                estimation.fit_model(data, settings)
            self.assertIn('mixture_weight', saved['estimates.csv'].parameter.tolist())
            report = json.loads((settings.output / 'fit.json').read_text())
            self.assertEqual(report['parameters'], len(saved['estimates.csv']))
            self.assertAlmostEqual(report['aic'], 2*report['nll'] + 2*report['parameters'])
            self.assertTrue((settings.output / 'run_notes.txt').exists())
            self.assertTrue((settings.output / 'gamma_distribution.pdf').exists())


if __name__ == '__main__':
    unittest.main()
