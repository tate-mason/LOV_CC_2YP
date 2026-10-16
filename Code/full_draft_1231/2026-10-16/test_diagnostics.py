"""Targeted checks for confounding diagnostics and raw price classification."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import demand_core as core
from diagnostics import design_diagnostics
from price_audit import key_id, status


class DiagnosticTests(unittest.TestCase):
    def design(self, log_exposure):
        counts = np.array([[2, 1], [1, 2], [2, 1], [1, 2]])
        weeks = pd.date_range('2024-01-06', periods=4, freq='7D').to_numpy()
        stock = np.column_stack([np.zeros(4), np.expm1(log_exposure)])
        panel = core.Panel('h', weeks, counts, np.column_stack([np.ones(4), [1, 2, 1, 2]]),
                           np.ones((4, 2), bool), np.zeros((4, 2, 0)), stock,
                           stock.copy(), np.ones(4, bool))
        data = core.Dataset([panel], ['a', 'b'], [], np.arange(2), .8, 0)
        with tempfile.TemporaryDirectory() as directory:
            result = design_diagnostics(data, Path(directory))
            variation = pd.read_csv(Path(directory) / 'household_variation.csv')
        return result, variation

    def test_constant_relative_history_is_confounded_with_beta(self):
        result, variation = self.design([1, 1, 1, 1])
        self.assertEqual(result['rank'], 2)
        self.assertLess(result['history_residual_rms'], 1e-10)
        self.assertEqual(variation.within_pair_history_sd.iloc[0], 0)

    def test_independent_history_restores_homogeneous_rank(self):
        result, variation = self.design([0, 0, 1, 1])
        self.assertEqual(result['rank'], 3)
        self.assertGreater(result['history_residual_rms'], .1)
        self.assertGreater(variation.within_pair_history_sd.iloc[0], 0)

    def test_price_failures_are_distinct(self):
        self.assertEqual(status(pd.Series([None, 'bad', np.inf, 0, -1, '1.5'])).tolist(),
                         ['missing', 'malformed', 'infinite', 'zero', 'negative', 'valid'])

    def test_key_normalization_keeps_missing_identifiers_missing(self):
        result = key_id(pd.Series(['00123', '123.0', None, ' 000 ']))
        self.assertEqual(result.iloc[0], result.iloc[1])
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertEqual(result.iloc[3], '0')


if __name__ == '__main__':
    unittest.main()
