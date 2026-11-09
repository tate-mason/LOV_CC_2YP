"""Check information scaling, correlation, and invalid-inference handling."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from demand.estimation import observed_covariance, hessian_standard_errors


class HessianChecks(unittest.TestCase):
    def test_correlated_information_and_household_scaling(self):
        information = np.array([[4., 1.], [1., 2.]])
        theta = np.array([.2, -.1])
        gradient = lambda x: information @ (x - theta)
        covariance = observed_covariance(theta, gradient, 100, [(-5, 5)] * 2)
        np.testing.assert_allclose(covariance, np.linalg.inv(100 * information), rtol=1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            table, report = hessian_standard_errors(theta, gradient, 100,
                [(-5, 5)] * 2, ['a', 'b'], Path(tmp))
            np.testing.assert_allclose(table.std_error, np.sqrt(np.diag(covariance)))
            self.assertTrue((table.ci_lower < theta).all())
            self.assertTrue((table.ci_upper > theta).all())
            self.assertTrue((Path(tmp) / 'covariance.csv').exists())
            self.assertEqual(report['first_stage_uncertainty'], 'not included')

    def test_boundary_and_indefinite_information_withheld(self):
        with self.assertRaisesRegex(ValueError, 'bounds'):
            observed_covariance(np.array([0.]), lambda x: x, 10, [(0, 3)])
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            observed_covariance(np.array([1.]), lambda x: -x, 10, [(-3, 3)])
        with self.assertRaisesRegex(ValueError, 'ill-conditioned'):
            observed_covariance(np.array([1., 1.]), lambda x: np.array([x[0], 0.]),
                                10, [(-3, 3)] * 2)
