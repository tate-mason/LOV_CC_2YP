import unittest
import numpy as np
import pandas as pd
from demand.estimation import implied_wtp


class WTPChecks(unittest.TestCase):
    def table(self, **values):
        return pd.DataFrame(dict(parameter=list(values), estimate=list(values.values())))

    def test_fixed_alpha_and_signed_history(self):
        result = implied_wtp(self.table(mu_log_minus_alpha=np.log(2), beta_berry=-4.,
                                       mean_gamma=-2.), .7).set_index('term')
        self.assertAlmostEqual(result.loc['beta_berry', 'mean_wtp'], -2.)
        self.assertAlmostEqual(result.loc['gamma', 'mean_wtp'], -1.)
        self.assertAlmostEqual(result.loc['gamma_novelty', 'mean_wtp'], -.3)
        self.assertAlmostEqual(result.loc['gamma_own_stock', 'mean_wtp'], .7)
        np.testing.assert_allclose(result.population_sd, 0., atol=1e-7)

    def test_mixture_moments_include_between_component_variation(self):
        result = implied_wtp(self.table(mu_log_minus_alpha=0., sd_log_minus_alpha=.4,
            mean_gamma=-1., sd_gamma=.2, gamma_mean_gap=3., sd_gamma_2=.3,
            mixture_weight=.25)).set_index('term')
        mean = .25 * -1 + .75 * 2
        second = .25 * (1 + .2**2) + .75 * (4 + .3**2)
        expected = mean * np.exp(.4**2/2)
        self.assertAlmostEqual(result.loc['gamma', 'mean_wtp'], expected)
        self.assertAlmostEqual(result.loc['gamma', 'population_sd'],
                               np.sqrt(second * np.exp(2*.4**2) - expected**2))
        self.assertAlmostEqual(result.loc['gamma', 'mean_wtp'],
            .25 * result.loc['gamma_component_1', 'mean_wtp'] +
            .75 * result.loc['gamma_component_2', 'mean_wtp'])

    def test_separate_history_wtp_has_no_fixed_weight(self):
        table = self.table(mu_log_minus_alpha=np.log(2), mean_novelty=.6,
                           mean_own_stock=-.8, sd_own_stock=.4)
        result = implied_wtp(table, .1).set_index("term")
        self.assertAlmostEqual(result.loc["novelty", "mean_wtp"], .3)
        self.assertAlmostEqual(result.loc["own_stock", "mean_wtp"], -.4)
        self.assertAlmostEqual(result.loc["own_stock", "population_sd"], .2)
        pd.testing.assert_frame_equal(implied_wtp(table, .1), implied_wtp(table, .9))

    def test_beta_averages_ratios(self):
        result = implied_wtp(self.table(mu_log_minus_alpha=.2, sd_log_minus_alpha=.5,
                                       beta_plain=2.)).iloc[0]
        self.assertAlmostEqual(result.mean_wtp, 2 * np.exp(-.2 + .5**2/2))
        self.assertGreater(result.population_sd, 0.)


if __name__ == '__main__':
    unittest.main()
