"""Pure numerical checks: no empirical inputs, records or panel files."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from demand_models import names, draws, objective, SPECS
from cross_price import fit_iv
from types import SimpleNamespace
from unittest.mock import patch
import cross_price
import estimate


class Models(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(11)
        return dict(outside=True, control=True, n_households=4, hh=np.repeat(np.arange(4),3),
            prices=rng.uniform(.5,2,(12,3)), novelty=rng.uniform(0,1,(12,3)),
            stock=rng.uniform(0,2,(12,3)), cf=rng.normal(size=(12,3)),
            counts=rng.multinomial(7,[.25]*4,size=12), available=np.ones((12,3),bool))

    def test_scores_and_specifications(self):
        flavors=['other','berry','plain']
        for spec in SPECS:
            parameters=names(spec,flavors,True)
            theta=np.linspace(-.2,.4,len(parameters))
            integration=draws(8,12,spec,flavors)
            data=self.data()
            value,score=objective(theta,data,integration,spec,flavors)
            numeric=approx_derivative(lambda t:objective(t,data,integration,spec,flavors)[0],theta).ravel()
            np.testing.assert_allclose(score,numeric,atol=1e-7,rtol=1e-6)
            if spec=='novelty_only':
                self.assertFalse(any('own_stock' in n for n in parameters))
                changed=dict(data,stock=data['stock']+100)
                self.assertAlmostEqual(value,objective(theta,changed,integration,spec,flavors)[0])
            else:
                self.assertNotIn('sd_log_minus_alpha',parameters)
                self.assertTrue(all('sd_beta_'+f in parameters for f in flavors))

    def test_empirical_entrypoints_fail_closed(self):
        with patch('server_data_policy.sys.platform', 'darwin'):
            for function in [estimate.run, cross_price.run, cross_price.flavor_instruments]:
                with self.assertRaises(PermissionError):
                    if function is cross_price.flavor_instruments:
                        function(SimpleNamespace(), None)
                    else:
                        function(SimpleNamespace())

    def test_iv_recovers_known_coefficients(self):
        rng=np.random.default_rng(20)
        z=rng.normal(size=(500,2))
        x=z@np.array([[1.,.3],[.2,1.]])+rng.normal(size=(500,2))
        error=rng.normal(size=500)
        error-=z@np.linalg.lstsq(z,error,rcond=None)[0]
        y=x@np.array([-2.,1.])+error
        fit=fit_iv(y,x,z,np.column_stack([np.arange(500)//10,np.arange(500)%20]))
        np.testing.assert_allclose(fit.params,[-2.,1.],atol=1e-12)
        self.assertEqual(len(fit.first_stage.diagnostics),2)


if __name__=='__main__':
    unittest.main()
