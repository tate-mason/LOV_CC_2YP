"""Synthetic type-array checks; no empirical records or I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'supply'))
from entry_switching_model import solve,evaluate
from pricing_model import static_price


class EntrySwitchingChecks(unittest.TestCase):
    def test_repeat_utility_and_terminal_price_have_no_gamma_term(self):
        g=np.array([-3.,0.,2.])
        r=evaluate(g,'j',-1.2,.7,1.,.5,1.4,.8)
        np.testing.assert_allclose(r['quantity1'],r['quantity1'][0])
        np.testing.assert_allclose(r['consumer_surplus1'],r['consumer_surplus1'][0])
        outside=evaluate(g,'0',-1.2,.7,1.,.5,1.4,.8)
        self.assertTrue((np.diff(outside['quantity1'])>0).all())
        for w in [np.array([.2,.5,.3]),np.array([.7,.2,.1])]:
            for h0 in ['j','0']:
                s=solve(g,w,h0,-1.2,.7)
                self.assertAlmostEqual(s['p2j'],static_price(.7,-1.2),places=12)

    def test_revenue_dense_search_and_nonnegative_prices(self):
        g=np.array([-3.,-.4,1.8]);w=np.array([.2,.5,.3])
        for h0 in ['j','0']:
            s=solve(g,w,h0,-1.3,.7)
            self.assertTrue(all(s[key]>=0 for key in ['p1','p2j','p20']))
            self.assertLess(s['kkt_residual'],1e-6)
            grid=np.linspace(0,12,24001)
            out=evaluate(g[None,:],h0,-1.3,.7,1.,grid[:,None],s['p2j'],s['p20'])
            revenue=out['revenue']@w
            self.assertGreaterEqual(s['objective']+1e-10,revenue.max())
            np.testing.assert_allclose(out['total_welfare'],out['consumer_surplus']+out['revenue'])


if __name__=='__main__':
    unittest.main()
