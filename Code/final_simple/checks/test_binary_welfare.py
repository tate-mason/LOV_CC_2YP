"""Independent welfare checks using pure synthetic theory inputs only."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit, xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demand'))
from binary_welfare import evaluate_policy
from quantified_theory import compare
from gamma_calibration import known_prices
from scipy.special import wrightomega


class WelfareChecks(unittest.TestCase):
    def test_total_comparative_statics(self):
        inside_price_signs, inside_profit_signs = set(), set()
        for beta in [-2., 0., 2., 5.]:
            for g in [-3., -.4, 0., .8, 3.]:
                for h in ['j','0']:
                    a, d, eps = 1.3, .9, 1e-4
                    r = known_prices(g,h,-a,beta,d)
                    sj,s0 = expit(beta+np.array([-g,g])-(1+wrightomega(beta+np.array([-g,g])-1)))
                    s1 = expit(beta+(-g if h=='j' else g)-a*r['expected_p1'])
                    q = -1 if h=='j' else 1
                    dp = (q*s1+d*(1-s1)*(sj+s0))/a
                    dpi = (q*s1-d*s1*sj+d*(1-s1)*s0)/a
                    low, high = [known_prices(g+change,h,-a,beta,d) for change in [-eps,eps]]
                    self.assertAlmostEqual(dp,(high['expected_p1']-low['expected_p1'])/(2*eps),places=7)
                    self.assertAlmostEqual(dpi,(high['profit']-low['profit'])/(2*eps),places=7)
                    if h=='0':
                        self.assertGreater(dp,0.)
                        self.assertGreater(dpi,0.)
                    else:
                        inside_price_signs.add(int(np.sign(dp)))
                        inside_profit_signs.add(int(np.sign(dpi)))
        self.assertTrue({-1,1} <= inside_price_signs)
        self.assertTrue({-1,1} <= inside_profit_signs)

    def test_static_price_integral(self):
        # CS changes must equal the area under demand at fixed history.
        for gamma in [-1.7, 0., 1.2]:
            for h in ['j','0']:
                args = (gamma, h, -1.3, 2., 0.)
                low = evaluate_policy(*args, .6, 1., 2.)
                high = evaluate_policy(*args, 2.4, 1., 2.)
                integral = quad(lambda p: expit(2.-1.3*p+(-gamma if h=='j' else gamma)), .6, 2.4)[0]
                self.assertAlmostEqual(low['consumer_surplus']-high['consumer_surplus'], integral, places=11)

    def test_direct_expected_utility_and_transfers(self):
        # Independent entropy representation of logit expected utility, using
        # absolute alternative utilities and enumerating the two histories.
        g, a, b, delta, p1, pj, p0, cost = 1.4, -1.2, 2.1, .85, -.3, 2.2, 1.1, .4
        def utility(h, p):
            uj = b+a*p+(g if h=='0' else 0.)
            u0 = g if h=='j' else 0.
            q = expit(uj-u0)
            entropy = -xlogy(q,q)-xlogy(1-q,1-q)
            return q, (q*uj+(1-q)*u0+entropy)/(-a)
        for h in ['j','0']:
            r = evaluate_policy(g,h,a,b,delta,p1,pj,p0,cost)
            q1, cs1 = utility(h,p1)
            qj, csj = utility('j',pj)
            q0, cs0 = utility('0',p0)
            cs = cs1+delta*(q1*csj+(1-q1)*cs0)
            profit = q1*(p1-cost)+delta*(q1*qj*(pj-cost)+(1-q1)*q0*(p0-cost))
            self.assertAlmostEqual(r['consumer_surplus'], cs, places=12)
            self.assertAlmostEqual(r['total_welfare'], cs+profit, places=12)
            self.assertAlmostEqual(r['revenue']-r['profit'], cost*r['quantity'], places=12)
            self.assertAlmostEqual(r['consumer_surplus']-r['zero_outside_consumer_surplus'],
                ((g if h=='j' else 0.)+delta*q1*g)/(-a), places=12)

    def test_no_heterogeneity_no_information_effect(self):
        for h in ['j','0']:
            g, a, b, d = np.array([.7,.7,.7]), -1., 2., 1.
            known, common, revenue = compare(g, np.array([.2,.3,.5]), a,b,d,h)
            pj,p0 = -(1+wrightomega(b+np.array([-.7,.7])-1))/a
            i = evaluate_policy(g,h,a,b,d,[r['expected_p1'] for r in known],pj,p0)
            c = evaluate_policy(g,h,a,b,d,common['expected_p1'],common['p2j'],common['p20'])
            for metric in ['consumer_surplus','profit','quantity','total_welfare']:
                np.testing.assert_allclose(i[metric], c[metric], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
