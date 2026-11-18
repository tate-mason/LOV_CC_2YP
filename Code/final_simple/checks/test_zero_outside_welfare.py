"""Independent normalization checks with numerical type arrays only."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demand'))
from binary_welfare import evaluate_policy
from zero_outside_welfare import evaluate_zero_outside


class NormalizedWelfareChecks(unittest.TestCase):
    def test_choices_revenue_and_exact_level_difference(self):
        gamma=np.array([-3.,-.4,1.2])
        for h0 in ['j','0']:
            args=(gamma,h0,-1.3,.7,.9,.25,1.8,.8)
            old=evaluate_policy(*args);new=evaluate_zero_outside(*args)
            for key in ['p1','p2j','p20','expected_p2','quantity1','quantity2','revenue','profit']:
                np.testing.assert_array_equal(old[key],new[key])
            expected=gamma/1.3*((h0=='j')+.9*old['quantity1'])
            np.testing.assert_allclose(old['consumer_surplus']-new['consumer_surplus'],expected,atol=1e-12)
            self.assertTrue((new['consumer_surplus']>=0).all())
            np.testing.assert_allclose(new['total_welfare'],new['consumer_surplus']+new['profit'])

    def test_expected_utility_using_choice_probabilities_and_entropy(self):
        gamma=np.array([-2.,.1,1.6]);alpha=-1.2;beta=.6;delta=.9
        def utility(p,history):
            v=beta+alpha*p+(-gamma if history=='j' else gamma)
            q=1/(1+np.exp(-v))
            entropy=-q*np.log(q)-(1-q)*np.log1p(-q)
            return q,(q*v+entropy)/(-alpha)
        for h0 in ['j','0']:
            q,c1=utility(.4,h0);_,cj=utility(1.6,'j');_,c0=utility(.8,'0')
            result=evaluate_zero_outside(gamma,h0,alpha,beta,delta,.4,1.6,.8)
            np.testing.assert_allclose(result['consumer_surplus'],c1+delta*(q*cj+(1-q)*c0),atol=1e-12)


if __name__=='__main__':
    unittest.main()
