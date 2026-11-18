"""Pure numerical checks with invented aggregate arrays; no empirical I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'demand'))
sys.path.insert(0,str(ROOT/'supply'))
from category_switching import objective, draws, pooled_prediction, logsumexp
from category_pricing import solve, outcomes, probabilities
from binary_welfare import evaluate_policy


class CategoryChecks(unittest.TestCase):
    def test_binary_likelihood_gradient(self):
        data=dict(hh=np.array([0,0,1,1]),n_households=2,
            prices=np.array([[1.,1.2,.8],[1.1,.9,1.3],[.8,1.4,1.],[1.3,1.1,.9]]),
            available=np.ones((4,3),bool),purchase=np.array([1.,0.,0.,1.]),
            history_sign=np.array([1.,-1.,-1.,1.]))
        theta=np.array([.4,-.7,.2,-.1,.3,-.4,.8])
        integration=draws(8,15)
        value,gradient=objective(theta,data,integration)
        finite=[]
        for j in range(len(theta)):
            step=np.zeros(len(theta));step[j]=1e-6
            finite.append((objective(theta+step,data,integration)[0]-
                           objective(theta-step,data,integration)[0])/2e-6)
        np.testing.assert_allclose(gradient,finite,atol=1e-7,rtol=1e-6)
        probability,participation=pooled_prediction(theta,data,integration,True)
        np.testing.assert_allclose(probability.sum(axis=1),1.)
        np.testing.assert_allclose(participation,1-probability[:,3])
        # Extra package counts cannot change this strictly binary likelihood.
        redundant=dict(data,yogurt_count=np.array([100.,0.,0.,200.]),outside_count=np.ones(4)*7)
        self.assertEqual(objective(theta,redundant,integration)[0],value)

    def test_exact_inclusive_value_aggregation_and_welfare(self):
        g=np.array([-1.8,.2,1.7]);beta=np.array([.3,-.4,.8]);alpha=-1.2
        effective=float(logsumexp(beta))
        for h0 in ['j','0']:
            full=outcomes(g,h0,alpha,beta,.9,np.full(3,1.1),np.full(3,1.8),np.full(3,.9))
            binary=evaluate_policy(g,h0,alpha,effective,.9,1.1,1.8,.9)
            for key in ['revenue','consumer_surplus','total_welfare','quantity1','quantity2']:
                np.testing.assert_allclose(full[key],binary[key],atol=1e-12)

    def test_common_and_separate_prices_and_vector_candidates(self):
        g=np.array([-1.8,.2,1.7]);w=np.array([.25,.5,.25]);beta=np.array([.3,-.4,.8]);alpha=-1.2
        random=np.random.default_rng(17)
        for h0 in ['j','0']:
            common=solve(g,w,h0,alpha,beta)
            separate=solve(g,w,h0,alpha,beta,mode='separate_flavor_prices')
            for key in ['p1','p2j','p20']:
                np.testing.assert_allclose(common[key],separate[key],atol=1e-12)
            self.assertLess(common['flavor_first_order_residual'],1e-6)
            for prices in random.uniform(-2.,6.,size=(250,3)):
                alternative=outcomes(g,h0,alpha,beta,1.,prices,common['p2j'],common['p20'])
                self.assertGreaterEqual(common['objective']+1e-9,w@alternative['revenue'])
        # No learning: terminal offers are identical under either initial history.
        inside=solve(g,w,'j',alpha,beta);outside=solve(g,w,'0',alpha,beta)
        for key in ['p2j','p20']:
            np.testing.assert_allclose(inside[key],outside[key],atol=1e-12)


if __name__ == '__main__':
    unittest.main()
