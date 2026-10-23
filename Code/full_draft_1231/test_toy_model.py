"""Economic accounting, first-order conditions, and independent solver checks."""
import unittest
import numpy as np
from scipy.optimize import minimize
import toy_model as tm

class PricingChecks(unittest.TestCase):
    def test_full_gradient(self):
        x=np.array([1.1,1.5,2.2,1.4,2.5,1.8]);types=[2.,1.];weights=[.4,.6]
        for state in tm.STATES:
            _,grad=tm.state_objective(x,types,weights,state)
            numerical=[]
            for i in range(6):
                eps=np.zeros(6);eps[i]=1e-6
                numerical.append((tm.state_objective(x+eps,types,weights,state)[0]-tm.state_objective(x-eps,types,weights,state)[0])/2e-6)
            np.testing.assert_allclose(grad,numerical,atol=3e-10)

    def test_initial_switching_and_no_history(self):
        p=[2.,2.]
        for state in [0,1]:
            low=tm.shares_jac(p,1.,state)[0];high=tm.shares_jac(p,2,state)[0]
            self.assertGreater(high[1-state],low[1-state])
        np.testing.assert_array_equal(tm.shares_jac(p,1.,None)[0],tm.shares_jac(p,2,None)[0])

    def test_accounting_and_exit(self):
        p=tm.solve_common([2.,1.],[.5,.5]);c=tm.components(p.p1,p.p2,[2.,1.])
        np.testing.assert_allclose(c[:,2],c[:,0]+tm.delta*c[:,1])
        for i,gamma in enumerate([2.,1.]):
            expected=0.;survival=0.
            for a,state in enumerate(tm.STATES):
                s1,_=tm.shares_jac(p.p1[a],gamma,state)
                survival+=tm.INITIAL_WEIGHTS[a]*s1.sum()
                for h in range(2):
                    s2,_=tm.shares_jac(p.p2[a,h],gamma,h)
                    expected+=tm.INITIAL_WEIGHTS[a]*s1[h]*(p.p1[a,h]-tm.costs[h]+tm.delta*(s2@(p.p2[a,h]-tm.costs)))
            self.assertAlmostEqual(c[i,2],expected,places=12)
            self.assertAlmostEqual(c[i,3],survival,places=12)
            self.assertLess(survival,1)

    def test_known_type_price_conditions(self):
        for gamma in [1.,1.5,2.,10.]:
            p=tm.solve_common([gamma],[1])
            for a,state in enumerate(tm.STATES):
                s1,_=tm.shares_jac(p.p1[a],gamma,state);w=[]
                for h in range(2):
                    s2,_=tm.shares_jac(p.p2[a,h],gamma,h);m=p.p2[a,h]-tm.costs
                    np.testing.assert_allclose(m,-1/(tm.alpha*(1-s2.sum())),atol=2e-5)
                    w.append(s2@m)
                np.testing.assert_allclose(p.p1[a]-tm.costs+tm.delta*np.array(w),-1/(tm.alpha*(1-s1.sum())),atol=2e-5)

    def test_information_dominance_and_independent_optimizer(self):
        types=[2.,1.];personal=[tm.solve_common([g],[1]) for g in types]
        for q in [0,.25,.5,.75,1]:
            p=tm.solve_common(types,[q,1-q])
            self.assertLessEqual(p.profit,q*personal[0].profit+(1-q)*personal[1].profit+1e-8)
            for i in range(2):self.assertLessEqual(p.type_profits[i],personal[i].profit+1e-8)
            for a,state in enumerate(tm.STATES):
                x=np.r_[p.p1[a],p.p2[a].ravel()]
                val,grad=tm.state_objective(x,types,[q,1-q],state)
                self.assertLess(np.max(np.abs(grad)),2e-6)
                independent=minimize(lambda y:-tm.state_objective(y,types,[q,1-q],state)[0],np.tile(tm.costs+2,3),method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-8,'fatol':1e-11})
                self.assertGreaterEqual(val+1e-7,-independent.fun)

    def test_invalid_inputs(self):
        for weights in [[.5,.6],[-.1,1.1],[np.nan,1]]:
            with self.assertRaises(ValueError):tm.solve_common([2.,1.],weights)

if __name__=='__main__':unittest.main()
