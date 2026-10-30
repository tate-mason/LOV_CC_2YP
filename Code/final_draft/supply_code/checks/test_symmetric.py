"""Symbolic identities and independent reoptimized finite differences."""
import unittest
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
from model.symbols import *
from model.utility import raw_utilities, U1, U2
from model.shares import sj1_expr, sj2_expr
from model.profit import Pi
from pricing.period1_foc import foc1_collected
from pricing.period2_foc import foc2
from comparative_statics import canonical as C
from model.numerical import evaluate, inside_share
import structural_switching_reference as reference


class SymbolicChecks(unittest.TestCase):
    def equal(self,left,right):
        self.assertEqual(sp.simplify(left-right),0)

    def test_raw_normalization_and_both_states(self):
        for history in [-1,1]:
            a,b = raw_utilities(p1,history)
            self.equal(a-b,beta+alpha*p1+history*gamma)
        self.equal(U1,beta+alpha*p1+q*gamma)

    def test_focs_from_actual_profit(self):
        literal = Pi.subs(sj1,sj1_expr)
        expected = foc1_collected.subs(sj1,sj1_expr)
        self.equal(sp.diff(literal,p1),expected)
        self.equal(sp.diff((p2-c)*sj2_expr,p2),foc2.subs(sj2,sj2_expr))
        self.equal(sp.diff(literal,Vj),delta*sj1_expr)
        self.equal(sp.diff(literal,V0),delta*(1-sj1_expr))
        self.equal(sp.diff(Vj,p1),0)

    def test_envelope_and_implicit_gamma_derivatives(self):
        self.equal(C.dp1_dgamma,-(q*sj1+delta*(1-sj1)*(u+v))/alpha)
        self.equal(C.dPi_dgamma,(-q*sj1+delta*(sj1*u-(1-sj1)*v))/alpha)
        expected2 = -(sj1*(1-sj1)**2*(q-delta*(u+v))**2
                      +delta*(1-sj1)*(-u*(1-u)**2+v*(1-v)**2))/alpha
        self.equal(C.d2p1_dgamma2,expected2)
        self.equal(C.ds_dgamma,sj1*(1-sj1)**2*(q-delta*(u+v)))
        self.equal(C.dVj_dgamma,u/alpha)
        self.equal(C.dV0_dgamma,-v/alpha)
        # Differentiate implicit pricing equations with endogenous shares.
        F = p1-c+1/(alpha*(1-sj1))+delta*(Vj-V0)
        self.equal(C.total_derivative(F,gamma),0)
        self.equal(sp.diff(p2-c+1/(alpha*(1-sj2)),p2)
                   +sp.diff(p2-c+1/(alpha*(1-sj2)),sj2)*alpha*sj2*(1-sj2),1/(1-sj2))

    def test_all_parameter_implicit_derivatives(self):
        F = p1-c+1/(alpha*(1-sj1))+delta*(Vj-V0)
        for theta in [alpha,beta,delta,c]:
            self.equal(C.total_derivative(F,theta),0)


class NumericalChecks(unittest.TestCase):
    def test_reference_on_entire_benchmark_grid(self):
        for g in np.linspace(-3,3,601):
            for state in ['j','0']:
                ours, original = evaluate(float(g),state),reference.evaluate(float(g),state)
                for key in ['p1','p2j','p20','profit','dp1','d2p1','dprofit','d_matched_ratio']:
                    self.assertAlmostEqual(ours[key],original[key],places=9)
                self.assertLess(max(abs(ours[k]) for k in ['foc1','foc2j','foc20']),1e-10)

    def test_independent_reoptimization_and_derivatives(self):
        h=1e-4
        for parameters in [dict(beta=2,alpha=-1,delta=.92,c=0),
                           dict(beta=-1,alpha=-2,delta=.7,c=1),
                           dict(beta=1.5,alpha=-.7,delta=.4,c=.3)]:
            for g in [-2.4,-1.2,0,.7,2.6]:
                for state in ['j','0']:
                    r=evaluate(g,state,**parameters)
                    plus,minus=evaluate(g+h,state,**parameters),evaluate(g-h,state,**parameters)
                    for derivative,level in [('dp1','p1'),('dprofit','profit'),('dp2j','p2j'),('dp20','p20')]:
                        self.assertAlmostEqual(r[derivative],(plus[level]-minus[level])/(2*h),delta=3e-7)
                    self.assertAlmostEqual(r['d2p1'],(plus['dp1']-minus['dp1'])/(2*h),delta=3e-7)
                    self.assertAlmostEqual(r['d2p1'],(plus['p1']-2*r['p1']+minus['p1'])/h**2,delta=3e-5)
                    denominator='p2j' if state=='j' else 'p20'
                    dr=(plus['p1']/plus[denominator]-minus['p1']/minus[denominator])/(2*h)
                    self.assertAlmostEqual(r['d_matched_ratio'],dr,delta=3e-7)
                    if state=='0':
                        self.assertAlmostEqual(r['d_induce_ratio'],(plus['p1']/plus['p2j']-minus['p1']/minus['p2j'])/(2*h),delta=3e-7)
                    self.assertAlmostEqual(r['p1'],r['static_component']+r['continuation_adjustment'],places=10)
                    # Independent optimizer of the complete two-branch profit.
                    def negative_profit(price):
                        share=inside_share(price,g,state,parameters['beta'],parameters['alpha'])
                        return -(share*(price-parameters['c'])+parameters['delta']*(share*r['V2j']+(1-share)*r['V20']))
                    optimum=minimize_scalar(negative_profit,bounds=(r['p1']-10,r['p1']+10),method='bounded')
                    self.assertAlmostEqual(optimum.x,r['p1'],delta=1e-5)
                    self.assertLess(negative_profit(r['p1']),(negative_profit(r['p1']+h)+negative_profit(r['p1']-h))/2)

    def test_all_parameter_and_mixed_derivatives_numerically(self):
        parameters=dict(beta=2.,alpha=-1.,delta=.92,c=0.)
        arguments=(alpha,beta,gamma,delta,c,q,sj1,u,v,p1,p2j,p20,Vj,V0)
        def arguments_at(result):
            return (-1.,2.,result['gamma'],.92,0.,-1 if result['initial_state']=='j' else 1,
                    result['s'],result['u'],result['v'],result['p1'],result['p2j'],result['p20'],result['V2j'],result['V20'])
        for theta,name in [(alpha,'alpha'),(beta,'beta'),(delta,'delta'),(c,'c')]:
            expression=C.price1_derivative(theta)
            terminal=C.price2_derivative(theta,u,-1,p2j)
            evaluator=sp.lambdify(arguments,[expression,terminal],'numpy')
            for state in ['j','0']:
                for g in [-1.,1.]:
                    result=evaluate(g,state,**parameters)
                    h=1e-5
                    plus_parameters=dict(parameters);plus_parameters[name]+=h
                    minus_parameters=dict(parameters);minus_parameters[name]-=h
                    plus=evaluate(g,state,**plus_parameters);minus=evaluate(g,state,**minus_parameters)
                    analytical=evaluator(*arguments_at(result))
                    self.assertAlmostEqual(float(analytical[0]),(plus['p1']-minus['p1'])/(2*h),delta=2e-6)
                    self.assertAlmostEqual(float(analytical[1]),(plus['p2j']-minus['p2j'])/(2*h),delta=2e-6)
        mixed=sp.lambdify(arguments,C.total_derivative(C.price1_derivative(beta),gamma),'numpy')
        slope=sp.lambdify(arguments,C.price1_derivative(beta),'numpy')
        for state in ['j','0']:
            g=.7;h=1e-4
            actual=mixed(*arguments_at(evaluate(g,state)))
            expected=(slope(*arguments_at(evaluate(g+h,state)))-slope(*arguments_at(evaluate(g-h,state))))/(2*h)
            self.assertAlmostEqual(float(actual),float(expected),delta=2e-6)

    def test_zero_switching_static_case_and_symmetry(self):
        for state in ['j','0']:
            r=evaluate(0,state)
            self.assertAlmostEqual(r['p1'],r['p2j'])
            self.assertAlmostEqual(r['p2j'],r['p20'])
            static=evaluate(2,state,delta=0)
            self.assertAlmostEqual(static['p1'],static['static_counterfactual'])
        for g in [-3,-1,1,3]:
            self.assertAlmostEqual(evaluate(g,'j')['p2j'],evaluate(-g,'0')['p20'])
        # Large negative gamma permits introductory subsidies; no price>=cost clamp.
        self.assertLess(evaluate(-8,'0')['p1'],0)


if __name__=='__main__':
    unittest.main()
