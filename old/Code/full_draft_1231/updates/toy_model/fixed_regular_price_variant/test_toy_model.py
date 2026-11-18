"""Economic and independent numerical checks for the targeted-discount model."""
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
import toy_model as tm


class DiscountModelChecks(unittest.TestCase):
    def test_discount_derivative_against_finite_difference(self):
        for gamma in (.1, .8, 2.):
            for d in (.01,.2,.7,.99):
                eps=1e-6
                numerical=(tm.terminal_profit(d+eps,gamma)-tm.terminal_profit(d-eps,gamma))/(2*eps)
                self.assertAlmostEqual(numerical,tm.terminal_discount_gradient(d,gamma),places=8)

    def test_solver_against_independent_bounded_optimizer_and_grid(self):
        types=[tm.gamma_H,tm.gamma_L]
        for q in (0,.1,.3,.5,.55,.7,1):
            weights=np.array([q,1-q]);p=tm.solve_common(types,weights)
            objective=lambda d: weights@tm.evaluate_discount(d,types)
            independent=minimize_scalar(lambda d:-objective(d),bounds=(0,tm.DISCOUNT_MAX),method='bounded')
            best=max(objective(0),objective(tm.DISCOUNT_MAX),-independent.fun)
            self.assertAlmostEqual(p.profit,best,places=9)
            grid=np.linspace(0,tm.DISCOUNT_MAX,401)
            self.assertGreaterEqual(p.profit+1e-10,max(map(objective,grid)))
            grad=sum(w*tm.terminal_discount_gradient(p.discount,g) for g,w in zip(types,weights))
            if p.discount==0:self.assertLessEqual(grad,1e-9)
            elif p.discount==tm.DISCOUNT_MAX:self.assertGreaterEqual(grad,-1e-9)
            else:self.assertAlmostEqual(grad,0,places=9)

    def test_prices_fixed_and_coupon_only_on_switch_to_good2(self):
        for q in (0,.5,1):
            p=tm.solve_common([tm.gamma_H,tm.gamma_L],[q,1-q])
            np.testing.assert_array_equal(p.p1,tm.regular_prices)
            np.testing.assert_array_equal(p.p2[1],tm.regular_prices)
            self.assertEqual(p.p2[0,0],tm.regular_prices[0])
            self.assertEqual(p.p2[0,1],tm.regular_prices[1]-p.discount)

    def test_profit_accounting_and_outside_exit(self):
        for gamma in (.1,2.):
            d=.3; first,_=tm.logit_probs(tm.beta_0+tm.alpha*tm.regular_prices)
            expected=0.
            for h in (0,1):
                expected+=first[h]*(tm.regular_prices[h]-tm.costs[h])
                s,_=tm.shares(d,gamma,h)
                prices=tm.effective_prices(d if h==0 else 0)
                expected+=tm.delta*first[h]*(s@(prices-tm.costs))
            self.assertAlmostEqual(tm.evaluate_discount(d,[gamma])[0],expected,places=12)

    def test_targeting_dominance_and_mispricing_loss(self):
        types=[tm.gamma_H,tm.gamma_L]
        personal=[tm.solve_common([g],[1]) for g in types]
        prior_d=None
        for q in np.linspace(0,1,21):
            p=tm.solve_common(types,[q,1-q])
            observed=q*personal[0].profit+(1-q)*personal[1].profit
            self.assertGreaterEqual(observed+1e-10,p.profit)
            if prior_d is not None:self.assertLessEqual(p.discount,prior_d+1e-9)
            prior_d=p.discount
            for i in (0,1):self.assertLessEqual(p.type_profits[i],personal[i].profit+1e-10)
        self.assertEqual(personal[0].discount,0.)
        self.assertGreater(personal[1].discount,personal[0].discount)

    def test_switching_increase_preserves_margin_advantage(self):
        p=tm.solve_common([tm.gamma_L],[1])
        regular,_=tm.shares(0,tm.gamma_L,0);discounted,_=tm.shares(p.discount,tm.gamma_L,0)
        self.assertGreater(discounted[1],regular[1])
        margins=tm.effective_prices(p.discount)-tm.costs
        self.assertGreater(margins[1],margins[0])
        self.assertLess(tm.terminal_profit(p.discount,tm.gamma_H),tm.terminal_profit(0,tm.gamma_H))
        self.assertGreater(tm.terminal_profit(p.discount,tm.gamma_L),tm.terminal_profit(0,tm.gamma_L))

    def test_fixed_coupon_crossing_and_regions(self):
        types=[tm.gamma_H,tm.gamma_L]
        policies=[tm.solve_common([g],[1]) for g in types]+[tm.solve_common(types,[.5,.5])]
        values=np.array([tm.evaluate_discount(p.discount,types) for p in policies])
        regions=tm.fixed_menu_regions(values)
        self.assertEqual([r[2] for r in regions],[1,2,0])
        for (_,b,i),(_,_,j) in zip(regions[:-1],regions[1:]):
            self.assertAlmostEqual(values[i]@[b,1-b],values[j]@[b,1-b],places=12)

    def test_invalid_inputs(self):
        for d in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):tm.effective_prices(d)
        for weights in ([.5,.6],[-.1,1.1],[float('nan'),1.]):
            with self.assertRaises(ValueError):tm.solve_common([2.,.1],weights)


if __name__=='__main__':unittest.main()
