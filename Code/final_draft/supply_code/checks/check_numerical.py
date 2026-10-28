import numpy as np
import sympy as sp
from model.symbols import *
from model.numerical import solve
from model.shares import sj1_expr,sj2_expr
from comparative_statics.gamma_p1 import dp1_dgamma
from comparative_statics.gamma_p2 import dp2_dgamma
from comparative_statics.beta_p1 import dp1_dbeta
from comparative_statics.beta_p2 import dp2_dbeta
from comparative_statics.alpha_p1 import dp1_dalpha
from comparative_statics.delta_p1 import dp1_ddelta
from comparative_statics.gamma_profit import dPi_dgamma
from comparative_statics.gamma_p1beta import dp1dbeta_dgamma_clean
from comparative_statics.gamma_p1p2 import ratio_derivative

profit = sj1_expr*(p1-c+delta*sj2_expr*(p2-c))
hessian = sp.lambdify((alpha,beta,gamma,delta,c,z1,z2,p1,p2),sp.hessian(profit,(p1,p2)),'numpy')
count=0
for a,b,d,cost in [(-2.,-1.,.7,1.),(-1.,.2,.4,1.),(-3.,1.,.9,.5)]:
    for history in [(0,0),(1,0),(0,1)]:
        for g in [0.,.7,1.8]:
            pars=dict(alpha=a,beta=b,gamma=g,delta=d,c=cost,z1=history[0],z2=history[1])
            x,y,s,t,pi=solve(**pars)
            sub={alpha:a,beta:b,gamma:g,delta:d,c:cost,z1:history[0],z2:history[1],p1:x,p2:y,sj1:s,s01:1-s,sj2:t,s02:1-t}
            assert abs(1+a*(1-t)*(y-cost)) < 1e-10
            assert abs(1+a*(1-s)*(x-cost+d*t*(y-cost))) < 1e-10
            H=hessian(a,b,g,d,cost,*history,x,y)
            assert np.linalg.eigvalsh(H).max() < 0
            for name, formulas in [('gamma',[dp1_dgamma,dp2_dgamma]),('beta',[dp1_dbeta,dp2_dbeta]),('alpha',[dp1_dalpha,(1-alpha*p2*sj2)/alpha**2]),('delta',[dp1_ddelta,sp.Integer(0)])]:
                h=1e-5
                plus=solve(**(pars|{name:pars[name]+h}))
                minus=solve(**(pars|{name:pars[name]-h}))
                numerical=(np.array(plus[:2])-np.array(minus[:2]))/(2*h)
                expected=[float(f.subs(sub)) for f in formulas]
                assert np.allclose(numerical,expected,rtol=2e-5,atol=1e-8),(name,numerical,expected)
                if name=='gamma':
                    assert np.isclose((plus[4]-minus[4])/(2*h),float(dPi_dgamma.subs(sub)),rtol=2e-5,atol=1e-8)
                    dr=(plus[0]/plus[1]-minus[0]/minus[1])/(2*h)
                    assert np.isclose(dr,float(ratio_derivative.subs(sub)),rtol=2e-5,atol=1e-8)
                    def beta_slope(result):
                        return (d*result[3]*(1-result[2])-result[2])/a
                    assert np.isclose((beta_slope(plus)-beta_slope(minus))/(2*h),float(dp1dbeta_dgamma_clean.subs(sub)),rtol=2e-5,atol=1e-8)
            count+=1
print(f'Numerical FOCs, negative definite Hessians, and re-solved finite differences: PASS ({count} solutions)')
