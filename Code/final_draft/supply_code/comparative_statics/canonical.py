"""All derivatives follow one implicit FOC and the terminal envelope."""
import sympy as sp
from model.symbols import *


def price2_derivative(theta, share=sj2, history=q2, price=p2):
    direct = sp.diff(beta, theta) + price*sp.diff(alpha, theta) + history*sp.diff(gamma, theta)
    return sp.simplify((1-share)*sp.diff(c, theta) + sp.diff(alpha, theta)/alpha**2 - share*direct/alpha)


def value2_derivative(theta, share, history, price):
    direct = sp.diff(beta, theta) + price*sp.diff(alpha, theta) + history*sp.diff(gamma, theta)
    return -share*direct/alpha - share*sp.diff(c, theta)


def price1_derivative(theta):
    direct = sp.diff(beta, theta) + p1*sp.diff(alpha, theta) + q*sp.diff(gamma, theta)
    dDV = value2_derivative(theta,u,-1,p2j)-value2_derivative(theta,v,1,p20)
    return sp.simplify(s01*sp.diff(c,theta)+sp.diff(alpha,theta)/alpha**2
                      -sj1*direct/alpha-s01*(sp.diff(delta,theta)*(Vj-V0)+delta*dDV))


def total_derivative(expr, theta):
    dp = price1_derivative(theta)
    dpj, dp0 = price2_derivative(theta,u,-1,p2j), price2_derivative(theta,v,1,p20)
    ds = sj1*s01*(sp.diff(beta,theta)+p1*sp.diff(alpha,theta)+q*sp.diff(gamma,theta)+alpha*dp)
    du = u*(1-u)*(sp.diff(beta,theta)+p2j*sp.diff(alpha,theta)-sp.diff(gamma,theta)+alpha*dpj)
    dv = v*(1-v)*(sp.diff(beta,theta)+p20*sp.diff(alpha,theta)+sp.diff(gamma,theta)+alpha*dp0)
    substitutions = {p1:dp, p2j:dpj, p20:dp0, sj1:ds, u:du, v:dv,
                     Vj:value2_derivative(theta,u,-1,p2j), V0:value2_derivative(theta,v,1,p20)}
    return sp.factor(sp.diff(expr,theta)+sum(sp.diff(expr,k)*val for k,val in substitutions.items()))

dp1_dgamma = price1_derivative(gamma)
dp2_dgamma = price2_derivative(gamma)
dVj_dgamma, dV0_dgamma = value2_derivative(gamma,u,-1,p2j), value2_derivative(gamma,v,1,p20)
ds_dgamma = sp.factor(sj1*s01*(q+alpha*dp1_dgamma))
du_dgamma, dv_dgamma = -u*(1-u)**2, v*(1-v)**2
d2p1_dgamma2 = total_derivative(dp1_dgamma,gamma)
def terminal_total_derivative(expr,theta):
    dp = price2_derivative(theta)
    ds = sj2*s02*(sp.diff(beta,theta)+p2*sp.diff(alpha,theta)+q2*sp.diff(gamma,theta)+alpha*dp)
    return sp.factor(sp.diff(expr,theta)+sp.diff(expr,p2)*dp+sp.diff(expr,sj2)*ds)


def profit_derivative(theta):
    direct = sp.diff(beta,theta)+p1*sp.diff(alpha,theta)+q*sp.diff(gamma,theta)
    partial = (direct*sj1*s01*(p1-c+delta*(Vj-V0))-sj1*sp.diff(c,theta)
               +sp.diff(delta,theta)*(sj1*Vj+s01*V0)
               +delta*(sj1*value2_derivative(theta,u,-1,p2j)+s01*value2_derivative(theta,v,1,p20)))
    return sp.simplify(partial.subs(p1,c-1/(alpha*s01)-delta*(Vj-V0)))


d2p2_dgamma2 = terminal_total_derivative(dp2_dgamma,gamma)
ds2_dgamma = sp.factor(sj2*s02*(q2+alpha*dp2_dgamma))
dp2dbeta_dgamma = terminal_total_derivative(price2_derivative(beta),gamma)
dPi_dgamma = profit_derivative(gamma)
dPi_dbeta = profit_derivative(beta)
ratio_derivative = (p2*dp1_dgamma-p1*dp2_dgamma)/p2**2
matched_j = ratio_derivative.subs({q:-1,p2:p2j,sj2:u,q2:-1})
matched_0 = ratio_derivative.subs({q:1,p2:p20,sj2:v,q2:1})
induce_harvest = ratio_derivative.subs({q:1,p2:p2j,sj2:u,q2:-1})
