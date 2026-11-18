"""Entry-only switching: gamma applies outside->plain, never plain->plain.

Pure numerical kernel. This changes demand relative to the +/-gamma model.
Outside utility is zero; all price choices obey a zero lower bound; no learning.
"""
import numpy as np
from pricing_model import expit, scalar_maximum, static_price


def evaluate(gamma,h0,alpha,beta,delta,p1,p2j,p20):
    if h0 not in ['j','0'] or alpha >= 0 or not 0 <= delta <= 1:
        raise ValueError('Invalid history or parameters.')
    g,p1,pj,p0=np.broadcast_arrays(*[np.asarray(x,float) for x in [gamma,p1,p2j,p20]])
    v1=beta+alpha*p1+(g if h0=='0' else 0.)
    vj=beta+alpha*pj
    v0=beta+alpha*p0+g
    q1,qj,q0=expit(v1),expit(vj),expit(v0)
    cs1=np.logaddexp(0.,v1)/(-alpha)
    cs2=(q1*np.logaddexp(0.,vj)+(1-q1)*np.logaddexp(0.,v0))/(-alpha)
    revenue1=p1*q1
    revenue2=q1*pj*qj+(1-q1)*p0*q0
    cs=cs1+delta*cs2;revenue=revenue1+delta*revenue2
    return dict(quantity1=q1,quantity2=q1*qj+(1-q1)*q0,
        consumer_surplus1=cs1,consumer_surplus2=cs2,consumer_surplus=cs,
        revenue1=revenue1,revenue2=revenue2,revenue=revenue,total_welfare=cs+revenue)


def solve(gamma,weights,h0,alpha,beta,delta=1.):
    g,w=np.asarray(gamma,float),np.asarray(weights,float)
    if (g.ndim!=1 or g.shape!=w.shape or not len(g) or (w<=0).any()
            or not np.isfinite(g).all() or not np.isfinite(w).all()
            or not np.isclose(w.sum(),1.) or h0 not in ['j','0']
            or not np.isfinite([alpha,beta,delta]).all() or alpha>=0 or not 0<=delta<=1):
        raise ValueError('Invalid type distribution or parameters.')
    scale=(30+abs(beta)+np.max(abs(g)))/abs(alpha)
    pj=static_price(beta,alpha)
    p0,_=scalar_maximum(lambda p:float(w@(p*expit(beta+alpha*p+g))),(0.,scale))
    rj=np.full_like(g,pj*float(expit(beta+alpha*pj)))
    r0=p0*expit(beta+alpha*p0+g)
    def objective(p):
        q=expit(beta+alpha*p+(g if h0=='0' else np.zeros_like(g)))
        return float(w@(p*q+delta*(q*rj+(1-q)*r0)))
    p1,value=scalar_maximum(objective,(0.,scale))
    _,wide=scalar_maximum(objective,(0.,2*scale),321)
    np.testing.assert_allclose(value,wide,atol=1e-7)
    outcome=evaluate(g,h0,alpha,beta,delta,p1,pj,p0)
    np.testing.assert_allclose(value,w@outcome['revenue'],atol=1e-12)
    q=outcome['quantity1']
    derivative=float(w@(q+alpha*q*(1-q)*(p1+delta*(rj-r0))))
    residual=max(derivative,0.) if p1<1e-7 else abs(derivative)
    return dict(p1=p1,p2j=pj,p20=p0,objective=value,
                reported_revenue=float(w@outcome['revenue']),kkt_residual=residual)
