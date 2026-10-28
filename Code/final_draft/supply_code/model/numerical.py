"""Solve the triangular logit FOCs with monotone, bracketed bisection."""
import math


def bisect(f):
    lo, hi = -100.0, 100.0
    if not f(lo) < 0 < f(hi):
        raise ValueError('Root outside numerical bracket')
    for _ in range(100):
        mid = (lo+hi)/2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def solve(alpha=-2., beta=-1., gamma=0., delta=.7, c=1., z1=0, z2=0):
    if not (alpha < 0 and 0 < delta < 1 and c >= 0):
        raise ValueError('Invalid parameters')
    def odds(p,z):
        return math.exp(beta+gamma*z+alpha*p)
    p2 = bisect(lambda p: p-c+(1+odds(p,z2))/alpha)
    p1 = bisect(lambda p: p-c+(1+odds(p,z1))/alpha-delta*odds(p2,z2)/alpha)
    s1 = odds(p1,z1)/(1+odds(p1,z1))
    s2 = odds(p2,z2)/(1+odds(p2,z2))
    if not (p1 > 0 and p2 > 0 and 0 < s1 < 1 and 0 < s2 < 1):
        raise ValueError('Solution outside positive-price, interior-share region')
    profit = s1*(p1-c+delta*s2*(p2-c))
    return p1,p2,s1,s2,profit
