"""Numerical kernel checks only. No empirical I/O or household record fixtures."""
from pathlib import Path
import importlib.util
import sys
import math
import numpy as np
import polars as pl
from scipy.optimize import approx_fprime

base=Path(__file__).resolve().parent/'Code/final_simple'
sys.path.insert(0,str(base))
sys.path.insert(0,str(base/'demand'))

def module(name, path):
    specification=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result

from flavor_specs import categories, classify_descriptor, classification_expression
cases=[('plain','plain'),('strawberry','berry'),('peach','other_fruits'),
       ('vanilla','desserts'),('chocolate','desserts'),('strawberry cheesecake','desserts'),
       ('banana cream pie','desserts'),('mango pineapple','other_fruits'),
       ('cherry','other_fruits'),('black cherry','other_fruits'),('blackberries','berry'),
       ('coffee','other'),('lemon tart','desserts'),('mixed berry','berry')]
for text,expected in cases:
    assert classify_descriptor(text,'five')==expected,(text,expected)
    observed=pl.select(classification_expression(pl.lit(text),'five')).item()
    assert observed==expected,(text,observed)
assert classify_descriptor('vanilla','three')=='other'
assert classify_descriptor('strawberry cheesecake','three')=='berry'
assert classify_descriptor('peach','three')=='other'

est=module('expanded_estimation',base/'demand/estimation.py')
price=module('expanded_split_pricing',base/'demand/split_pricing.py')
# Aggregate multinomial counts for two abstract choice sets. No purchase histories
# or household IDs are constructed; these matrices exercise the likelihood kernel.
def mathematical_data(size):
    counts=np.array([[2.,1.,1.,3.],[1.,2.,0.,4.]])
    prices=np.array([[1.2,1.5,1.1],[1.3,1.4,1.2]])
    stock=np.array([[.4,1.1,.5],[1.2,.3,.5]])
    novelty=np.array([[.8,.3,.9],[.4,.9,.7]])
    available=np.ones((2,3),dtype=bool)
    if size==5:
        counts=np.column_stack([counts[:,:3],np.zeros((2,2)),counts[:,-1]])
        prices=np.column_stack([prices,np.zeros((2,2))])
        stock=np.column_stack([stock,np.zeros((2,2))])
        novelty=np.column_stack([novelty,np.zeros((2,2))])
        available=np.column_stack([available,np.zeros((2,2),dtype=bool)])
    return dict(counts=counts,prices=prices,stock=stock,novelty=novelty,
                xi=.3*novelty-.7*stock,available=available,cf=np.zeros_like(prices),
                control=False,hh=np.array([0,0]),n_households=1,outside=True)

results=[]
for spec in ['three','five']:
    flavors=categories(spec)
    est.FLAVORS[:]=flavors
    price.FLAVORS[:]=flavors
    data=mathematical_data(len(flavors))
    names=est.parameter_names('heterogeneous_history',True,False)
    parameter={'inside_constant':1.,'beta_berry':-.4,'beta_plain':-.2,
               'mu_log_minus_alpha':-.1,'sd_log_minus_alpha':.25,
               'mean_novelty':-.6,'sd_novelty':.3,'mean_own_stock':.05,'sd_own_stock':.1}
    theta=np.array([parameter.get(n,0.) for n in names])
    draws=est.make_draws(8,306,'heterogeneous_history')
    value,gradient=est.demand_objective_gradient(theta,data,draws,'heterogeneous_history')
    numerical=approx_fprime(theta,lambda t:est.demand_objective_gradient(t,data,draws,'heterogeneous_history')[0],1e-6)
    np.testing.assert_allclose(gradient,numerical,atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(value,-est.household_loglike(theta,data,draws,'heterogeneous_history').mean())
    results.append((value,{name:gradient[k] for k,name in enumerate(names)}))
    # Check basket coverage, uniqueness, and multinomial probability normalization.
    for capacity in [1,3,7]:
        baskets=price.compositions(capacity)
        assert len(baskets)==math.comb(capacity+len(flavors),len(flavors))
        assert len(np.unique(baskets,axis=0))==len(baskets)
        assert np.all(baskets>=0) and np.all(baskets.sum(axis=1)==capacity)
        probabilities=np.arange(1,len(flavors)+2,dtype=float)
        probabilities/=probabilities.sum()
        logcoef=math.lgamma(capacity+1)-np.vectorize(math.lgamma)(baskets+1).sum(axis=1)
        mass=np.exp(logcoef+(baskets*np.log(probabilities)).sum(axis=1)).sum()
        np.testing.assert_allclose(mass,1.,atol=1e-12)
# Activate the additional alternatives, rather than testing masked columns only.
active=mathematical_data(5)
active['available'][:]=True
active['prices'][:,3:]=[[1.7,1.8],[1.6,1.9]]
active['counts'][:,3:5]=[[1.,2.],[2.,1.]]
active['stock'][:,3:]=[[.2,.3],[.8,.4]]
active['novelty'][:,3:]=[[.9,.8],[.7,.6]]
names=est.parameter_names('heterogeneous_history',True,False)
theta=np.array([parameter.get(n,-.15 if n.startswith('beta_') else 0.) for n in names])
value,gradient=est.demand_objective_gradient(theta,active,draws,'heterogeneous_history')
numerical=approx_fprime(theta,lambda t:est.demand_objective_gradient(t,active,draws,'heterogeneous_history')[0],1e-6)
np.testing.assert_allclose(gradient,numerical,atol=3e-6,rtol=3e-5)
assert abs(gradient[names.index('beta_other_fruits')])>1e-5
assert abs(gradient[names.index('beta_desserts')])>1e-5
np.testing.assert_allclose(results[0][0],results[1][0],atol=1e-12)
for name,value in results[0][1].items():
    np.testing.assert_allclose(value,results[1][1][name],atol=1e-12)

cal=module('expanded_gamma_calibration',base/'demand/gamma_calibration.py')
# Aggregate theoretical price moments identify a known interior calibration.
import pandas as pd
rows=[]
gamma=np.array([-1.,0.,1.])
for group,g in zip(cal.GROUPS,gamma):
    for h in ['j','0']:
        p=cal.known_prices(g,h,-1.2,2.,.92)
        rows.append(dict(group=group,h0=h,households=100,observed_p1=p['expected_p1'],observed_p2=p['expected_p2']))
a,b,_,rms=cal.calibrate(pd.DataFrame(rows),gamma,.92)
np.testing.assert_allclose([a,b],[-1.2,2.],atol=1e-6)
assert rms<1e-8
# Identical type support cannot separately identify scale/intercept from equal
# first/terminal mean-price targets; retain the rejection rather than invent a fit.
flat=pd.DataFrame([dict(group=g,h0=h,households=100,observed_p1=2.,observed_p2=2.)
                   for g in cal.GROUPS for h in ['j','0']])
try:
    cal.calibrate(flat,np.zeros(3),.92)
except cal.CalibrationError as error:
    assert error.diagnostics['status']=='failed'
    assert len(error.diagnostics['attempts'])==3
else:
    raise AssertionError('Unidentified calibration was accepted')
print('PASS: descriptor precedence; Polars/Python agreement; three/five gradients; legacy likelihood equivalence; six basket-mass checks; interior calibration recovery and unidentified-calibration rejection.')
