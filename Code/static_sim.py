import pandas as pd
import os
import numpy as np
import scipy as sp
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import namedtuple
from scipy.stats import gumbel_r

"""
This file will be used to do a static simulation of the love of variety problem:
    - Consumer Problem: U_ij = beta_i*X_j - gamma_i*(Sigma_j) + a_ij + epsilon_ij
        s.t. gamma ~ N(0, sigma_gamma); Sigma_jt = 1/T-t X_bar - X_j
    - Firm Problem: V(a_t;Sigma_jt, lambda_t(gamma_i)) = {pi + int_gamma V(a';Sigma_jt+1, lambda_t+1(gamma_i))dlambda_t(gamma_i)}
        s.t. pi = (1+a)p - c
             lambda_t(gamma) ~ N(1/Sigma_jt, sigma_lambda), a = {0,1}
Structure:
    1) solve consumer's choice problem
    2) compute individual shares of products
    3) compute firm's problem, choosing optimal a
    4) find partial equilibrium
Landscape:
    1) T = 1
    2) J = 5
    3) i = 1
    4) gamma = range(0.2(0.3)0.8), beta = range(0.2(0.3)0.8)
Functions:

Libraries Used:
    - numpy: used for numerical operation
    - scipy: used for optimization routines
    - matplotlib, seaborn: used for plotting
    - tabulate: better output text results

"""

#=============================================================#
# Setup and Parameter Definition                              #
#=============================================================#

output_dir = os.chdir('../Output/')

rng = np.random.default_rng(seed=219)

# Parameterization
S = 1000
prod_space = [1.0, 2.0, 3.0, 4.0, 5.0]
J = len(prod_space)
kappa = 2


#=============================================================#
# Consumer Problem                                            #
#=============================================================#

beta = 2.5
gamma = 2.5
X_bar = 2.5 

U_all = np.zeros((S, len(prod_space)))
shares_all = np.zeros((S, len(prod_space)))

for s in range(S):
    X_j = np.array(prod_space)
    Sigma_j = X_bar - X_j
    eps = rng.gumbel(0,1,len(prod_space))
    U_ij = beta*X_j - gamma*Sigma_j + eps
    # Compute shares
    exp_U_ij = np.exp(U_ij)
    sum_exp_U_ij = np.sum(exp_U_ij)
    shares = exp_U_ij / sum_exp_U_ij

    U_all[s] = U_ij
    shares_all[s] = shares

mean_shares = np.mean(shares_all, axis=0)

plt.figure(figsize=(8,4))
plt.bar(range(1, len(prod_space)+1), mean_shares)
plt.xlabel('Product')
plt.ylabel('Average Share')
plt.title('Average Market Shares of Products')
plt.xticks(range(1, len(prod_space)+1), [f"j={x:.1f}" for x in prod_space])
plt.tight_layout()
plt.show()

# put this right after simulate_cons_markov is defined, before any plotting

T = 100 
beta_test, gamma_test = 1, 4  # pick a regime you care about
epsilon_ijt = rng.gumbel(0, 1, size=(T, J))
x_chosen = np.zeros(T)
X_jt = np.zeros((T, J))
x_bar = np.zeros(T)
x_bar[0] = X_bar
Sigma = np.zeros((T, J))
V = np.zeros((T, J))
chosen_idx = np.zeros(T, dtype=int)

X_jt[0] = np.array(prod_space)
u0 = beta_test*X_jt[0] + epsilon_ijt[0]
V[0] = u0
chosen_idx[0] = np.argmax(u0)
x_chosen[0] = X_jt[0, chosen_idx[0]]
Sigma[0, chosen_idx[0]] = x_chosen[0]

for t in range(1, T):
    x_bar[t] = np.mean(x_chosen[:t])
    X_jt[t] = np.array(prod_space)
    a_t = kappa*(x_bar[t-1] - x_chosen[t-1])**2
    Sigma[t] = Sigma[t-1].copy()
    Sigma[t, chosen_idx[t-1]] += x_chosen[t-1]
    u = beta_test*X_jt[t] + gamma_test*Sigma[t] + a_t + epsilon_ijt[t]
    V[t] = u
    chosen_idx[t] = np.argmax(V[t])
    x_chosen[t] = X_jt[t, chosen_idx[t]]

for t in range(T):
    print(f"t={t:3d} | chosen={chosen_idx[t]} | x={x_chosen[t]:.2f} | "
          f"Sigma={Sigma[t].round(2)} | V={V[t].round(2)}")
