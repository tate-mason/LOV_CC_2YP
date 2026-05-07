import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp, expit
import statsmodels.api as sm

import os
os.makedirs('../Output/Tables', exist_ok=True)
os.makedirs('../Output/Plots', exist_ok=True)
def save_tex_table(latex_str, filename):
    with open(f'../Output/Tables/{filename}.tex', 'w') as f:
        f.write(latex_str)

rng = np.random.default_rng(seed=219)

# Parameters

S = 1000 # simulations

T = 100 # time periods
T_prior = 5 # history formation
t_star = 50 # product introduction time

beta = [0.5, 2] # quality utility
gamma = np.array([0, 6, 9, 12]) # variety utility

cons_res = namedtuple('cons_res', [
    'IV_S',
    'IV_tstar',
    'prob_S',
    'U_S',
    'x_bar_S',
])
# ============================================================================================= #

J = 5 # number of products

def ccp_iv_base(S, T, T_prior, J, prod_space1, prod_space2, beta, gamma):
    x_chosen_S = np.zeros((S, T))
    x_bar_S = np.zeros((S, T))
    X_jt_S = np.zeros((S, T, J))
    V_S = np.zeros((S, T, J))
    IV_S = np.zeros((S, T))
    prob_S = np.zeros((S, T, J))
    U_S = np.zeros((S, T, J))
    
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T,J))

        prior_choices = np.zeros(T_prior)
        X_prior1 = np.array(prod_space1)
        X_prior2 = np.array(prod_space2)
        x_bar_prior = 0

        # initial state
        for t in range(T_prior):
            eps_prior = rng.gumbel(0, 1, size=J)
            U_prior = beta[0] * X_prior1 + beta[1] * X_prior2 + gamma * np.log(1 + (X_prior1 - x_bar_prior)**2) + eps_prior
            chosen_prior = np.argmax(U_prior)
            prior_choices[t] = X_prior1[chosen_prior]
            x_bar_prior = np.mean(prior_choices[:t+1])

        x_chosen = np.zeros(T) # empty vector of choices per period
        X1_jt = np.zeros((T,J))
        X2_jt = np.zeros((T,J))
        x_bar = np.zeros(T)

        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)

        X1_jt[0] = np.array(prod_space1)
        X2_jt[0] = np.array(prod_space2)
        a = np.zeros(T)
        x_bar[0] = x_bar_prior # informed by choices before model 
        a[0] = 0.0
        u0 = beta[0]*X1_jt[0] + beta[1]*X2_jt[0] + gamma*(X1_jt[0] - x_bar[0])**2 + a[0] + epsilon_ijt[0]
        V[0] = u0
        chosen_idx[0] = np.argmax(u0)
        x_chosen[0] = X1_jt[0, chosen_idx[0]]
        for t in range(1, T):
            u = np.zeros(J)
            X1_jt[t] = np.array(prod_space1)
            X2_jt[t] = np.array(prod_space2)
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X1_jt[t] - x_bar[t]
            u = beta[0]*X1_jt[t] + beta[1]*X2_jt[t] + gamma*np.log(1+Sigma**2) + epsilon_ijt[t]
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X1_jt[t, chosen_idx[t]]
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        U_S[s] = V
        IV_S[s] = IV
        prob_S[s] = prob
        IV_tstar = IV[50]
        x_bar_S[s] = x_bar

    return cons_res(
        IV_S.mean(axis=0),
        IV_tstar.mean(),
        prob_S.mean(axis=0),
        U_S.mean(axis=0),
        x_bar_S.mean(axis=0),
    )

M = 10
CCP_M = np.zeros((M, len(gamma), J))
prod_spaces1 = np.zeros((M, J))
prod_spaces2 = np.zeros((M, J))

for m in range(M):
    prod_space1 = rng.uniform(0, 10, size=J)
    prod_spaces1[m] = prod_space1[]
    prod_space2 = rng.uniform(0, 10, size=J)
    prod_spaces2[m] = prod_space2
    for g_idx, g in enumerate(gamma):
        prob_S = ccp_iv_base(S, T, T_prior, J, prod_space1, prod_space2, beta, g).prob_S
        CCP_M[m, g_idx] = prob_S.mean(axis=0)
        X_bar = ccp_iv_base(S, T, T_prior, J, prod_space1, prod_space2, beta, g).x_bar_S.mean(axis=0)

# Naive Regressions


## Difference in share b/w product 5&5, 1&1, 3&3, 1&5, 5&1
diff_55 = np.log(CCP_M[:, 2, 4]) - np.log(CCP_M[:, 0, 4])
diff_11 = np.log(CCP_M[:, 2, 0]) - np.log(CCP_M[:, 0, 0])
diff_33 = np.log(CCP_M[:, 2, 2]) - np.log(CCP_M[:, 0, 2])
diff_15 = np.log(CCP_M[:, 2, 0]) - np.log(CCP_M[:, 0, 4])
diff_51 = np.log(CCP_M[:, 2, 4]) - np.log(CCP_M[:, 0, 0])

## Regress log difference on beta*X
print(
    sm.OLS(diff_55, sm.add_constant(prod_spaces1[:, 4] + prod_spaces2[:, 4])).fit().summary(),
    sm.OLS(diff_11, sm.add_constant(prod_spaces1[:, 0] + prod_spaces2[:, 0])).fit().summary(),
    sm.OLS(diff_33, sm.add_constant(prod_spaces1[:, 2] + prod_spaces2[:, 2])).fit().summary(),
    sm.OLS(diff_15, sm.add_constant(np.column_stack([prod_spaces1[:, 0], prod_spaces2[:, 4]]))).fit().summary(),
    sm.OLS(diff_51, sm.add_constant(np.column_stack([prod_spaces1[:, 4], prod_spaces2[:, 0]]))).fit().summary()
)

# LOV Regresssions

lo_hi = np.column_stack([prod_spaces1[:, 0], prod_spaces2[:, 4]])
hi_lo = np.column_stack([prod_spaces1[:, 4], prod_spaces2[:, 0]])

Sigma_lohi = (lo_hi - X_bar)**2
X_lohi = np.column_stack([lo_hi, Sigma_lohi])

Sigma_hilo = (hi_lo - X_bar)**2
X_hilo = np.column_stack([hi_lo, Sigma_hilo])


print(
    sm.OLS(diff_15, sm.add_constant(X_lohi)).fit().summary(),
    sm.OLS(diff_51, sm.add_constant(X_hilo)).fit().summary()
)


