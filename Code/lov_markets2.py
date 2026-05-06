import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp, expit
import statsmodels.api as sm
from itertools import product as iproduct

import os
os.makedirs('../Output/Tables', exist_ok=True)
os.makedirs('../Output/Plots', exist_ok=True)
def save_tex_table(latex_str, filename):
    with open(f'../Output/Tables/{filename}.tex', 'w') as f:
        f.write(latex_str)

r"""
Setting out to see how to incorporate preference heterogeneity into model via augmenting the U_{ijt} term it is now defined as follows:

    U_ijt = \Sum_{j\in J^c} \beta_{ji}X_{jt} + \gamma_i\log(1 + \Xi_{ijt}^2) + \varepsilon_{ijt}
    s.t.
        \Xi_{ijt} = \Sum_{j\in J^c}\omega_j(X_{jt} - \theta_{it})

This allows for \beta to vary with the specific characteristics the product holds rather than just a beta for the overall bundle. Augmented variety to take this into account as well, weighting the different characteristics difference from their history. Would really like to discuss this, but seemed best way to operationalize.

Results reported:
    \begin{itemize}
        \item Utility graphs
        \item Regression results on log difference in shares
        \item Inclusive value tables
    \end{itemize}

"""

r"""
Structure of file:
    1. Parameterization:
        - set time, simulations, markets, products, etc.
    2. Logits:
        - build logit with new utility function, keep simulated history, save results in namedtuple for easier calling
    3. Graphics:
        - graph utility over time and across markets for consumer i
    4. Table:
        - inclusive values under different gamma values and permutations of two set beta's (high, low)
    5: Regressions:
        - simple OLS of log(s_j) - log(s_k) = \Sum \beta X_jt + \varepsilon_{ijt}
        - simple OLS of log(s_j) - log(s_k) = \Sum \beta X_jt + \gamma_i\log(1 + \Xi^2) + \varepsilon_{ijt}

Packages used:
    - pandas: data manipulation
    - numpy: numerical operations
    - scipy: statistical functions
    - matplotlib: plotting
    - seaborn: enhanced plotting
    - collections.namedtuple: structured data storage
    - prettytable: formatted tables
    - scipy.special: logsumexp for numerical stability, expit for logistic function
    - statsmodels: regression analysis

Functions:
"""

# === Parameterization === #

r"""
Parameters:
    - Markets: 10 markets s.t. M = \{m_1, m_2, ..., m_M\}
    - Product Space: 5 products each with differing characteristics drawn from a uniform distribution 
    - Time: 100 periods
    - Simulations: 1000 loops
    - Consumers: 1 consumer
    - \beta: \beta = [2, 0.5]
    - \gamma: \gamma = [6, 9, 12]
    - Results: data storage
"""

M = 10 # markets
J = 5 # products
T = 100 # time
T_prior = 10 # history time
S = 1000 # simulations

rng = np.random.default_rng(seed=219) # seed for reproducibility

beta = [0.5, 2] # preference for product traits 
beta_pairs = list(iproduct(beta, beta)) # creates each permutation of pairings of beta
gamma = np.array([0, 6, 9, 12]) # preference for variety

Results = namedtuple('Results',[
    'inclusive_value',
    'prob_choose',
    'theta_it',
    'utility',
])

# === Consumer Logit === #

r"""
function consumer_choice_prob simulates a consumption path for a consumer i under each gamma specification and beta pairing. Computes the probability of choosing each product, returns inclusive value, utility, and the consumers theta to create the \Xi term for regression.
"""

def consumer_choice_prob(S, T, T_prior, J, product_space, M, beta_pairs, gamma):
    x_chosen = np.zeros((S,T))
    x_bar = np.zeros((S,T))
    x_jt = np.zeros((S,T,J))
    V = np.zeros((S,T,J))
    IV = np.zeros((S,T))
    prob = np.zeros((S,T,J))
    U = np.zeros((S,T,J))

    for s in range(S):
        eps_ijt = rng.gumbel(0,1, size=(T,J))
        eps_history = rng.gumbel(0,1, size=(T_prior,J))

        history_choices = np.zeros((T_prior, J))
        x_bar_history = 0.0

        for t in range(T_prior):
            u_history = 0.0
            x_history1[t] = np.array(product_space[:,1])
            x_history2[t] = np.array(product_space[:,2])
            Xi_history = (0.5*x_history1[t] + 0.5*x_history2[t]) - x_bar_history[t]
            U_history = beta[0]*x_history1[t] + beta[1]*x_history2[t] + gamma*np.log(1+Xi_history**2) + eps_history[t]
            chosen_history = np.argmax(U_history)
            history_chioces[t] = x_history1[chosen_history] + x_history2[chosen_history]
            x_bar_history = np.mean(history_choices[:t+1])











product_space = np.column_stack([
    rng.uniform(0,10,size=J),
    rng.uniform(0,10,size=J)
])

print(product_space.shape)

x_history1 = np.zeros((T_prior, J))
x_history2 = np.zeros((T_prior,J))
x_bar_history = 0.0
history_choices = np.zeros((T_prior, J))

for t in range(T_prior):
    u_history = 0.0
    x_history1 = product_space[:,0]
    x_history2 = product_space[:,1]
    bundle = 0.5*(x_history1 + x_history2)
    Xi_history = bundle - x_bar_history
    U_history = beta[0]*x_history1 + beta[1]*x_history2 + gamma[1]*np.log(1+Xi_history**2) + rng.gumbel(0,1,J)
    chosen_history = np.argmax(U_history)
    history_choices[t] = bundle[chosen_history]
    x_bar_history = np.mean(history_choices[:t+1])

print(x_bar_history)
