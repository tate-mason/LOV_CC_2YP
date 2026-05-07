import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable, TableStyle
from scipy.special import logsumexp, expit
from scipy.spatial.distance import euclidean
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
        \Xi_{ijt} = \Sum_{j} \sqrt{(X_{jt} - \bar{X}_{jt})^2}
        \varepsilon_{ijt} \sim Gumbel(0,1)

This allows for \beta to vary with the specific characteristics the product holds rather than just a beta for the overall bundle. Variety now takes the euclidean distance of each characteristic of the product's bundle from the amount the consumer has already consumed in the past, allowing for a more nuanced measure of variety that captures the idea that consumers may want to try products that have some different characteristics from their past consumption, but not necessarily products that are completely different from what they have already consumed.

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

#beta = [0.5, 2] # preference for product traits 
beta_pairs = [(2, 0)] # creates each permutation of pairings of beta
gamma = np.array([0, 6, 9, 12]) # preference for variety

product_space = np.column_stack([
    rng.uniform(0,10,size=J),
    rng.uniform(0,10,size=J)
])

Results = namedtuple('Results',[
    'inclusive_value',
    'prob_choose',
    'theta',
    'utility',
    'x1_bar_path',
    'x2_bar_path',
])

# === Consumer Logit === #

r"""
function consumer_choice_prob simulates a consumption path for a consumer i under each gamma specification and beta pairing. Computes the probability of choosing each product, returns inclusive value, utility, and the consumers theta to create the \Xi term for regression.
"""

def consumer_choice_prob(S, T, T_prior, J, product_space, M, beta_pairs, gamma):
    x_chosen = np.zeros((S,T))
    x_bar = np.zeros((S,T))
    x_jt = np.zeros((S,T,J))
    V_S = np.zeros((S,T,J))
    IV_S = np.zeros((S,T))
    prob_S = np.zeros((S,T,J))
    U_S = np.zeros((S,T,J))
    theta_S = np.zeros((S,2))
    x1_bar_S = np.zeros((S,T))
    x2_bar_S = np.zeros((S,T))

    for s in range(S):
        eps_ijt = rng.gumbel(0,1, size=(T,J))
        eps_history = rng.gumbel(0,1, size=(T_prior,J))

        x_history1 = np.zeros(T_prior)
        x_history2 = np.zeros(T_prior)
        x_bar_history1 = 0.0
        x_bar_history2 = 0.0
        history_choices1 = np.zeros(T_prior)
        history_choices2 = np.zeros(T_prior)
        Xi_history = np.zeros((T_prior,J))

        for t in range(T_prior):
            u_history = 0.0
            x_history1 = product_space[:,0]
            x_history2 = product_space[:,1]
            Xi_history[t] = np.sqrt((x_history1 - x_bar_history1)**2 + (x_history2 - x_bar_history2)**2)
            U_history = beta_pairs[0]*x_history1 + beta_pairs[1]*x_history2 + gamma*np.log(1+Xi_history[t]**2) + rng.gumbel(0,1,J)
            chosen_history = np.argmax(U_history)
            history_choices1[t] = x_history1[chosen_history]
            history_choices2[t] = x_history2[chosen_history]
            x_bar_history1 = np.mean(history_choices1[:t+1])
            x_bar_history2 = np.mean(history_choices2[:t+1])
        x1_chosen_t = np.zeros(T)
        x2_chosen_t = np.zeros(T)
        x1_jt = np.zeros((T,J))
        x2_jt = np.zeros((T,J))
        x1_bar_t = np.zeros(T)
        x2_bar_t = np.zeros(T)
        Xi_t = np.zeros((T,J))

        V = np.zeros((T,J))
        chosen1_idx = np.zeros(T, dtype=int)
        chosen2_idx = np.zeros(T, dtype=int)

        x1_jt[0] = product_space[:,0]
        x2_jt[0] = product_space[:,1]
        x1_bar_t[0] = x_bar_history1
        x2_bar_t[0] = x_bar_history2
        Xi_t[0] = np.sqrt((x1_jt[0] - x1_bar_t[0])**2 + (x2_jt[0] - x2_bar_t[0])**2)
        U0 = beta_pairs[0]*x1_jt[0] + beta_pairs[1]*x2_jt[0] + gamma*np.log(1+Xi_t[0]**2) + eps_ijt[0]
        V[0] = U0
        chosen1_idx[0] = np.argmax(V[0])
        chosen2_idx[0] = np.argmax(V[0])
        x1_chosen_t[0] = x1_jt[0, chosen1_idx[0]]
        x2_chosen_t[0] = x2_jt[0, chosen2_idx[0]]

        for t in range(1, T):
            u = np.zeros(J)
            x1_jt[t] = product_space[:,0]
            x2_jt[t] = product_space[:,1]
            x1_bar_t[t] = np.mean(x1_chosen_t[:t])
            x2_bar_t[t] = np.mean(x2_chosen_t[:t])
            Xi_t[t] = np.sqrt((x1_jt[t] - x1_bar_t[t])**2 + (x2_jt[t] - x2_bar_t[t])**2)
            U = beta_pairs[0]*x1_jt[t] + beta_pairs[1]*x2_jt[t] + gamma*np.log(1+Xi_t[t]**2) + eps_ijt[t]
            V[t] = U
            chosen1_idx[t] = np.argmax(V[t])
            chosen2_idx[t] = np.argmax(V[t])
            x1_chosen_t[t] = x1_jt[t, chosen1_idx[t]]
            x2_chosen_t[t] = x2_jt[t, chosen2_idx[t]]
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        x_chosen[s] = x1_chosen_t
        theta_S[s] = [x1_bar_t[-1], x2_bar_t[-1]]
        x1_bar_S[s] = x1_bar_t
        x2_bar_S[s] = x2_bar_t
        x_jt[s] = x1_jt
        IV_S[s] = IV
        prob_S[s] = prob
        U_S[s] = V

    return Results(
        inclusive_value = IV_S.mean(axis=0),
        prob_choose = prob_S.mean(axis=0),
        theta = theta_S.mean(axis=0),
        utility = U_S.mean(axis=0),
        x1_bar_path = x1_bar_S.mean(axis=0),
        x2_bar_path = x2_bar_S.mean(axis=0),
    )

product_table = PrettyTable()
product_table.title = "Product Characteristics"
product_table.field_names = ["Product", "Characteristic 1", "Characteristic 2", "Total Value"]
product_table.align["Product"] = "l"
product_table.align["Characteristic 1"] = "r"
product_table.align["Characteristic 2"] = "r"
product_table.align["Total Value"] = "r"

for j in range(J):
    product_table.add_row([
        f"{j+1}",
        f"{product_space[j,0]:.2f}",
        f"{product_space[j,1]:.2f}",
        f"{product_space[j,0] + product_space[j,1]:.2f}"
    ])

print(product_table)

save_tex_table(product_table.get_latex_string(), 'product_characteristics')

IV_Table = PrettyTable()
IV_Table.title = "Inclusive Value by Gamma and Beta Pair"
IV_Table.field_names = ["Gamma", "Beta Pair", "Inclusive Value"]
IV_Table.align["Gamma"] = "r"
IV_Table.align["Beta Pair"] = "c"
IV_Table.align["Inclusive Value"] = "r"


for g in gamma:
    for b in beta_pairs:
        res = consumer_choice_prob(S, T, T_prior, J, product_space, M, b, g)
        fig, ax = plt.subplots(figsize=(10,6))
        for j in range(J):
            sns.lineplot(x=np.arange(T), y=res.utility[:,j], label=f'Product {j+1}', ax=ax)
        ax.set_title(f'Utility over Time for Gamma={g} and Beta Pair={b}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Utility')
        ax.legend()
        plt.savefig(f'../Output/Plots/utility_gamma_{g}_beta2.pdf')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(x=np.arange(T), y=res.inclusive_value, label='Inclusive Value', ax=ax)
        ax.set_title(f'Inclusive Value over Time for Gamma={g} and Beta Pair={b}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Inclusive Value')
        ax.legend()
        plt.savefig(f'../Output/Plots/inclusive_value_gamma_{g}_beta_.pdf')
        plt.close()

        IV_Table.add_row([f"{g}", f"{b}", f"{res.inclusive_value[-1]:.2f}"])

print(IV_Table)
save_tex_table(IV_Table.get_latex_string(), 'inclusive_value_table')
# === Regressions === #

CCP_M = np.zeros((M, len(gamma), T, J))
theta_M = np.zeros((M, len(gamma), T, 2))
prod_spaces = np.zeros((M, J, 2))

for m in range(M):
    prod_spaces[m] = np.column_stack([
        rng.uniform(0,10,size=J),
        rng.uniform(0,10,size=J)
    ])
    for g_idx, g in enumerate(gamma):
        res = consumer_choice_prob(S, T, T_prior, J, prod_spaces[m], M, (2,0), g)
        CCP_M[m, g_idx] = res.prob_choose
        theta_M[m, g_idx,:, 0] = res.x1_bar_path
        theta_M[m, g_idx,:, 1] = res.x2_bar_path

diff_14 = np.log(CCP_M[:,0,:,1]) - np.log(CCP_M[:,0,:,3])
diff_14 = diff_14.reshape(M*T)
x1_diff = np.repeat(prod_spaces[:,1,0] - prod_spaces[:,3,0], T)
x2_diff = np.repeat(prod_spaces[:,1,0] - prod_spaces[:,3,0], T)
X = sm.add_constant(x1_diff)
print(f"γ = {gamma[0]}, β = 2")
print(sm.OLS(diff_14, X).fit().summary())


for g_idx, g in enumerate(gamma):
    diff_14 = np.log(CCP_M[:,g_idx,:,1]) - np.log(CCP_M[:,g_idx,:,3])
    diff_14 = diff_14.reshape(M*T)
    x1_diff = np.repeat(prod_spaces[:,1,0] - prod_spaces[:,3,0], T)
    x2_diff = np.repeat(prod_spaces[:,1,0] - prod_spaces[:,3,0], T)
    xi_prod2 = np.sqrt((prod_spaces[:,1,0][:,None] - theta_M[:,g_idx,:,0])**2 + 
                    (prod_spaces[:,1,1][:,None] - theta_M[:,g_idx,:,1])**2)
    xi_prod4 = np.sqrt((prod_spaces[:,3,0][:,None] - theta_M[:,g_idx,:,0])**2 + 
                    (prod_spaces[:,3,1][:,None] - theta_M[:,g_idx,:,1])**2)
    xi_diff = np.log(1 + xi_prod2**2) - np.log(1 + xi_prod4**2)
    xi_diff = xi_diff.flatten()
    print(f"Correlation x1_diff and xi_diff: {np.corrcoef(x1_diff, xi_diff)[0,1]:.4f}")
    X = sm.add_constant(np.column_stack([x1_diff, xi_diff]))
    print(f"γ = {g}, β = 2")
    print(sm.OLS(diff_14, X).fit().summary())
    save_tex_table(sm.OLS(diff_14, X).fit().summary().as_latex(), f'regression_gamma_{g}_beta_2')

# === NLLS === #

from scipy.optimize import least_squares


for g_idx, g in enumerate(gamma):
    diff_14 = np.log(CCP_M[:,g_idx,:,1]) - np.log(CCP_M[:,g_idx,:,3])
    diff_14 = diff_14.reshape(M*T)
    x1_diff = np.repeat(prod_spaces[:,1,0] - prod_spaces[:,3,0], T)
    x2_diff = np.repeat(prod_spaces[:,1,1] - prod_spaces[:,3,1], T)
    xi_prod2 = np.sqrt((prod_spaces[:,1,0][:,None] - theta_M[:,g_idx,:,0])**2 + 
                    (prod_spaces[:,1,1][:,None] - theta_M[:,g_idx,:,1])**2)
    xi_prod4 = np.sqrt((prod_spaces[:,3,0][:,None] - theta_M[:,g_idx,:,0])**2 + 
                    (prod_spaces[:,3,1][:,None] - theta_M[:,g_idx,:,1])**2)
    xi_diff = np.log(1 + xi_prod2**2) - np.log(1 + xi_prod4**2)
    xi_diff = xi_diff.flatten()
    X = np.column_stack([x1_diff, xi_diff])
    
    def residuals(params):
        beta_hat, gamma_hat = params
        U = beta_hat * prod_spaces[:, :, 0][:, None, :] + \
            gamma_hat * np.log(1 + ((prod_spaces[:, :, 0][:, None, :] - theta_M[:, g_idx, :, 0][:, :, None])**2 +
                                    (prod_spaces[:, :, 1][:, None, :] - theta_M[:, g_idx, :, 1][:, :, None])**2))
        IV = logsumexp(U, axis=2)  # shape (M, T)
        log_prob = U - IV[:, :, None]  # shape (M, T, J)
        predicted_diff = log_prob[:, :, 1] - log_prob[:, :, 3]  # products 2 and 4
        predicted_diff = predicted_diff.flatten()
        return diff_14 - predicted_diff
    result = least_squares(residuals, np.array([1.0,3.0]))
    print(f"γ={g}, β=2 → estimated: {result.x}")
    print(f"γ, β Std. Errors: {np.sqrt(np.diag(result.jac.T @ result.jac))}")
    save_tex_table(f"Estimated Parameters: {result.x}, Std. Errors: {np.sqrt(np.diag(result.jac.T @ result.jac))}", f'nlls_gamma_{g}_beta_2')
