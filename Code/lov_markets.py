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
    'prob_S',
    'U_S',
    'x1_bar_S',
    'x2_bar_S',
    's_0_S'
])
# ============================================================================================= #

J = 5 # number of products

def ccp_iv_base(S, T, T_prior, J, x1, x2, beta, gamma):
    x_chosen_S = np.zeros((S, T))
    x1_bar_S = np.zeros((S, T))
    x2_bar_S = np.zeros((S,T))
    V_S = np.zeros((S, T, J+1))
    IV_S = np.zeros((S, T))
    prob_S = np.zeros((S, T, J+1))
    U_S = np.zeros((S, T, J+1))
    s_0_S = np.zeros((S,T))
    
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T,J+1))

        prior_choices1 = np.zeros(T_prior)
        prior_choices2 = np.zeros(T_prior)
        x1_bar_prior = 0
        x2_bar_prior = 0

        # initial state
        for t in range(T_prior):
            xi_prior = np.sqrt((x1 - x1_bar_prior)**2 + (x2 - x2_bar_prior)**2)
            U_prior = beta[0] * x1 + beta[1] * x2 + gamma * np.log(1 + xi_prior**2) + rng.gumbel(0,1, J)
            U_out_prior = rng.gumbel(0,1)
            chosen_prior = np.argmax(np.concatenate([[U_out_prior],U_prior]))
            if chosen_prior > 0:
                prior_choices1[t] = x1[chosen_prior-1]
                prior_choices2[t] = x2[chosen_prior-1]
            else:
                prior_choices1[t] = x1_bar_prior
                prior_choices2[t] = x2_bar_prior
            x1_bar_prior = np.mean(prior_choices1[:t+1])
            x2_bar_prior = np.mean(prior_choices2[:t+1])


        x1_chosen = np.zeros(T) # empty vector of choices per period
        x2_chosen = np.zeros(T)
        x1_bar = np.zeros(T)
        x2_bar = np.zeros(T)

        V = np.zeros((T, J+1))
        x1_bar[0] = x1_bar_prior
        x2_bar[0] = x2_bar_prior

        for t in range(1, T):
            if t > 0:
                x1_bar[t] = x1_bar_prior
                x2_bar[t] = x2_bar_prior
            xi = np.sqrt((x1 - x1_bar[t])**2 + (x2 - x2_bar[t])**2)
            u = beta[0]*x1 + beta[1]*x2 + gamma*np.log(1+xi**2) + epsilon_ijt[t, 1:]
            u_out = epsilon_ijt[t, 0]
            u_all = np.concatenate([[u_out], u])
            V[t] = u_all
            chosen_idx = np.argmax(u_all)
            s_0_S[s,t] = int(chosen_idx == 0)
            if chosen_idx > 0:
                x1_chosen[t] = x1[chosen_idx-1]
                x2_chosen[t] = x2[chosen_idx-1]
            else:
                x1_chosen[t] = x1_bar[t]
                x2_chosen[t] = x2_bar[t]
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        U_S[s] = V
        IV_S[s] = IV
        prob_S[s] = prob
        x1_bar_S[s] = x1_bar
        x2_bar_S[s] = x2_bar

    return cons_res(
        IV_S.mean(axis=0),
        prob_S.mean(axis=0),
        U_S.mean(axis=0),
        x1_bar_S.mean(axis=0),
        x2_bar_S.mean(axis=0),
        s_0_S.mean(axis=0),
    )

def save_tex_table(rows, headers, title, filename, caption=""):
    col_fmt = "c" * len(headers)
    header_row = " & ".join(headers) + " \\\\"
    lines = [
        "\\begin{tabular}{" + col_fmt + "}",
        "\\toprule",
        header_row,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f'../Output/Tables/{filename}.tex', 'w') as f:
        f.write("\n".join(lines))

M = 10

CCP_M   = np.zeros((M, len(gamma), T, J + 1))
theta1 = np.zeros((M, len(gamma),  T))
theta2 = np.zeros((M, len(gamma),  T))

prod_space1 = rng.uniform(0, 100, size=(M,J))
prod_space2 = rng.uniform(0, 100, size=(M,J))

for m in range(M):
    for g_idx,g in enumerate(gamma):
        res = ccp_iv_base(S, T, T_prior, J, prod_space1[m], prod_space2[m], beta, g)
        CCP_M[m, g_idx]  = res.prob_S
        theta1[m, g_idx] = res.x1_bar_S
        theta2[m, g_idx] = res.x2_bar_S

for g_idx, g in enumerate(gamma):
    dep, x1v, x2v = [], [], []
    for m in range(M):
        s0 = CCP_M[m, g_idx, :, 0]
        for j in range(1, J+1):
            sj = CCP_M[m, g_idx, :, j]
            x1_j = prod_space1[m, j-1]
            x2_j = prod_space2[m, j-1]
            dep.append(np.log(sj) - np.log(s0))
            x1v.append(np.full(T, x1_j))
            x2v.append(np.full(T, x2_j))

    dep = np.concatenate(dep)
    x1v = np.concatenate(x1v)
    x2v = np.concatenate(x2v)
    rhs = np.column_stack([x1v, x2v])

    res_no_LOV = sm.OLS(dep, rhs).fit()
    print(beta[0], beta[1], g)
    print(res_no_LOV.summary())
    save_tex_table(
    [[f"{res_no_LOV.params[0]:.4f}", f"{res_no_LOV.params[1]:.4f}", f"{res_no_LOV.bse[0]:.4f}", f"{res_no_LOV.bse[1]:.4f}", f"{res_no_LOV.rsquared:.4f}"]],
        headers=[r"$\hat{\beta_2}", r"$\hat{\beta_2}$", r"SE($\hat{\beta_1}$)", r"SE($\hat{\beta_2}$)", "$R^2$"],
        title=f"Naive OLS",
        filename=f'naive_regression_summary_gamma_{g}',
    )

for g_idx, g in enumerate(gamma):
    dep, x1v, x2v, xiv = [], [], [], []
    for m in range(M):
        s0 = CCP_M[m, g_idx, :, 0]
        for j in range(1, J+1):
            sj = CCP_M[m, g_idx, :, j]
            x1_j = prod_space1[m, j-1]
            x2_j = prod_space2[m, j-1]
            xi_j = np.sqrt((x1_j - theta1[m, g_idx])**2 + (x2_j - theta2[m, g_idx])**2)
            dep.append(np.log(sj) - np.log(s0))
            x1v.append(np.full(T, x1_j))
            x2v.append(np.full(T, x2_j))
            xiv.append(np.log(1 + xi_j**2))

    dep = np.concatenate(dep)
    x1v = np.concatenate(x1v)
    x2v = np.concatenate(x2v)
    xiv = np.concatenate(xiv)

    rhs = np.column_stack([x1v, x2v, xiv])

    res_LOV = sm.OLS(dep, rhs).fit()
    print(beta[0], beta[1], g)
    print(res_LOV.summary())
    save_tex_table(
        [[f"{res_LOV.params[0]:.4f}", f"{res_LOV.params[1]:.4f}", f"{res_LOV.params[2]:.4f}", f"{res_LOV.bse[0]:.4f}", f"{res_LOV.bse[1]:.4f}", f"{res_LOV.bse[2]:.4f}", f"{res_LOV.rsquared:.4f}"]],
        headers=[r"$\hat{\beta_1}$", r"$\hat{\beta_2}$", r"$\hat{\gamma}$", r"SE($\hat{\beta_1}$)", r"SE($\hat{\beta_2}$)", r"SE($\hat{\gamma}$)", "$R^2$"],
        title=f"OLS with LOV",
        filename=f'lov_regression_summary_gamma_{g}',
    )
