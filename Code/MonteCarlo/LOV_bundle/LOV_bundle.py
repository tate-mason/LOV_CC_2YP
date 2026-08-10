"""
Load packages:
    pandas:      data manipualtion
    numpy:       numerical operations
    scipy:       scientific computation
    matplotlib:  plotting program
    seaborn:     more in depth plotting
    collections: namedtuple for easy pulling of results
    prettytable: customizable output tables for results
    statsmodels: simple regression operations
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp, expit
import statsmodels.api as sm
import sys

"""
Functions for outputting:
    Tee: Creates a text file of output alongside console output. Used for communicating output quickly.
    save_tex_table: Function for saving LaTeX tables from package output tables
"""

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.terminal = sys.__stdout__
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee('mle_results.txt')

import os
os.makedirs('../Output/Tables', exist_ok=True)
os.makedirs('../Output/Plots', exist_ok=True)
def save_tex_table(latex_str, filename):
    with open(f'../Output/Tables/{filename}.tex', 'w') as f:
        f.write(latex_str)

tol = 1e-300
buff = 1e10
seed = 219

rng = np.random.default_rng(seed)

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

"""
ccp_iv_base computes conditional choice probabilities over five products, 100 time periods, and 1000 simulations. It takes in two product characteristics, x1 and x2 as well as
individual preference parameters beta and gamma. Each period, it computes the probability of consumer i choosing product j among the 5 products and outside option. 
"""

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
        x1_chosen = np.zeros(T) # empty vector of choices per period
        x2_chosen = np.zeros(T)
        x1_bar = np.zeros(T)
        x2_bar = np.zeros(T)

        V = np.zeros((T, J+1))

        for t in range(1, T):
            x1_bar_t = np.mean(x1_chosen[:t])
            x2_bar_t = np.mean(x2_chosen[:t])
            xi = np.sqrt((x1 - x1_bar_t)**2 + (x2 - x2_bar_t)**2)
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
                x1_chosen[t] = x1_bar_t
                x2_chosen[t] = x2_bar_t
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        U_S[s] = V
        IV_S[s] = IV
        prob_S[s] = prob
        x1_bar_S[s] = x1_bar_t
        x2_bar_S[s] = x2_bar_t

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

S = 500
M = 10

rng_mle = np.random.default_rng(seed=45)
rng_data = np.random.default_rng(seed=99)
def ccp_iv_mle(S, T, T_prior, J, x1, x2, beta, gamma, epsilon):
    prob_S = np.zeros((S, T, J+1))
    chosen_S = np.zeros((S, T), dtype=int)

    for s in range(S):
        eps_ijt = epsilon[s]

        x1_bar_prior, x2_bar_prior = 0.0,0.0

        x1_chose, x2_chose = np.zeros(T), np.zeros(T)
        V = np.zeros((T, J+1))

        for t in range(1, T):
            x1_bar = np.mean(x1_chose[:t]) if t > 1 else x1_bar_prior
            x2_bar = np.mean(x2_chose[:t]) if t > 1 else x2_bar_prior
            xi = np.sqrt((x1 - x1_bar)**2 + (x2 - x2_bar)**2)

            u = beta[0]*x1 + beta[1]*x2 + gamma*np.log(1+xi**2) + eps_ijt[t,1:]
            u_out = eps_ijt[t,0]
            u_all = np.concatenate([[u_out], u])
            V[t] = u_all

            chosen_idx = np.argmax(u_all)
            chosen_S[s,t] = chosen_idx

            if chosen_idx > 0:
                x1_chose[t] = x1[chosen_idx-1]
                x2_chose[t] = x2[chosen_idx-1]
            else:
                x1_chose[t] = x1_bar
                x2_chose[t] = x2_bar
        IV = logsumexp(V,axis=1)
        prob_S[s] = np.exp(V - IV[:,None])
    return prob_S, chosen_S

epsilon_fixed = rng_mle.gumbel(0, 1, size=(M, S, T, J+1))
x1_all = rng_data.uniform(0,10,(M,J))
x2_all = rng_data.uniform(0,10,(M,J))

def neg_ll(params, chosen_all, epsilon_fixed, x1_all, x2_all, M, S, T, T_prior, J, include_lov):
    if include_lov:
        beta1, beta2, gamma = params
    else:
        beta1, beta2 = params
        gamma = 0.0
    if gamma < 0 or beta1 < 0 or beta2 < 0:
        return 1e10

    total_ll = 0.0
    for m in range(M):
        prob_S_cand = np.zeros((S, T, J+1))
        for s in range(S):
            eps_ijt = epsilon_fixed[m, s]          # fix: index correctly

            # Prior formation — deterministic init, no stochastic draws
            x1_bar_prior, x2_bar_prior = 0.0, 0.0

            x1_chose, x2_chose = np.zeros(T), np.zeros(T)
            V = np.zeros((T, J+1))

            for t in range(1, T):
                x1_bar_t = np.mean(x1_chose[:t]) if t > 0 else x1_bar_prior
                x2_bar_t = np.mean(x2_chose[:t]) if t > 0 else x2_bar_prior
                xi = np.sqrt((x1_all[m] - x1_bar_t)**2 + (x2_all[m] - x2_bar_t)**2)

                u = beta1*x1_all[m] + beta2*x2_all[m] + gamma*np.log(1 + xi**2) + eps_ijt[t, 1:]
                u_out = eps_ijt[t, 0]
                u_all = np.concatenate([[u_out], u])
                V[t] = u_all

                chosen_idx = np.argmax(u_all)
                if chosen_idx > 0:
                    x1_chose[t] = x1_all[m, chosen_idx-1]
                    x2_chose[t] = x2_all[m, chosen_idx-1]
                else:
                    x1_chose[t] = x1_bar_t
                    x2_chose[t] = x2_bar_t

            IV = logsumexp(V, axis=1)
            prob_S_cand[s] = np.exp(V - IV[:, None])

        chosen_probs_m = prob_S_cand[np.arange(S)[:, None], np.arange(T)[None, :], chosen_all[m]]
        total_ll += np.sum(np.log(chosen_probs_m + 1e-300))
    return -total_ll

def gen_choices(g, M):
    chosen_all = []
    for m in range(M):
        _, chosen_S_m = ccp_iv_mle(S,T,T_prior,J,x1_all[m],x2_all[m],beta,g,epsilon_fixed[m])
        chosen_all.append(chosen_S_m)
    return chosen_all

def est(chosen_all):
    res_no_LOV = sp.optimize.minimize(
        neg_ll,
        x0=[1.0, 1.0],
        args=(chosen_all, epsilon_fixed, x1_all, x2_all,M, S, T, T_prior, J, False),
        method='Nelder-Mead'
    )
    res_LOV = sp.optimize.minimize(
        neg_ll,
        x0=[1.0, 1.0, 5.0],
        args=(chosen_all, epsilon_fixed, x1_all, x2_all,M, S, T, T_prior, J, True),
        method='Nelder-Mead'
    )
    return res_no_LOV,res_LOV

def print_res(res_no_LOV, res_LOV, chosen_all, g):
    print(f'True: beta1={beta[0]}, beta2={beta[1]}, gamma={g}')
    print(f'Estimated (No LOV): beta1={res_no_LOV.x[0]:.4f}, beta2={res_no_LOV.x[1]:.4f}')
    print(f'Estimated (LOV):    beta1={res_LOV.x[0]:.4f}, beta2={res_LOV.x[1]:.4f}, gamma={res_LOV.x[2]:.4f}')
    print(f'Converged (No LOV): {res_no_LOV.success} | Converged (LOV): {res_LOV.success}')
    print(f'LL (No LOV): {-res_no_LOV.fun:.2f} | LL (LOV): {-res_LOV.fun:.2f}')
    print(f'Percent taking outside option: {np.mean([c == 0 for c in chosen_all]):.2f}')
    print('-'*60)

def comp_ll(S, T, T_prior, J, M, beta, gamma, epsilon_fixed, x1_all, x2_all):
    chosen_all = gen_choices(gamma, M)
    res = est(chosen_all)
    print_res(res[0],res[1], chosen_all, g)

for g in gamma:
    mle = comp_ll(S, T, T_prior, J, M, beta, g, epsilon_fixed, x1_all, x2_all)
