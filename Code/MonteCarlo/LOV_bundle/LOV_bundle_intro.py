import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp
import statsmodels.api as sm
from itertools import product as iproduct
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

r"""
Case of heterogenous preferences over the bundle of characteristics which comprise a product. Script simulates product introduction in this case. New product introduced at t=50
"""

rng = np.random.default_rng(219)

# === Parameters === #

M = 10 # markets
J = 5 # initial product menu
T = 100 # time horizon
T_prior = 10 # history
t_star = 50 # period of introduction
S = 1000 # simulations

beta = [0.5,2]
gamma = np.array([0, 6, 9, 12])

res = namedtuple('res', [
    'V_s',
    'U_s',
    'IV_s',
    'IV_50',
    's0_s',
    's0_50',
    'prob_s',
    'x1_bar_s',
    'x2_bar_s',
])

# === Choice Problem === #

def ccp_iv_base(S, T, T_prior, t_star, J, x1, x2, beta, gamma):
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
                x1_bar[t] = np.mean(x1_chosen[:t])
                x2_bar[t] = np.mean(x2_chosen[:t])
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

    return res(
        V_s = U_S.mean(axis=0),
        U_s = U_S.mean(axis=0),
        IV_s = IV_S.mean(axis=0),
        IV_50 = IV_S[:,t_star].mean(),
        s0_s = s_0_S.mean(axis=0),
        s0_50 = s_0_S[:,t_star].mean(),
        prob_s = prob_S.mean(axis=0),
        x1_bar_s = x1_bar_S.mean(axis=0),
        x2_bar_s = x2_bar_S.mean(axis=0),
    )
def CCP_intro_bundle(T, T_prior, t_star, S, J, gamma, beta, x1, x2, x1_new, x2_new):
    # intitialize results vectors
    x_chosen_S = np.zeros((S, T)) # track chosen products
    x1_bar_S = np.zeros((S, T)) # mean first char
    x2_bar_S = np.zeros((S,T)) # mean second char
    V_S = np.zeros((S, T, J+2)) # value
    IV_S = np.zeros((S, T)) # inclusive value
    prob_S = np.zeros((S, T, J+2)) # choice probability
    U_S = np.zeros((S, T, J+2)) # utility
    s_0_S = np.zeros((S,T)) # outside share

    X1_base = np.array(x1)
    X2_base = np.array(x2)

    X1_full = np.append(x1, x1_new)
    X2_full = np.append(x2, x2_new)

    for s in range(S):
        eps = rng.gumbel(0, 1, size=(T,J+2))
        prior_choice1 = np.zeros(T_prior)
        prior_choice2 = np.zeros(T_prior)
        X1_prior = np.array(x1)
        X2_prior = np.array(x2)
        x1_bar_prior = 0
        x2_bar_prior = 0

        for t in range(T_prior):
            eps_prior = rng.gumbel(0, 1, size=J)
            xi_prior = np.sqrt((X1_prior - x1_bar_prior)**2 + (X2_prior - x2_bar_prior)**2)
            U_prior = beta[0] * X1_prior + beta[1] * X2_prior + gamma * np.log(1 + xi_prior**2) + eps_prior
            U_out_prior = rng.gumbel(0,1)
            U_prior = np.concatenate([[U_out_prior],U_prior])
            chosen_prior = np.argmax(U_prior)
            if chosen_prior > 0:
                prior_choice1[t] = X1_prior[chosen_prior-1]
                prior_choice2[t] = X2_prior[chosen_prior-1]
            else:
                prior_choice1[t] = x1_bar_prior
                prior_choice2[t] = x2_bar_prior
            x1_bar_prior = np.mean(prior_choice1[:t+1])
            x2_bar_prior = np.mean(prior_choice2[:t+1])

        x1_chosen = np.zeros(T)
        x2_chosen = np.zeros(T)
        x1_bar = np.zeros(T)
        x2_bar = np.zeros(T)

        V = np.zeros((T, J+2))

        for t in range(T):
            if t == 0:
                x1_bar[t] = x1_bar_prior
                x2_bar[t] = x2_bar_prior
            elif t < t_star:
                x1_bar[t] = np.mean(x1_chosen[:t])
                x2_bar[t] = np.mean(x2_chosen[:t])
            else:
                x1_bar[t] = np.mean(x1_chosen[:t])
                x2_bar[t] = np.mean(x2_chosen[:t])

            # compute u_all for every t
            if t < t_star:
                X1_t, X2_t = X1_base, X2_base
                xi = np.sqrt((X1_t - x1_bar[t])**2 + (X2_t - x2_bar[t])**2)
                u = beta[0]*X1_t + beta[1]*X2_t + gamma*np.log(1+xi**2) + eps[t, 1:J+1]
                u_out = eps[t, 0]
                u_all = np.concatenate([[u_out], u, [-np.inf]])
            else:
                X1_t, X2_t = X1_full, X2_full
                xi = np.sqrt((X1_t - x1_bar[t])**2 + (X2_t - x2_bar[t])**2)
                u = beta[0]*X1_t + beta[1]*X2_t + gamma*np.log(1+xi**2) + eps[t, 1:]
                u_out = eps[t, 0]
                u_all = np.concatenate([[u_out], u])

            V[t] = u_all
            chosen_idx = np.argmax(u_all)

            if chosen_idx > 0:
                if t < t_star:
                    x1_chosen[t] = X1_base[chosen_idx-1]
                    x2_chosen[t] = X2_base[chosen_idx-1]
                else:
                    x1_chosen[t] = X1_full[chosen_idx-1]
                    x2_chosen[t] = X2_full[chosen_idx-1]
            else:
                x1_chosen[t] = x1_bar[t]
                x2_chosen[t] = x2_bar[t]
            s_0_S[s,t] = int(chosen_idx == 0)
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        IV_S[s] = IV
        prob_S[s] = prob
        U_S[s] = V
        x1_bar_S[s] = x1_bar
        x2_bar_S[s] = x2_bar
    IV_tstar = IV_S[:,t_star].mean()

    return res(
        V_s = U_S.mean(axis=0),
        U_s = U_S.mean(axis=0),
        IV_s = IV_S.mean(axis=0),
        IV_50 = IV_tstar,
        s0_s = s_0_S.mean(axis=0),
        s0_50 = s_0_S[:,t_star].mean(),
        prob_s = prob_S.mean(axis=0),
        x1_bar_s = x1_bar_S.mean(axis=0),
        x2_bar_s = x2_bar_S.mean(axis=0),
    )


x1 = rng.uniform(0, 10, J)
x2 = rng.uniform(0, 10, J)

x1_new = rng.uniform(0, 10)
x2_new = rng.uniform(0, 10)

for g_idx, g in enumerate(gamma):
    res_base = ccp_iv_base(S, T, T_prior, t_star, J, x1, x2, beta, g)
    res_intro = CCP_intro_bundle(T, T_prior, t_star, S, J, g, beta, x1, x2, x1_new, x2_new)

    tab = PrettyTable()
    tab.title = f"LOV with Product Introduction (γ={g})"
    tab.field_names = ['Product', 'CCP (No Intro)', 'CCP (Intro)']

    # Base case: J+1 columns (outside + J products)
    # Intro case: J+2 columns (outside + J products + new product)
    for j in range(J + 2):
        if j == 0:
            label = "Outside Good"
        elif j <= J:
            label = f"Product {j}"
        else:
            label = f"Product {j} (New)"

        ccp_base = round(res_base.prob_s[:, j].mean(), 4) if j < J + 1 else "—"
        ccp_intro = round(res_intro.prob_s[:, j].mean(), 4)

        tab.add_row([label, ccp_base, ccp_intro])

    # Inclusive value comparison
    iv_base_mean  = res_base.IV_s.mean()
    iv_intro_mean = res_intro.IV_s.mean()
    iv_pct_change = round((iv_intro_mean - iv_base_mean) / abs(iv_base_mean) * 100, 4)

    tab.add_row(["Mean IV (No Intro)",  round(iv_base_mean,  4), ""])
    tab.add_row(["Mean IV (Intro)",     round(iv_intro_mean, 4), ""])
    tab.add_row(["% Δ IV",              "",  f"{iv_pct_change}%"])
    tab.add_row([f"IV at t={t_star}",   "", round(res_intro.IV_50, 4)])

    # Outside share comparison
    s0_base  = round(res_base.s0_s.mean(),  4)
    s0_intro = round(res_intro.s0_s.mean(), 4)
    tab.add_row(["Mean Outside Share", s0_base, s0_intro])

    print(tab)


print("Product Values: ")
for j in range(J):
    print(f"Product {j+1}: x1={x1[j]:.2f}, x2={x2[j]:.2f}")
    print(f"New Product: x1={x1_new:.2f}, x2={x2_new:.2f}")
