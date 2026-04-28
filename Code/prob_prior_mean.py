import pandas as pd
import os
import numpy as np
import scipy as sp
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import namedtuple
from prettytable import PrettyTable

"""
This file does initial simulation with a mean informed by t-5 -> t-1 therefore the consumer enters with some prior mean
"""

rng = np.random.default_rng(219)

# Parameters
T = 100 # time periods
T_prior = 5 # prior time

# preferences settings
regimes = [
    (2, 3.00, "β = 2.0, γ = 3.0", "-"),
    (2, 5.00, "β = 2.0, γ = 5.0", "--"),
    (2, 7.00, "β = 2.0, γ = 7.0", ":"),
]

prod_space = np.linspace(1, 5, 5)
J = len(prod_space) # set of products
kappa = 3

sigma_gamma = [0, 0.5, 1.0] # variance of gamma across consumers
sigma_x = 0.5 # variance of product characteristics across consumers
X_bar = np.mean(prod_space)

S = 500 # number of simualtions

ConsResult = namedtuple('ConsResult', ['x_chosen_S', 'x_bar_S', 'X_jt_S', 'V_S', 'prob_S', 'U_S', 'll_S'])

def simulate_cons(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S=1000):
    x_chosen_S = np.zeros((S, T))
    x_bar_S = np.zeros((S, T))
    X_jt_S = np.zeros((S, T, J))
    V_S = np.zeros((S, T, J))
    ll_S = np.zeros(S)
    prob_S = np.zeros((S, T, J))
    U_S = np.zeros((S,T))
    
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T,J))

        prior_choices = np.zeros(T_prior)
        X_prior = np.array(prod_space)
        x_bar_prior = 0

        # initial state
        for t in range(T_prior):
            eps_prior = rng.gumbel(0, 1, size=J)
            U_prior = beta * X_prior + gamma * np.log(1 + (X_prior - x_bar_prior)**2) + eps_prior
            chosen_prior = np.argmax(U_prior)
            prior_choices[t] = X_prior[chosen_prior]
            x_bar_prior = np.mean(prior_choices[:t+1])

        x_chosen = np.zeros(T) # empty vector of choices per period
        X_jt = np.zeros((T,J))
        x_bar = np.zeros(T)

        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)

        X_jt[0] = np.array(prod_space)
        a = np.zeros(T)
        x_bar[0] = x_bar_prior # informed by choices before model 
        a[0] = 0.0
        u0 = beta*X_jt[0] + gamma*(X_jt[0] - x_bar[0])**2 + a[0] + epsilon_ijt[0]
        V[0] = u0
        chosen_idx[0] = np.argmax(u0)
        x_chosen[0] = X_jt[0, chosen_idx[0]]

        for t in range(1, T):
            u = np.zeros(J)
            if t == 1:
                V[t-1] = u0
                chosen_idx[t-1] = np.argmax(u0)
                x_chosen[t-1] = X_jt[t-1, chosen_idx[t-1]]
            X_jt[t] = np.array(prod_space)
            a[t] = 0
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta*X_jt[t] + gamma*np.log(1 + Sigma**2) + kappa*a[t] + epsilon_ijt[t] # love variety
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx[t]] 

        log_denom = logsumexp(V, axis=1) # shape (T,)
        prob = np.exp(V - log_denom[:, None])
        ll = -np.sum(V[np.arange(T), chosen_idx] - log_denom)
        
        prob_S[s] = prob
        x_chosen_S[s] = x_chosen
        x_bar_S[s] = x_bar
        X_jt_S[s] = X_jt
        V_S[s] = V
        U_S[s] = V[np.arange(T), chosen_idx]
        ll_S[s] = ll
    return ConsResult(
            x_chosen_S.mean(axis=0),  # shape (T,)
            x_bar_S.mean(axis=0),      # shape (T,)
            X_jt_S.mean(axis=0),       # shape (T, J)
            V_S.mean(axis=0),          # shape (T, J)
            prob_S.mean(axis=0),             # shape (T,J)
            U_S.mean(axis=0),              # shape (T,)
            ll_S.mean())               # scalar

colors = plt.cm.tab10(np.linspace(0,1,len(regimes)))

# Consumer Choice by Regime


for j in range(J):
    fig, ax = plt.subplots(figsize=(8, 4))
    for (beta, gamma, label, ls), color in zip(regimes, colors):
        x_chosen, x_bar, X_jt, V, prob, U, ll = simulate_cons(
            beta, gamma, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng, S
        )
        prob_smooth = np.convolve(prob[:, j], np.ones(5)/5, mode='same')
        ax.plot(np.arange(T), prob_smooth, label=label, color=color, linewidth=1.5, linestyle=ls)
    ax.set_title(f"Product {j+1} (X={prod_space[j]:.1f})", fontsize=9)
    ax.set_xlabel("Period")
    ax.set_xlim(0, 100)
    ax.set_ylabel("Choice Probability")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f'../Output/prior_mean_prob_product_{j+1}.pdf', bbox_inches='tight', format='pdf')
    plt.close()

# pick one regime to diagnose
beta_diag, gamma_diag = 1.30, .7

# run with S=10 to get individual paths
def simulate_cons_raw(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S=10):
    prob_S = np.zeros((S, T, J))
    
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T, J))

        prior_choices = np.zeros(T_prior)
        X_prior = np.array(prod_space)
        x_bar_prior = 0

        for t in range(T_prior):
            eps_prior = rng.gumbel(0, 1, size=J)
            U_prior = beta * X_prior + gamma * np.log(1 + (X_prior - x_bar_prior)**2) + eps_prior
            chosen_prior = np.argmax(U_prior)
            prior_choices[t] = X_prior[chosen_prior]
            x_bar_prior = np.mean(prior_choices[:t+1])

        x_chosen = np.zeros(T)
        X_jt = np.zeros((T, J))
        x_bar = np.zeros(T)
        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)

        X_jt[0] = np.array(prod_space)
        x_bar[0] = x_bar_prior
        u0 = beta*X_jt[0] + gamma*np.log(1 + (X_jt[0] - x_bar[0])**2) + epsilon_ijt[0]
        V[0] = u0
        chosen_idx[0] = np.argmax(u0)
        x_chosen[0] = X_jt[0, chosen_idx[0]]

        for t in range(1, T):
            X_jt[t] = np.array(prod_space)
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta*X_jt[t] + gamma*np.log(1 + Sigma**2) + epsilon_ijt[t]
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx[t]]

        log_denom = logsumexp(V, axis=1)
        prob = np.exp(V - log_denom[:, None])
        prob_S[s] = prob

    return prob_S  # shape (S, T, J) — raw, no averaging

prob_raw = simulate_cons_raw(beta_diag, gamma_diag, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng, S=10)

colors_diag = plt.cm.tab10(np.linspace(0, 1, 10))

for j in range(J):
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in range(10):
        ax.plot(np.arange(T), prob_raw[s, :, j],
                color=colors_diag[s], linewidth=1.0, alpha=0.7, label=f"s={s}")
    ax.set_title(f"Product {j+1} (X={prod_space[j]:.1f}) — β={beta_diag}, γ={gamma_diag}", fontsize=9)
    ax.set_xlabel("Period")
    ax.set_xlim(0, 100)
    ax.set_ylabel("Choice Probability")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.close()

def simulate_cons_naive(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S=1000):
    x_chosen_S = np.zeros((S, T))
    x_bar_S = np.zeros((S, T))
    X_jt_S = np.zeros((S, T, J))
    V_S = np.zeros((S, T, J))
    ll_S = np.zeros(S)
    prob_S = np.zeros((S, T, J))
    U_S = np.zeros((S,T))
    
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T,J))

        x_chosen = np.zeros(T) # empty vector of choices per period
        X_jt = np.zeros((T,J))
        x_bar = np.zeros(T)

        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)

        X_jt[0] = np.array(prod_space)
        a = np.zeros(T)
        x_bar[0] = X_bar # uninformed prior mean
        a[0] = 0
        u0 = beta*X_jt[0] + gamma*0 + epsilon_ijt[0]
        V[0] = u0
        chosen_idx[0] = np.argmax(u0)
        x_chosen[0] = X_jt[0, chosen_idx[0]]

        for t in range(1, T):
            u = np.zeros(J)
            X_jt[t] = np.array(prod_space)
            a[t] = 0
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta*X_jt[t] + epsilon_ijt[t] # love variety
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx[t]] 

        log_denom = logsumexp(V, axis=1) # shape (T,)
        prob = np.exp(V - log_denom[:, None])
        ll = -np.sum(V[np.arange(T), chosen_idx] - log_denom)
        prob_S[s] = prob
        x_chosen_S[s] = x_chosen
        x_bar_S[s] = x_bar
        X_jt_S[s] = X_jt
        V_S[s] = V
        U_S[s] = V[np.arange(T), chosen_idx]
        ll_S[s] = ll
    return ConsResult(
            x_chosen_S.mean(axis=0),  # shape (T,)
            x_bar_S.mean(axis=0),      # shape (T,)
            X_jt_S.mean(axis=0),       # shape (T, J)
            V_S.mean(axis=0),          # shape (T, J)
            prob_S.mean(axis=1),             # shape (T,J)
            U_S.mean(axis=0),              # shape (T,)
            ll_S.mean())               # scalar

naive_regimes = [(2, "β = 2.0", "-")]

prob_tab = PrettyTable()
prob_tab.field_names = ["Product", "Regime", "Mean Prob (t=50)"]

for (beta, label, ls) in naive_regimes:
    result = simulate_cons_naive(beta, gamma, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng, S)
    for j in range(J):
        prob_at_50 = result.prob_S[49, j]
        prob_tab.add_row([f"Product {j+1}", label, f"{prob_at_50:.4f}"])

print(prob_tab)
#def simulate_cons_markov(beta, gamma, kappa, sigma_gamma, T, T_prior, J, sigma_x, rng, S):
#    x_chosen_S = np.zeros((S, T))
#    x_bar_S = np.zeros((S, T))
#    X_jt_S = np.zeros((S, T, J))
#    V_S = np.zeros((S, T, J))
#    ll_S = np.zeros(S)
#    prob_S = np.zeros((S, T, J))
#    U_S = np.zeros((S,T))
#    
#    for s in range(S):
#        epsilon_ijt = rng.gumbel(0, 1, size=(T,J))
#
#        prior_choices = np.zeros(T_prior)
#        X_prior = np.array(prod_space)
#        x_bar_prior = np.mean(prod_space)
#
#        for t in range(T_prior):
#            var = gamma*(X_prior - x_bar_prior)**2
#            eps_prior = rng.gumbel(0, 1, size=J)
#            U_prior = beta * X_prior - var + eps_prior
#            chosen_prior = np.argmax(U_prior)
#            prior_choices[t] = X_prior[chosen_prior]
#            x_bar_prior = np.mean(prior_choices[:t+1])
#
#        x_chosen = np.zeros(T) # empty vector of choices per period
#        X_jt = np.zeros((T,J))
#        x_bar = np.zeros(T)
#
#        V = np.zeros((T, J))
#        chosen_idx = np.zeros(T, dtype=int)
#
#        X_jt[0] = np.array(prod_space)
#        a = np.zeros(T)
#        x_bar[0] = 2.5 # informed by choices before model 
#        a[0] = 0
#        u0 = beta*X_jt[0] - gamma*(X_jt[0] - x_bar[0])**2 + a[0] + epsilon_ijt[0]
#        V[0] = u0
#        chosen_idx[0] = np.argmax(u0)
#        x_chosen[0] = X_jt[0, chosen_idx[0]]
#
#        for t in range(1, T):
#            u = np.zeros(J)
#            X_jt[t] = np.array(prod_space)
#            a[t] = (x_bar[t] - x_chosen[t-1])**2
#            x_bar[t] = np.mean(x_chosen[:t])
#            Sigma = X_jt[t] - x_bar[t]
#            u = beta*X_jt[t] + gamma*Sigma**2 + kappa*a[t] + epsilon_ijt[t] # love variety
#            V[t] = u
#            chosen_idx[t] = np.argmax(V[t])
#            x_chosen[t] = X_jt[t, chosen_idx[t]] 
#
#        log_denom = logsumexp(V, axis=1) # shape (T,)
#        prob = np.exp(V - log_denom[:, None])
#        ll = -np.sum(V[np.arange(T), chosen_idx] - log_denom)
#        
#        prob_S[s] = prob
#        x_chosen_S[s] = x_chosen
#        x_bar_S[s] = x_bar
#        X_jt_S[s] = X_jt
#        V_S[s] = V
#        U_S[s] = V[np.arange(T), chosen_idx]
#        ll_S[s] = ll
#    return ConsResult(
#            x_chosen_S.mean(axis=0),  # shape (T,)
#            x_bar_S.mean(axis=0),      # shape (T,)
#            X_jt_S.mean(axis=0),       # shape (T, J)
#            V_S.mean(axis=0),          # shape (T, J)
#            prob_S.mean(axis=0),             # shape (T,J)
#            U_S.mean(axis=0),              # shape (T,)
#            ll_S.mean())               # scalar
#
#colors = plt.cm.tab10(np.linspace(0,1,len(regimes)))
#
## Consumer Choice by Regime
#
#for j in range(J):
#    fig, ax = plt.subplots(figsize=(8, 4))
#    for (beta, gamma, label, ls), color in zip(regimes, colors):
#        x_chosen, x_bar, X_jt, V, prob, U, ll = simulate_cons_markov(
#            beta, gamma, kappa, sigma_gamma[1], T, T_prior, J, sigma_x, rng, S)
#        prob_smooth = np.convolve(prob[:, j], np.ones(5)/5, mode='same')
#        ax.plot(np.arange(T), prob_smooth, label=label, color=color, linewidth=1.5, linestyle=ls)
#
#    ax.set_title(f"Product {j+1} (X={prod_space[j]:.1f})", fontsize=9)
#    ax.set_xlabel("Period")
#    ax.set_xlim(5, 95)
#    ax.set_ylabel("Choice Probability")
#    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
#    fig.suptitle("Choice Probability by Product and Regime", fontsize=13, y=1.01)
#    plt.tight_layout()
#    plt.savefig(f"prior_mean_choice_prob_markov_product_{j+1}.pdf", bbox_inches='tight', format='pdf')
#    plt.close()
#
#print(np.var(x_chosen, axis=0))
#print(np.var(x_bar, axis=0))
#print(np.var(prob, axis=0))
