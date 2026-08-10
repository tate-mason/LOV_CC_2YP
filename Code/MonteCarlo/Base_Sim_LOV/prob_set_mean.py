import pandas as pd
import os
import numpy as np
import scipy as sp
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import namedtuple

"""
This file will be used to do initial simulation of the love of variety problem:
    - Consumer Problem: U_ijt = beta_i*X_jt - gamma_i*(Sigma_jt) + a_ijt + epsilon_ijt
        s.t. gamma ~ N(0, sigma_gamma); Sigma_jt = X_jt - 1/t*sum_{k=1}^t x_kt 
    - Firm Problem: V(a_t;Sigma_jt, lambda_t(gamma_i)) = {pi + int_gamma V(a';Sigma_jt+1, lambda_t+1(gamma_i))dlambda_t(gamma_i)}
        s.t. pi = (1+a)p - c
             lambda_t(gamma) ~ N(1/Sigma_jt, sigma_lambda), a = {0,1}
Structure:
    1) solve consumer's choice problem
    2) compute individual shares of products
    3) compute firm's problem, choosing optimal a
    4) find partial equilibrium
Landscape:
    1) T = 100
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
T = 100
sigma_gamma = [0.5, 1.0, 1.5]
kappa = 3

regimes = [
    (1.30, 0.50, "β=1.3, γ=0.5", '-'),
    (1.30, 1.30,   "β=1.3, γ=1.3", '-'),
    (2.00, 1.30, "β=2.0, γ=1.3", '-.'),
    (2.00, 2.00, "β=2.0, γ=2.0", '--'),
    (3.00, 2.50, "β=3.0, γ=2.5", '--'),
    (1.30, 1.00, "β=1.3, γ=1.0", '-.'),
    (3.30, 2.30 , "β=3.3, γ=2.3", '-.'),
]

# 15 products evenly spaced between 1 and 5
prod_space = np.linspace(1, 5, 5)
J = len(prod_space)
S = 1000
sigma_x = rng.uniform(0.5, 1.5)
X_bar = np.mean(prod_space)

ConsResult = namedtuple('ConsResult', ['x_chosen_all', 'x_bar_all', 'X_jt_all', 'V_all', 'prob_all', 'U_all', 'll'])

# Consumer Choice Problem

def simulate_cons(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S=1000):
    x_chosen_all = np.zeros((S, T))
    x_bar_all    = np.zeros((S, T))
    X_jt_all     = np.zeros((S, T, J))
    V_all        = np.zeros((S, T, J))
    ll_all       = np.zeros(S)
    prob_all     = np.zeros((S, T, J))
    U_all = np.zeros((S, T))

    for s in range(S):

        epsilon_ijt = rng.gumbel(0,1, size=(T,J))

        x_chosen = np.zeros(T) # empty vector of choices per period
        X_jt = np.zeros((T,J))
        x_bar = np.zeros(T)

        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)

        X_jt[0] = np.array(prod_space)
        a = np.zeros(T)

        a[0] = 0
        x_bar[0] = 2.5
        u0 = beta*X_jt[0] - gamma*0 + a[0] + epsilon_ijt[0]
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
            a[t] = 0.0
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta*X_jt[t] + gamma*np.log(1 + Sigma**2) + kappa*a[t] + epsilon_ijt[t] # love variety
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx[t]] 

        log_denom = logsumexp(V, axis=1) # shape (T,)
        prob = np.exp(V - log_denom[:, None])
        ll = -np.sum(V[np.arange(T), chosen_idx] - log_denom)
        
        prob_all[s] = prob
        x_chosen_all[s] = x_chosen
        x_bar_all[s] = x_bar
        X_jt_all[s] = X_jt
        V_all[s] = V
        U_all[s] = V[np.arange(T), chosen_idx]
        ll_all[s] = ll
    return ConsResult(
            x_chosen_all.mean(axis=0),  # shape (T,)
            x_bar_all.mean(axis=0),      # shape (T,)
            X_jt_all.mean(axis=0),       # shape (T, J)
            V_all.mean(axis=0),          # shape (T, J)
            prob_all.mean(axis=0),             # shape (T,J)
            U_all.mean(axis=0),              # shape (T,)
            ll_all.mean())               # scalar
snr_by_regime = {}

for (beta, gamma, label, ls) in regimes:
    snr_draws = np.zeros(S)
    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T, J))
        x_chosen = np.zeros(T)
        X_jt = np.zeros((T, J))
        x_bar = np.zeros(T)
        V = np.zeros((T, J))
        chosen_idx = np.zeros(T, dtype=int)
        X_jt[0] = np.array(prod_space)
        x_bar[0] = 2.5
        u0 = beta*X_jt[0] + gamma*np.log(1 + (X_jt[0] - 2.5)**2) + epsilon_ijt[0]
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
        snr_draws[s] = (V[50].max() - V[50].min()) / np.sqrt(np.pi**2 / 6)
    snr_by_regime[label] = (snr_draws.mean(), snr_draws.std())

print("\nSNR Summary by Regime")
print(f"{'Regime':<20} {'Mean SNR':>10} {'Std SNR':>10} {'Min SNR':>10}")
print("-" * 52)
for label, (mean, std) in snr_by_regime.items():
    print(f"{label:<20} {mean:>10.3f} {std:>10.3f}")
colors = plt.cm.tab10(np.linspace(0,1,len(regimes)))

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("../Output/fixed_mean_choice_prob_all_products.pdf") as pdf:
    for j in range(J):
        fig, ax = plt.subplots(figsize=(8, 4))
        for (beta, gamma, label, ls), color in zip(regimes, colors):
            x_chosen, x_bar, X_jt, V, prob, U, ll = simulate_cons(
                beta, gamma, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng
            )
            prob_smooth = np.convolve(prob[:, j], np.ones(5)/5, mode='same')
            ax.plot(np.arange(T), prob_smooth, label=label, color=color, linewidth=1.5, linestyle=ls)
        ax.set_title(f"Product {j+1} (X={prod_space[j]:.1f})", fontsize=9)
        ax.set_xlabel("Period")
        ax.set_xlim(0, 100)
        ax.set_ylabel("Choice Probability")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 10))
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"Mean Utility: {U.mean():.4f}")

print(f"SNR = {(U.max() - U.min()) / np.sqrt(np.pi**2 / 6):.4f}") # SNR = (mean utility)^2 / (variance of Gumbel noise)

# Want SNR > 2 to ensure that the signal from the utility is strong enough relative to the noise.



#def simulate_cons_markov(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S=1000):
#    x_chosen_all = np.zeros((S, T))
#    x_bar_all    = np.zeros((S, T))
#    X_jt_all     = np.zeros((S, T, J))
#    V_all        = np.zeros((S, T, J))
#    ll_all       = np.zeros(S)
#    prob_all     = np.zeros((S, T, J))
#    U_all = np.zeros((S, T))
#
#    for s in range(S):
#
#        epsilon_ijt = rng.gumbel(0,1, size=(T,J))
#
#        x_chosen = np.zeros(T) # empty vector of choices per period
#        X_jt = np.zeros((T,J))
#        x_bar = 5
#        #x_bar[0] = np.mean(prod_space)
#
#
#        V = np.zeros((T, J))
#        chosen_idx = np.zeros(T, dtype=int)
#
#        X_jt[0] = np.array(prod_space)
#        a = np.zeros(T)
#
#        a[0] = 0
#        u0 = beta*X_jt[0] - gamma*0 + kappa*a[0] + epsilon_ijt[0]
#        V[0] = u0
#        x_chosen[0] = X_jt[0, chosen_idx[0]]
#
#        for t in range(1, T):
#            X_jt[t] = np.array(prod_space)
#            u = np.zeros(J)
#            a[t] = 0
#            Sigma = (X_jt[t] - x_bar)
#            u = beta*X_jt[t] + gamma*Sigma**2 + kappa*a[t] + epsilon_ijt[t]
#            V[t] = u
#            chosen_idx[t] = np.argmax(V[t])
#            x_chosen[t] = X_jt[t, chosen_idx[t]] 
#
#        
#        log_denom = logsumexp(V, axis=1)
#        prob = np.exp(V - log_denom[:,None])
#        ll = -np.sum(V[np.arange(T), chosen_idx] - log_denom)
#        
#        prob_all[s] = prob
#        x_chosen_all[s] = x_chosen
#        x_bar_all[s] = x_bar
#        X_jt_all[s] = X_jt
#        V_all[s] = V
#        U_all[s] = V[np.arange(T), chosen_idx]
#        ll_all[s] = ll
#    return ConsResult(
#            x_chosen_all.mean(axis=0),  # shape (T,)
#            x_bar_all.mean(axis=0),      # shape (T,)
#            X_jt_all.mean(axis=0),       # shape (T, J)
#            V_all.mean(axis=0),          # shape (T, J)
#            prob_all.mean(axis=0),             # shape (T,J)
#            U_all.mean(axis=0),              # shape (T,)
#            ll_all.mean()
#            )               
## Utility Plots
#colors = plt.cm.tab10(np.linspace(0,1,len(regimes)))
#
## Utility Plots
#fig, ax = plt.subplots(figsize=(8, 4))
#colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
#
#for (beta, gamma, label, ls), color in zip(regimes, colors):
#    x_chosen, x_bar, X_jt, V, prob_all, U, ll = simulate_cons_markov(
#        beta, gamma, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng
#    )
#    ax.plot(np.arange(T), U, label=label, color=color, linewidth=1.5, linestyle=ls)  # <-- ax.plot, not plt.plot
#
#ax.set_xlabel("Period")
#ax.set_ylabel("Utility")
#ax.set_title("Consumer Utility by Regime (Markov)")
## put legend in top right
#ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1,1))
#plt.tight_layout()
#plt.savefig("consumer_utility_by_regime_markov.png", bbox_inches='tight', format='png')
#plt.close()
#
#
#for j in range(J):
#    fig, ax = plt.subplots(figsize=(8, 4))
#    for (beta, gamma, label, ls), color in zip(regimes, colors):
#        x_chosen, x_bar, X_jt, V, prob, U, ll = simulate_cons_markov(
#            beta, gamma, kappa, sigma_gamma[1], T, J, X_bar, sigma_x, rng
#        )
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
#    plt.savefig(f"choice_prob_product_{j+1}_markov.png", bbox_inches='tight', format='png')
#    plt.close()
#
##=============================================================#
## Firm Value Function w.r.t. CCP                              #
##=============================================================#
#
#p = 2.0
#mc = 1.0
#beta_disc = 0.9
#
#def firm_value_function(beta, gamma, kappa, beta_disc, mc, sigma_gamma, T, J, X_bar, sigma_x, rng, p, S=1000):
#
#    cons = simulate_cons(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S)
#
#    prob = cons.prob_all
#    X_bar_avg = cons.x_bar_all
#    x_chosen = cons.x_chosen_all
#    X_jt = cons.X_jt_all
#
#    Vf = np.zeros((T+1, J))
#    a_opt = np.zeros((T,J))
#    V_diff = np.zeros((T,J))
#
#    for t in range(T-1, -1, -1):
#        eps_f = rng.standard_normal(size=J)
#        for j in range(J):
#            X_kt = X_jt[t].sum() - X_jt[t,j] 
#            s_jt = prob[t,j]
#            for a_val in [0,1]:
#                ad_cost = a_val
#                pi = s_jt*(p-mc) - kappa*ad_cost**2
#                EV_f = Vf[t+1, j]
#                V_t = pi + beta_disc*EV_f
#                if a_val == 0:
#                    V_no_ad = V_t
#                else:
#                    V_ad = V_t
#
#            V_diff[t,j] = V_ad - V_no_ad
#
#            if V_ad >= V_no_ad:
#                Vf[t,j] = V_ad
#                a_opt[t,j] = 1
#            else:
#                Vf[t,j] = V_no_ad
#                a_opt[t,j] = 0
#    return Vf, a_opt, V_diff
#
#from scipy.ndimage import uniform_filter1d
#
#fig, axes = plt.subplots(1,2,figsize=(14, 4))
#for (beta, gamma, label, ls), color in zip(regimes, colors):
#    Vf, a, V_diff = firm_value_function(
#        beta, gamma, kappa, beta_disc, mc, sigma_gamma, T, J, X_bar, sigma_x, rng, p, 1000
#    )
#    smoothed = uniform_filter1d(a.mean(axis=1), size=10)  # Smooth the average advertising choice over time
#    axes[0].plot(np.arange(T), smoothed, label=label, color=color, linewidth=1.5, linestyle=ls)  # <-- ax.plot, not plt.plot
#    axes[1].plot(np.arange(T), V_diff.mean(axis=1), label=label, color=color, linewidth=1.5, linestyle=ls)  # <-- ax.plot, not plt.plot
#
#axes[0].set_xlabel("Period")
#axes[0].set_ylabel("Pr(Advertise)")
#axes[0].set_title("Smoothed Ad Probability (Fixed)")
#axes[0].axhline(0.5, color='k', linewidth=0.5, linestyle=':')  # indifference line
#
#axes[1].set_xlabel("Period")
#axes[1].set_ylabel("V(ad) - V(no ad)")
#axes[1].set_title("Ad Incentive Over Time (Fixed)")
#axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')  # zero = indifference
#
#handles, labels_leg = axes[1].get_legend_handles_labels()
#fig.legend(handles, labels_leg, fontsize=7, loc='lower center',
#           bbox_to_anchor=(0.5, -0.15), ncol=3)
#
#plt.tight_layout()
#plt.savefig("ad_choice_both.pdf", bbox_inches='tight', format='png')
#plt.close()
#
#def simulate_firm_markov(beta, gamma, kappa, beta_disc, mc, sigma_gamma, T, J, X_bar, sigma_x, rng, p, S=1000):
#    
#    cons = simulate_cons_markov(beta, gamma, kappa, sigma_gamma, T, J, X_bar, sigma_x, rng, S)
#
#    prob = cons.prob_all
#    X_bar = cons.x_bar_all
#    x_chosen = cons.x_chosen_all
#    X_jt = cons.X_jt_all
#
#    Vf = np.zeros((T+1, J))
#    a_opt = np.zeros((T,J))
#    V_diff = np.zeros((T,J))
#
#    for t in range(T-1, -1, -1):
#        eps_f = rng.standard_normal(size=J)
#        for j in range(J):
#            X_kt = X_jt[t].sum() - X_jt[t,j] 
#            s_jt = prob[t,j]
#            markov_a = kappa*(X_bar[t] - x_chosen[t])**2
#            for a_val in [0,markov_a]:
#                ad_cost = a_val
#                pi = s_jt*(p-mc) - ad_cost
#                EV_f = Vf[t+1, j]
#                V_t = pi + beta_disc*EV_f
#                if a_val == 0:
#                    V_no_ad = V_t
#                else:
#                    V_ad = V_t
#
#            V_diff[t,j] = V_ad - V_no_ad
#
#            if V_ad >= V_no_ad:
#                Vf[t,j] = V_ad
#                a_opt[t,j] = 1
#            else:
#                Vf[t,j] = V_no_ad
#                a_opt[t,j] = 0
#    return Vf, a_opt, V_diff
#
#fig, axes = plt.subplots(1,2,figsize=(14, 4))
#for (beta, gamma, label, ls), color in zip(regimes, colors):
#    Vf_m, a_m, V_diff_m = simulate_firm_markov(
#        beta, gamma, kappa, beta_disc, mc, sigma_gamma, T, J, X_bar, sigma_x, rng, p, 1000
#    )
#    smoothed_m = uniform_filter1d(a_m.mean(axis=1), size=10)  # Smooth the average advertising choice over time
#    axes[0].plot(np.arange(T), smoothed_m, label=label, color=color, linewidth=1.5, linestyle=ls)
#    axes[1].plot(np.arange(T), V_diff_m.mean(axis=1), label=label, color=color, linewidth=1.5, linestyle=ls) 
#
#axes[0].set_xlabel("Period")
#axes[0].set_ylabel("Pr(Advertise)")
#axes[0].set_title("Smoothed Ad Probability (Markov)")
#axes[0].axhline(0.5, color='k', linewidth=0.5, linestyle=':')  # indifference line
#
#axes[1].set_xlabel("Period")
#axes[1].set_ylabel("V(ad) - V(no ad)")
#axes[1].set_title("Ad Incentive Over Time (Markov)")
#axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')  # zero = indifference
#
#handles, labels_leg = axes[1].get_legend_handles_labels()
#fig.legend(handles, labels_leg, fontsize=7, loc='lower center',
#           bbox_to_anchor=(0.5, -0.15), ncol=3)
#
#plt.tight_layout()
#plt.savefig("ad_choice_markov_both.png", bbox_inches='tight', format='png')
#plt.close()
#
