import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp, expit

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

prod_space = np.linspace(1,5,5) # menu of products
J = len(prod_space) # number of products

beta = 2 # quality utility
gamma = np.array([0, 6, 9, 12]) # variety utility

cons_res = namedtuple('cons_res', [
    'IV_S',
    'IV_tstar',
    'prob_S',
    'U_S',
])
# ============================================================================================= #

def ccp_iv_base(S, T, T_prior, J, beta, gamma):
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
            X_jt[t] = np.array(prod_space)
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta*X_jt[t] + gamma*np.log(1+Sigma**2) + epsilon_ijt[t]
            V[t] = u
            chosen_idx[t] = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx[t]]
        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:,None])

        U_S[s] = V
        IV_S[s] = IV
        prob_S[s] = prob
        IV_tstar = IV[50]

    return cons_res(
        IV_S.mean(axis=0),
        IV_tstar.mean(),
        prob_S.mean(axis=0),
        U_S.mean(axis=0)
    )

for g in gamma:
    IV_LOV, IV_tstar, prob_LOV, U_lov = ccp_iv_base(S=1000, T=100, T_prior=5, J=5, beta=2, gamma=g)

    tab_LOV_g = PrettyTable()
    tab_LOV_g.title = f"CCP with LOV γ={g}"
    tab_LOV_g.field_names = (['Product', 'CCP'])
    for j in range(J):
        tab_LOV_g.add_row([f"Product {j+1}", round(prob_LOV[:, j].mean(), 4)])
    tab_LOV_g.add_row(["Inclusive Value", round(IV_LOV.mean(),4)])
    tab_LOV_g.add_row([f"IV at t={t_star}", round(IV_tstar, 4)])

    print(tab_LOV_g)
    save_tex_table(tab_LOV_g.get_latex_string(), f"tab_lov_{g}_3")

colors = sns.color_palette("Set2", 4)
plt.figure(figsize=(10,6))
for i, g in enumerate(gamma):
    plt.plot(U_lov, label=f"LOV γ={g}", color=colors[i])
    plt.title("Consumer Utility Over Time")
    plt.xlabel("Time Period")
    plt.ylabel("Utility")
plt.legend()
plt.tight_layout()
plt.savefig('../Output/Plots/LOV_Utility.pdf')
plt.show()
    

#======================================================================================================#

for x in [0, 1, 3, 5, 6]:
    prod_space_new = np.append(prod_space, x)
    J_prime = len(prod_space_new)

    def ccp_iv_intro(S, T, T_prior, t_star, J, beta, gamma):
        x_chosen_S = np.zeros((S, T))
        x_bar_S = np.zeros((S, T))
        X_jt_S = np.zeros((S, T, J+1))
        V_S = np.zeros((S, T, J+1))
        IV_S = np.zeros((S, T))
        prob_S = np.zeros((S, T, J+1))
        U_S = np.zeros((S, T, J+1))
        
        for s in range(S):
            eps = rng.gumbel(0, 1, size=(T,J+1))

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
            X_jt = np.zeros((T,J+1))
            x_bar = np.zeros(T)

            V = np.zeros((T, J+1))
            chosen_idx = np.zeros(T, dtype=int)
            for t in range(0, T):
                if t < t_star:
                    X_jt[t,:J] = prod_space
                    X_jt[t,J] = -np.inf
                    V[t,J] = -np.inf
                    x_bar[t] = x_bar_prior if t==0 else np.mean(x_chosen[:t])
                    Sigma = X_jt[t,:J] - x_bar[t]
                    u = beta*X_jt[t,:J] + gamma*np.log(1+Sigma**2) + eps[t,:J]
                    V[t,:J] = u

                else:
                    X_jt[t] = prod_space_new
                    x_bar[t] = np.mean(x_chosen[:t])
                    Sigma = X_jt[t] - x_bar[t]
                    u = beta*X_jt[t] + gamma*np.log(1+Sigma**2) + eps[t]
                    V[t] = u
                chosen_idx[t] = np.argmax(V[t])
                x_chosen[t] = X_jt[t, chosen_idx[t]]
            IV = logsumexp(V, axis=1)
            prob = np.exp(V - IV[:,None])

            IV_S[s] = IV
            IV_tstar = IV[t_star]
            prob_S[s] = prob
            U_S[s] = V

        return cons_res(
            IV_S.mean(axis=0),
            IV_tstar.mean(),
            prob_S.mean(axis=0),
            U_S.mean(axis=0)
        )

    for g in gamma:
        IV_lov_no, IV_tstar_no, prob_lov_no, U_no = ccp_iv_base(S=1000, T=100, T_prior=5, J=5, beta=2, gamma=g)
        IV_LOV_intro, IV_tstar, prob_LOV_intro, U_intro = ccp_iv_intro(S=1000, T=100, T_prior=5, t_star=50, J=5, beta=2, gamma=g)

        tab_LOV_intro_g = PrettyTable()
        tab_LOV_intro_g.title = f"LOV with Product Introduction (γ={g})"
        tab_LOV_intro_g.field_names = (['Product', 'CCP'])
        for j in range(J_prime):
            tab_LOV_intro_g.add_row([f"Product {j+1}", round(prob_LOV_intro[:, j].mean(), 4)])
        tab_LOV_intro_g.add_row(["$\%\Delta$ Inclusive Value", round((IV_LOV_intro.mean()/IV_lov_no.mean()),4)])
        tab_LOV_intro_g.add_row([f"IV at t={t_star}", round(IV_tstar, 4)])

        print(tab_LOV_intro_g)
        save_tex_table(tab_LOV_intro_g.get_latex_string(), f"tab_lov_intro_{g}_3")

    for g in gamma:
        IV_lov_no, IV_tstar_no, prob_lov_no, U_no = ccp_iv_base(S=1000, T=100, T_prior=5, J=5, beta=2, gamma=g)
        IV_LOV_intro, IV_tstar, prob_LOV_intro, U_intro = ccp_iv_intro(S=1000, T=100, T_prior=5, t_star=50, J=5, beta=2, gamma=g)

        U_plot = U_intro.copy()
        U_plot[:t_star, J] = np.nan

        fig, ax = plt.subplots(figsize=(10,5))

        for j in range(J):
            ax.plot(np.arange(T), U_plot[:, j], label=f"Product {j+1} (X = {prod_space[j]: .1f})", linewidth=1.2)
        ax.plot(np.arange(T), U_plot[:, J], label=f"New Product (X={prod_space_new[J]:.1f})",
                linewidth=1.5, linestyle='--', color='black')

        ax.axvline(t_star, color='red', linestyle=':', linewidth=1.0, label=f"Product Introduction (t={t_star})")
        ax.set_title(f"Per-Product Utility Over Time with Product Introduction (γ={g})")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Utility")
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(f'../Output/Plots/LOV_Utility_Intro_{g}_prod{x}.pdf')
        plt.show()
