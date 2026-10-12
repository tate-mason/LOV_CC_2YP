import os
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from prettytable import PrettyTable
from scipy.special import logsumexp

os.makedirs("../Output/Plots", exist_ok=True)
os.makedirs("../Output/Tables", exist_ok=True)


def save_tex_table(latex_str, filename):
    with open(f"../Output/Tables/{filename}.tex", "w") as f:
        f.write(latex_str)


rng = np.random.default_rng(seed=219)

# Parameters
S = 1000  # Simulations
T = 100  # Time periods
T_prior = 5  # History formation
t_star = 50  # Reference period for evaluation

prod_space = np.linspace(1, 5, 5)  # Menu of products
J = len(prod_space)  # Number of products

beta = 2  # Quality utility
gamma = np.array([0, 3, 6, 9])  # Variety utility levels

cons_res = namedtuple(
    "cons_res",
    [
        "IV_S",
        "IV_tstar",
        "prob_S",
        "U_S",
    ],
)


def ccp_iv_base(S, T, T_prior, J, beta, gamma):
    IV_S = np.zeros((S, T))
    prob_S = np.zeros((S, T, J))
    U_S = np.zeros((S, T, J))

    for s in range(S):
        epsilon_ijt = rng.gumbel(0, 1, size=(T, J))

        prior_choices = np.zeros(T_prior)
        X_prior = np.array(prod_space)
        x_bar_prior = 0

        # Initial history state formation
        for t in range(T_prior):
            eps_prior = rng.gumbel(0, 1, size=J)
            U_prior = (
                beta * X_prior
                + gamma * np.log(1 + (X_prior - x_bar_prior) ** 2)
                + eps_prior
            )
            chosen_prior = np.argmax(U_prior)
            prior_choices[t] = X_prior[chosen_prior]
            x_bar_prior = np.mean(prior_choices[: t + 1])

        x_chosen = np.zeros(T)
        X_jt = np.zeros((T, J))
        x_bar = np.zeros(T)
        V = np.zeros((T, J))

        X_jt[0] = np.array(prod_space)
        x_bar[0] = x_bar_prior

        u0 = beta * X_jt[0] + gamma * (X_jt[0] - x_bar[0]) ** 2 + epsilon_ijt[0]
        V[0] = u0
        chosen_idx = np.argmax(u0)
        x_chosen[0] = X_jt[0, chosen_idx]

        for t in range(1, T):
            X_jt[t] = np.array(prod_space)
            x_bar[t] = np.mean(x_chosen[:t])
            Sigma = X_jt[t] - x_bar[t]
            u = beta * X_jt[t] + gamma * np.log(1 + Sigma**2) + epsilon_ijt[t]
            V[t] = u
            chosen_idx = np.argmax(V[t])
            x_chosen[t] = X_jt[t, chosen_idx]

        IV = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:, None])

        U_S[s] = V
        IV_S[s] = IV
        prob_S[s] = prob

    IV_tstar = IV_S[:, t_star]

    return cons_res(
        IV_S.mean(axis=0), IV_tstar.mean(), prob_S.mean(axis=0), U_S.mean(axis=0)
    )


# --- Run Simulations across Gammas ---
results_by_gamma = {}
for g in gamma:
    results_by_gamma[g] = ccp_iv_base(
        S=S, T=T, T_prior=T_prior, J=J, beta=beta, gamma=g
    )


# --- Consolidated Master Table Output ---
master_table = PrettyTable()
master_table.title = "CCP and Inclusive Value Comparison across Variety Utility (γ)"
master_table.field_names = ["Metric / Product"] + [f"γ = {g}" for g in gamma]

# Product Choice Probabilities (CCPs)
for j in range(J):
    row = [f"Product {j + 1} (X={prod_space[j]:.1f})"]
    for g in gamma:
        row.append(f"{results_by_gamma[g].prob_S[:, j].mean():.4f}")
    master_table.add_row(row)

# Summary Inclusive Values
iv_mean_row = ["Inclusive Value (Mean)"] + [
    f"{results_by_gamma[g].IV_S.mean():.4f}" for g in gamma
]
iv_tstar_row = [f"IV at t={t_star}"] + [
    f"{results_by_gamma[g].IV_tstar:.4f}" for g in gamma
]

master_table.add_row(iv_mean_row)
master_table.add_row(iv_tstar_row)

print(master_table)
save_tex_table(master_table.get_latex_string(), "tab_lov_master_gamma_comparison")


# --- Plot 1: Mean Utility Across Gamma Levels ---
sns.set_theme(style="whitegrid")
colors = sns.color_palette("Set2", len(gamma))
plt.figure(figsize=(10, 6))

for i, g in enumerate(gamma):
    mean_u_over_time = results_by_gamma[g].U_S.mean(axis=1)
    plt.plot(
        np.arange(T),
        mean_u_over_time,
        label=f"LOV γ={g}",
        color=colors[i],
        linewidth=2,
    )

plt.title("Consumer Utility Over Time across Variety Preferences (γ)", fontsize=13)
plt.xlabel("Time Period")
plt.ylabel("Average Utility")
plt.legend(title="Variety Utility (γ)")
plt.tight_layout()
plt.savefig("../Output/Plots/LOV_Utility_Gamma_Comparison.pdf")
plt.close()


# --- Plot 2: Product Share Dynamics (CCPs) for Each Gamma ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
prod_colors = sns.color_palette("tab10", J)

for idx, g in enumerate(gamma):
    ax = axes[idx // 2, idx % 2]
    prob_S = results_by_gamma[g].prob_S  # shape: (T, J)

    for j in range(J):
        ax.plot(
            np.arange(T),
            prob_S[:, j],
            label=f"Product {j + 1} (X={prod_space[j]:.1f})",
            color=prod_colors[j],
            linewidth=1.8,
        )

    ax.set_title(f"Product Shares over Time (γ = {g})", fontweight="bold")
    ax.set_ylim(0, 1)
    if idx % 2 == 0:
        ax.set_ylabel("Choice Probability (Share)")
    if idx // 2 == 1:
        ax.set_xlabel("Time Period")
    ax.legend(fontsize=8, loc="upper right")

plt.suptitle(
    "Evolution of Product Choice Probabilities Across Gamma Levels",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("../Output/Plots/LOV_Product_Shares_Grid.pdf")
plt.show()
