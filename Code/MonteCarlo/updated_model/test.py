import numpy as np

rng = np.random.default_rng(219)

S = 1000
T = 100
J = 5

x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
price = np.array([1.0, 1.2, 1.1, 1.3, 1.4])

beta_0 = 0.5  # base utility
lam = 0.6  # weighting on satiation
delta = 0.9  # discount factor

choice_count = np.zeros((T, J + 1))

for i in range(S):
    beta_i = rng.normal(0.0, 0.4)
    alpha_i = -abs(rng.normal(1.0, 0.1))
    gamma_i = rng.choice([0.0, 0.5, 1.0])

    xi = np.zeros((T, J))
    prev_choice = 0  # outside option to start

    for t in range(T):
        for j in range(J):
            for s in range(max(0, t - 8), t):
                xi[t, j] += lam * delta**s * xi[s, j]

            if prev_choice != 0 and prev_choice != j + 1:
                xi[t, j] += (1 - lam) * x[prev_choice - 1]

        beta_ijt = beta_0 + beta_i + gamma_i * xi[t]
        eps = rng.gumbel(0.0, 1.0, size=J + 1)

        utility_prod = beta_ijt * x + alpha_i * price + eps[1:]
        utility = np.concatenate(([eps[0]], utility_prod))

        choice = np.argmax(utility)
        choice_count[t, choice] += 1
        prev_choice = choice


Q = choice_count / S

print("Product Attributes:", x)
print("Period 1 shares (outside, prod 1-5)", np.round(Q[0], 3))
print("Final period shares (outside, prod 1-5", np.round(Q[-1], 3))
