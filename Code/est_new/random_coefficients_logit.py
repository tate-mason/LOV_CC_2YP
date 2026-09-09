"""Beginner example: mixed logit with one choice per person.

Run: python3 random_coefficients_logit.py
Read GUIDE.md for a line-by-line explanation and the mathematics.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def log_choice_probabilities(price, quality, beta_price, beta_quality):
    """Conditional log probabilities, one column per alternative."""
    utility = beta_price * price + beta_quality * quality
    return utility - logsumexp(utility, axis=-1, keepdims=True)


def simulate_data(n_people=2000, n_alternatives=3, seed=42):
    """Make a fake dataset so this example runs without a data file."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(1.0, 5.0, size=(n_people, n_alternatives))
    quality = rng.uniform(0.0, 3.0, size=(n_people, n_alternatives))

    true_mean_price = -1.0
    true_sd_price = 0.5
    true_beta_quality = 0.8

    beta_price = rng.normal(true_mean_price, true_sd_price, size=(n_people, 1))
    log_p = log_choice_probabilities(price, quality, beta_price, true_beta_quality)
    probabilities = np.exp(log_p)
    choices = np.array([
        rng.choice(n_alternatives, p=probabilities[i])
        for i in range(n_people)
    ])
    return price, quality, choices


def negative_log_likelihood(theta, price, quality, choices, draws):
    """Negative simulated log likelihood, the number we minimize."""
    mean_price, log_sd_price, beta_quality = theta
    sd_price = np.exp(log_sd_price)
    n_people, n_draws = draws.shape

    beta_price = mean_price + sd_price * draws
    log_p = log_choice_probabilities(
        price[:, None, :],
        quality[:, None, :],
        beta_price[:, :, None],
        beta_quality,
    )
    chosen_log_p = log_p[np.arange(n_people), :, choices]
    log_average_p = logsumexp(chosen_log_p, axis=1) - np.log(n_draws)
    return -np.sum(log_average_p)


def main():
    price, quality, choices = simulate_data()
    n_people = price.shape[0]
    n_draws = 200

    # Keep these draws FIXED while the optimizer tries different parameters.
    rng = np.random.default_rng(123)
    draws = rng.standard_normal(size=(n_people, n_draws))
    initial_guess = np.array([-0.5, np.log(0.3), 0.5])

    result = minimize(
        negative_log_likelihood,
        x0=initial_guess,
        args=(price, quality, choices, draws),
        method="L-BFGS-B",
        bounds=[(None, None), (-5.0, 2.0), (None, None)],
        options={"maxiter": 200},
    )

    print("Optimizer converged:", result.success)
    print("Optimizer message:", result.message)
    if not result.success:
        print("Treat the estimates cautiously: optimization did not converge.")

    mean_price, log_sd_price, beta_quality = result.x
    print("\nParameter             True value    Estimate")
    print(f"Mean price effect        -1.000    {mean_price:8.3f}")
    print(f"SD of price effect        0.500    {np.exp(log_sd_price):8.3f}")
    print(f"Quality effect            0.800    {beta_quality:8.3f}")
    print(f"Negative log likelihood: {result.fun:.3f}")


if __name__ == "__main__":
    main()
