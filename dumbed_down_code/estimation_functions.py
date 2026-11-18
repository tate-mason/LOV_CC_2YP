"""
estimation_functions.py
=======================
Purpose:
    Contains all the helper functions for estimating consumer preferences for
    product switching (novelty) and price sensitivity using Simulated Maximum
    Likelihood (SML).

Specification:
    - 3 inside flavors: other (index 0), berry (index 1), plain (index 2)
    - 1 outside option: index 3 (no yogurt)
    - Novelty: N_ijt = share of other flavors in previous week (no habit)
    - Price coefficient: alpha < 0 (lognormal: alpha = -exp(mu_a + sd_a * z_a))
    - Switching coefficient: gamma unrestricted (normal: gamma = mu_g + sd_g * z_g)

Style:
    Written for a beginner CS student.
    Takes the long way: loops over flavors, calculates numerator and denominator
    explicitly, and explains every math operation.
"""

import math

# Ordered names of the alternatives
FLAVORS = ['other', 'berry', 'plain', 'outside']

def compute_single_flavor_utility(flavor_index, price, novelty, beta0, beta_berry, beta_plain, alpha, gamma):
    """
    Computes deterministic utility V for a single flavor:
    
    If outside good (index 3):
        V = 0.0 (normalized base)
    If 'other' (index 0):
        V = beta0 + alpha * price + gamma * novelty
    If 'berry' (index 1):
        V = beta0 + beta_berry + alpha * price + gamma * novelty
    If 'plain' (index 2):
        V = beta0 + beta_plain + alpha * price + gamma * novelty
    """
    if flavor_index == 3:  # Outside option (no yogurt)
        return 0.0

    # Start with base inside intercept
    v = beta0

    # Add flavor-specific taste premium
    if flavor_index == 1:
        v += beta_berry
    elif flavor_index == 2:
        v += beta_plain
    # flavor_index == 0 is 'other' (baseline flavor, so premium is 0.0)

    # Add price disutility: alpha * price (remember alpha < 0)
    v += alpha * price

    # Add switching/novelty utility: gamma * novelty
    v += gamma * novelty

    return v

def compute_choice_probabilities(utilities):
    """
    Given a list of 4 utilities [V_other, V_berry, V_plain, V_outside],
    calculates the Multinomial Logit choice probabilities:
        P_j = exp(V_j) / sum_k exp(V_k)
        
    Takes the long way with a numerical stability trick (subtract max).
    """
    # Numerical stability trick:
    # exp(1000) will crash with an overflow error.
    # To prevent this, we subtract the maximum utility max_v from all utilities.
    # Mathematically: exp(V_j - max_v) / sum exp(V_k - max_v) == exp(V_j) / sum exp(V_k).
    max_v = max(utilities)

    exp_utilities = []
    for v in utilities:
        # Subtract max_v to keep numbers manageable
        shifted_v = v - max_v
        exp_utilities.append(math.exp(shifted_v))

    denominator = sum(exp_utilities)

    probabilities = []
    for ev in exp_utilities:
        p = ev / denominator
        probabilities.append(p)

    return probabilities

def compute_household_likelihood_one_draw(household_weeks, beta0, beta_berry, beta_plain, alpha, gamma):
    """
    Computes the likelihood for ONE household across all their observed weeks,
    for ONE specific random draw of (alpha, gamma).
    
    A household's likelihood over multiple independent weeks is the product:
        L_i(alpha, gamma) = Product_{week t} P_{chosen_flavor, t}
        
    To prevent multiplying many tiny fractions (underflow), we sum log-probabilities:
        log L_i = Sum_{week t} log(P_{chosen_flavor, t})
        L_i = exp(log L_i)
    """
    log_likelihood = 0.0

    for week in household_weeks:
        prices = week['prices']          # List of 3 prices: [p_other, p_berry, p_plain]
        novelties = week['novelties']    # List of 3 novelty values: [N_other, N_berry, N_plain]
        chosen_flavor = week['choice']   # Integer: 0 (other), 1 (berry), 2 (plain), 3 (outside)

        # Step 1: Calculate utilities for the 4 choices
        utilities = []
        for j in range(3):
            v_j = compute_single_flavor_utility(
                flavor_index=j,
                price=prices[j],
                novelty=novelties[j],
                beta0=beta0,
                beta_berry=beta_berry,
                beta_plain=beta_plain,
                alpha=alpha,
                gamma=gamma
            )
            utilities.append(v_j)
            
        # Outside option (index 3) has utility 0.0
        utilities.append(0.0)

        # Step 2: Compute choice probabilities
        probs = compute_choice_probabilities(utilities)

        # Step 3: Find probability of the actual chosen product
        p_chosen = probs[chosen_flavor]

        # Guard against log(0) with a tiny lower bound
        if p_chosen < 1e-15:
            p_chosen = 1e-15

        log_likelihood += math.log(p_chosen)

    return math.exp(log_likelihood)

def compute_household_simulated_likelihood(household_weeks, beta0, beta_berry, beta_plain, mu_a, sd_a, mu_g, sd_g, standard_draws):
    """
    Integrates over persistent unobserved consumer heterogeneity.
    
    We take R simulation draws of standard normal variables (z_alpha, z_gamma).
    For each draw r:
        alpha_r = -exp(mu_a + sd_a * z_a)
        gamma_r = mu_g + sd_g * z_g
        
    Simulated likelihood is the average across all R draws:
        L_hat_i = (1 / R) * Sum_{r=1}^R L_i(alpha_r, gamma_r)
    """
    R = len(standard_draws)
    total_likelihood = 0.0

    for z_a, z_g in standard_draws:
        # Transform standard normal draws into random coefficients
        alpha_r = -math.exp(mu_a + sd_a * z_a)
        gamma_r = mu_g + sd_g * z_g

        # Compute likelihood for this draw
        draw_likelihood = compute_household_likelihood_one_draw(
            household_weeks=household_weeks,
            beta0=beta0,
            beta_berry=beta_berry,
            beta_plain=beta_plain,
            alpha=alpha_r,
            gamma=gamma_r
        )
        total_likelihood += draw_likelihood

    # Return the average across all draws
    average_likelihood = total_likelihood / R
    
    # Keep away from absolute zero so log() does not crash
    if average_likelihood < 1e-300:
        average_likelihood = 1e-300
        
    return average_likelihood

def total_negative_log_likelihood(parameter_vector, all_households_data, standard_draws):
    """
    Objective function for optimization!
    
    Inputs:
        parameter_vector: list/array of 7 numbers:
            [beta0, beta_berry, beta_plain, mu_a, sd_a, mu_g, sd_g]
        all_households_data: list of households (each household has a list of weeks)
        standard_draws: list of (z_a, z_g) pairs
        
    Output:
        Negative Log-Likelihood (we minimize this to find maximum likelihood!)
    """
    beta0 = parameter_vector[0]
    beta_berry = parameter_vector[1]
    beta_plain = parameter_vector[2]
    mu_a = parameter_vector[3]
    sd_a = parameter_vector[4]
    mu_g = parameter_vector[5]
    sd_g = parameter_vector[6]

    # Standard deviation cannot be negative! Add penalty if optimizer tries negative SD.
    if sd_a < 0.01 or sd_g < 0.01:
        return 1e12

    total_nll = 0.0

    for household_weeks in all_households_data:
        sim_lik = compute_household_simulated_likelihood(
            household_weeks=household_weeks,
            beta0=beta0,
            beta_berry=beta_berry,
            beta_plain=beta_plain,
            mu_a=mu_a,
            sd_a=sd_a,
            mu_g=mu_g,
            sd_g=sd_g,
            standard_draws=standard_draws
        )
        # Add negative log likelihood for this household
        total_nll -= math.log(sim_lik)

    return total_nll
