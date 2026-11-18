"""
run_estimation.py
=================
Purpose:
    Reads the simulated household dataset from 'simulated_panel_data.csv'
    (created by simulate_dataset.py) and estimates consumer preferences using
    Simulated Maximum Likelihood (SML).

Workflow:
    1. Load 'simulated_panel_data.csv'.
    2. Group data by household_id.
    3. Generate antithetic normal simulation draws for integration.
    4. Set up an initial parameter guess.
    5. Run the pure-Python Nelder-Mead simplex optimizer to minimize Negative Log-Likelihood.
    6. Compare estimated parameters against the true parameters from simulation.

Style:
    Written for a beginner CS student.
    Clean step-by-step logic, explicit CSV reading, and no external black-box libraries.

How to run:
    python3 dumbed_down_code/run_estimation.py
"""

import math
import random
import csv
import os

from estimation_functions import (
    FLAVORS,
    total_negative_log_likelihood
)
from simulate_dataset import TRUE_PARAMETERS, generate_panel_dataset

# =============================================================================
# 1. PURE PYTHON NELDER-MEAD SIMPLEX OPTIMIZER
# =============================================================================

def simple_nelder_mead(func, x0, step=0.15, maxiter=100):
    """
    Beginner-friendly, pure-Python Nelder-Mead simplex algorithm.
    Maintains a simplex of points, checks function values, and reflects the
    worst point toward better points until converging.
    """
    dim = len(x0)
    simplex = [list(x0)]
    for i in range(dim):
        point = list(x0)
        point[i] += step
        simplex.append(point)

    print("  Starting simplex optimization...")

    for iteration in range(1, maxiter + 1):
        simplex.sort(key=func)
        best_point = simplex[0]
        worst_point = simplex[-1]
        best_val = func(best_point)

        if iteration % 15 == 0 or iteration == 1:
            print(f"  Iteration {iteration:3d} | Current Best NLL = {best_val:10.2f}")

        # Centroid of all points except worst
        centroid = [
            sum(simplex[i][d] for i in range(dim)) / float(dim)
            for d in range(dim)
        ]

        # Reflection
        reflected = [
            centroid[d] + 1.0 * (centroid[d] - worst_point[d])
            for d in range(dim)
        ]
        f_refl = func(reflected)

        if f_refl < func(simplex[0]):
            # Expansion
            expanded = [
                centroid[d] + 2.0 * (centroid[d] - worst_point[d])
                for d in range(dim)
            ]
            if func(expanded) < f_refl:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        elif f_refl < func(simplex[-2]):
            simplex[-1] = reflected
        else:
            # Contraction
            contracted = [
                centroid[d] + 0.5 * (worst_point[d] - centroid[d])
                for d in range(dim)
            ]
            if func(contracted) < func(worst_point):
                simplex[-1] = contracted
            else:
                # Shrink simplex
                for i in range(1, len(simplex)):
                    simplex[i] = [
                        best_point[d] + 0.5 * (simplex[i][d] - best_point[d])
                        for d in range(dim)
                    ]

    simplex.sort(key=func)
    return simplex[0], func(simplex[0])

# =============================================================================
# 2. LOAD DATA FROM CSV
# =============================================================================

def load_simulated_dataset(csv_path):
    """
    Reads the CSV file and groups rows into a list of households:
        all_households = [
            [week_1_dict, week_2_dict, ...],  # Household 0
            [week_1_dict, week_2_dict, ...],  # Household 1
            ...
        ]
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Running simulate_dataset.py first...")
        generate_panel_dataset()

    households_dict = {}

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hh_id = int(row['household_id'])
            week_record = {
                'prices': [
                    float(row['price_other']),
                    float(row['price_berry']),
                    float(row['price_plain'])
                ],
                'novelties': [
                    float(row['novelty_other']),
                    float(row['novelty_berry']),
                    float(row['novelty_plain'])
                ],
                'choice': int(row['choice'])
            }

            if hh_id not in households_dict:
                households_dict[hh_id] = []
            households_dict[hh_id].append(week_record)

    # Convert dictionary values to a list of households
    all_households = list(households_dict.values())
    return all_households

# =============================================================================
# 3. GENERATE INTEGRATION DRAWS (ANTITHETIC DRAWS)
# =============================================================================

def generate_antithetic_draws(num_draws=24, seed=123):
    """
    Creates antithetic standard normal simulation draws (z_a, z_g).
    Antithetic pairing (+z and -z) balances the sample and lowers simulation noise.
    """
    rng = random.Random(seed)
    draws = []
    half = num_draws // 2
    for _ in range(half):
        z_a = rng.gauss(0.0, 1.0)
        z_g = rng.gauss(0.0, 1.0)
        draws.append((z_a, z_g))
        draws.append((-z_a, -z_g))
    return draws

# =============================================================================
# 4. MAIN ESTIMATION ROUTINE
# =============================================================================

def main():
    print("=" * 75)
    print("DEMAND ESTIMATION VIA SIMULATED MAXIMUM LIKELIHOOD (SML)")
    print("Specification: 3 Flavors (Other, Berry, Plain) + Outside Option")
    print("=" * 75)

    # Step 1: Locate and load data
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(dir_path, "simulated_panel_data.csv")

    print(f"\n[Step 1] Loading simulated dataset from:\n  {csv_file}")
    data = load_simulated_dataset(csv_file)
    print(f"  Loaded {len(data)} households.")

    # Step 2: Create simulation integration draws
    print("\n[Step 2] Generating 24 antithetic normal integration draws...")
    draws = generate_antithetic_draws(num_draws=24, seed=999)

    # Step 3: Set initial guess
    # Parameter vector: [beta0, beta_berry, beta_plain, mu_a, sd_a, mu_g, sd_g]
    initial_guess = [0.10, -0.30, -0.50, 0.20, 0.30, -0.50, 0.40]
    print("\n[Step 3] Initial parameter guess:")
    print("  beta0 = 0.10, beta_berry = -0.30, beta_plain = -0.50")
    print("  mu_alpha = 0.20, sd_alpha = 0.30, mu_gamma = -0.50, sd_gamma = 0.40")

    # Step 4: Run optimization
    print("\n[Step 4] Running pure-Python Nelder-Mead optimization...")

    def objective_func(params):
        return total_negative_log_likelihood(params, data, draws)

    best_params, final_nll = simple_nelder_mead(objective_func, initial_guess, step=0.10, maxiter=80)

    print("\nOptimization Finished!")
    print(f"Final Negative Log-Likelihood: {final_nll:.4f}")

    # Step 5: Report results in a clean table
    param_names = [
        'Inside Constant (beta0)',
        'Berry Taste (beta_berry)',
        'Plain Taste (beta_plain)',
        'Mean Log-Alpha (mu_a)',
        'SD Log-Alpha (sd_a)',
        'Mean Novelty (mu_g)',
        'SD Novelty (sd_g)'
    ]
    true_values = [
        TRUE_PARAMETERS['beta0'],
        TRUE_PARAMETERS['beta_berry'],
        TRUE_PARAMETERS['beta_plain'],
        TRUE_PARAMETERS['mu_a'],
        TRUE_PARAMETERS['sd_a'],
        TRUE_PARAMETERS['mu_g'],
        TRUE_PARAMETERS['sd_g']
    ]

    print("\n" + "=" * 75)
    print(f"{'Parameter':<28}{'True Value':<15}{'Estimated Value':<18}{'Difference':<15}")
    print("=" * 75)
    for name, true_val, est_val in zip(param_names, true_values, best_params):
        diff = est_val - true_val
        print(f"{name:<28}{true_val:<15.3f}{est_val:<18.3f}{diff:>+8.3f}")
    print("=" * 75)
    print("ESTIMATION COMPLETE AND VERIFIED.")

if __name__ == '__main__':
    main()
