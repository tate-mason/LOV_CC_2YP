"""
simulate_dataset.py
===================
Purpose:
    Generates a realistic, synthetic dataset of household yogurt purchases over time.
    Saves the data to 'simulated_panel_data.csv' so run_estimation.py can read it.

Economic Setup:
    - 3 inside flavors: other (0), berry (1), plain (2)
    - 1 outside option: no yogurt (3)
    - Households have heterogeneous (different) preferences:
        * Price sensitivity: alpha_i = -exp(mu_a + sd_a * z_a)  (always negative)
        * Switching preference: gamma_i = mu_g + sd_g * z_g     (can be positive or negative)
    - Novelty:
        * If last week bought flavor k:
            N_ijt = 0.0 for flavor k (not novel)
            N_ijt = 1.0 for all other flavors j != k (novelty boost)
        * If last week bought outside option (no yogurt):
            N_ijt = 0.0 for all flavors

Style:
    Written for a beginner CS student.
    Loops over households and weeks explicitly, writes rows to a standard CSV file,
    and prints out summary statistics so you can inspect the data.

How to run:
    python3 dumbed_down_code/simulate_dataset.py
"""

import math
import random
import csv
import os

from estimation_functions import (
    FLAVORS,
    compute_single_flavor_utility,
    compute_choice_probabilities
)

# =============================================================================
# TRUE GROUND-TRUTH PARAMETERS (USED TO GENERATE DATA)
# =============================================================================
TRUE_PARAMETERS = {
    'beta0': 0.20,       # Base inside constant
    'beta_berry': -0.50, # Taste for berry relative to other
    'beta_plain': -0.80, # Taste for plain relative to other
    'mu_a': 0.40,        # Mean of log(-alpha) -> average alpha around -1.5
    'sd_a': 0.20,        # Heterogeneity in price sensitivity
    'mu_g': -0.80,       # Mean switching preference (negative = habit/switching cost)
    'sd_g': 0.50         # Heterogeneity in switching preferences
}

def generate_panel_dataset(num_households=100, weeks_per_hh=15, seed=42, output_filename="simulated_panel_data.csv"):
    """
    Simulates shopping weeks for a panel of households and writes to CSV.
    """
    rng = random.Random(seed)

    print(f"Generating synthetic panel data for {num_households} households over {weeks_per_hh} weeks...")

    # We will record rows in a CSV file
    # Columns: household_id, week, price_other, price_berry, price_plain,
    #          novelty_other, novelty_berry, novelty_plain, choice, choice_name
    fieldnames = [
        'household_id', 'week',
        'price_other', 'price_berry', 'price_plain',
        'novelty_other', 'novelty_berry', 'novelty_plain',
        'choice', 'choice_name'
    ]

    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, output_filename)

    choice_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hh_id in range(num_households):
            # Step 1: Draw persistent random preferences for this household
            # Standard normal shocks: z_a ~ N(0, 1), z_g ~ N(0, 1)
            z_a = rng.gauss(0.0, 1.0)
            z_g = rng.gauss(0.0, 1.0)

            hh_alpha = -math.exp(TRUE_PARAMETERS['mu_a'] + TRUE_PARAMETERS['sd_a'] * z_a)
            hh_gamma = TRUE_PARAMETERS['mu_g'] + TRUE_PARAMETERS['sd_g'] * z_g

            # State tracking: what did the household buy last week?
            last_purchased_flavor = None  # None means outside option (no yogurt)

            for week in range(weeks_per_hh):
                # Step 2: Generate realistic weekly prices for the 3 inside flavors
                price_other = round(rng.uniform(1.80, 3.20), 2)
                price_berry = round(rng.uniform(1.80, 3.20), 2)
                price_plain = round(rng.uniform(1.80, 3.20), 2)
                prices = [price_other, price_berry, price_plain]

                # Step 3: Compute novelty values N_ijt based on last week's choice
                # If last week bought flavor k: N_k = 0, but other flavors get N = 1
                novelties = [0.0, 0.0, 0.0]
                if last_purchased_flavor is not None:
                    for j in range(3):
                        if j != last_purchased_flavor:
                            novelties[j] = 1.0
                        else:
                            novelties[j] = 0.0

                # Step 4: Compute deterministic utilities for all 4 alternatives
                utilities = []
                for j in range(3):
                    v_j = compute_single_flavor_utility(
                        flavor_index=j,
                        price=prices[j],
                        novelty=novelties[j],
                        beta0=TRUE_PARAMETERS['beta0'],
                        beta_berry=TRUE_PARAMETERS['beta_berry'],
                        beta_plain=TRUE_PARAMETERS['beta_plain'],
                        alpha=hh_alpha,
                        gamma=hh_gamma
                    )
                    utilities.append(v_j)
                utilities.append(0.0)  # Outside option utility is normalized to 0

                # Step 5: Compute Multinomial Logit choice probabilities
                probs = compute_choice_probabilities(utilities)

                # Step 6: Simulate the discrete choice using random sampling
                rand_num = rng.random()
                cumulative_prob = 0.0
                chosen_idx = 3  # Default to outside
                for idx, p in enumerate(probs):
                    cumulative_prob += p
                    if rand_num <= cumulative_prob:
                        chosen_idx = idx
                        break

                choice_counts[chosen_idx] += 1

                # Step 7: Write row to CSV
                writer.writerow({
                    'household_id': hh_id,
                    'week': week,
                    'price_other': price_other,
                    'price_berry': price_berry,
                    'price_plain': price_plain,
                    'novelty_other': novelties[0],
                    'novelty_berry': novelties[1],
                    'novelty_plain': novelties[2],
                    'choice': chosen_idx,
                    'choice_name': FLAVORS[chosen_idx]
                })

                # Step 8: Update history for next week
                if chosen_idx in [0, 1, 2]:
                    last_purchased_flavor = chosen_idx
                else:
                    last_purchased_flavor = None

    print(f"Data saved successfully to: {file_path}")
    print()
    print("=" * 60)
    print("SIMULATED DATASET SUMMARY")
    print("=" * 60)
    total_obs = num_households * weeks_per_hh
    print(f"Total Households:    {num_households}")
    print(f"Weeks per Household: {weeks_per_hh}")
    print(f"Total Observations:  {total_obs}")
    print("\nSimulated Market Shares:")
    for idx, name in enumerate(FLAVORS):
        count = choice_counts[idx]
        share = 100.0 * count / total_obs
        print(f"  {name.capitalize():<10} (Index {idx}): {count:5d} choices ({share:5.1f}%)")
    print("=" * 60)

    return file_path

if __name__ == '__main__':
    generate_panel_dataset()
