# Dumbed Down Code: Dynamic Pricing with Switching Preferences

This folder contains a complete, beginner-friendly reimplementation of the research codebase. It is written in a simple, imperative style (modeled after `LOV_bundle.py`) with explicit loops, straightforward functions, step-by-step arithmetic, and zero black-box dependencies.

---

## 📚 In-Depth Documentation & Theory Guides

We have prepared three comprehensive theory and documentation guides:

1. 📖 **[GUIDE 1: Dynamic Monopoly Pricing & SymPy Derivations](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/GUIDE_THEORY_AND_SYMPY.md)**
   - Theoretical framework: consumer utility, logit choice probabilities, and demand derivatives.
   - Backward induction: Period 2 terminal FOC and Period 1 continuation value.
   - Comparative statics ($dp/d\gamma$) and price ratios ($R = p_1^* / p_2^*$).
   - Full documentation for `sympy_theory_derivations.py` and `theory_pricing_simulation.py`.

2. 📖 **[GUIDE 2: Demand Estimation & Dataset Simulation](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/GUIDE_ESTIMATION_AND_SIMULATION.md)**
   - The 3-flavor demand model (Other, Berry, Plain) + outside option.
   - Novelty variable definition ($N_{ijt} = 1 - w_{ij, t-1}$) without habit stock.
   - Unobserved heterogeneity: lognormal price sensitivity ($\alpha_i < 0$) and normal switching utility ($\gamma_i$).
   - Data simulation process and Simulated Maximum Likelihood (SML) theory.
   - Full documentation for `simulate_dataset.py`, `estimation_functions.py`, and `run_estimation.py`.

3. 📖 **[GUIDE 3: Empirical Pricing: Plain Yogurt vs. All Else](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/GUIDE_PRICING_AND_POLICY_COMPARISON.md)**
   - Empirical application: pricing Plain Yogurt against All Other Choices.
   - Trimming the estimated normal switching distribution (central 90% retained, 5 quintiles).
   - Uniform Pricing (baseline) vs. Price Discrimination (quintile-informed counterfactual).
   - Consumer surplus normalization ($CS^Z$) and replication of Table 3 from `mason_draft.pdf`.
   - Full documentation for `empirical_pricing_plain_vs_all.py`.

---

## 📂 File Directory

| File | Type | Purpose |
|---|---|---|
| [simulate_dataset.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/simulate_dataset.py) | **Data Generator** | Simulates a synthetic household panel dataset and saves it to `simulated_panel_data.csv`. |
| [estimation_functions.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/estimation_functions.py) | **Function Storage** | Helper module containing functions for computing flavor utilities, logit probabilities, and simulated likelihood. |
| [run_estimation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/run_estimation.py) | **Demand Estimation** | Loads `simulated_panel_data.csv`, runs Simulated Maximum Likelihood with a pure-Python Nelder-Mead optimizer, and displays estimated vs. true parameters. |
| [sympy_theory_derivations.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/sympy_theory_derivations.py) | **Symbolic Math** | Derives analytical expressions for probabilities, derivatives, FOCs, and comparative statics using SymPy. |
| [theory_pricing_simulation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/theory_pricing_simulation.py) | **Theoretical Model** | Solves the numerical 2-period model ($\beta=2, \alpha=-1, \gamma \sim \mathcal{N}(0, 1.25^2)$) comparing Uniform vs. Price Discrimination. |
| [empirical_pricing_plain_vs_all.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/empirical_pricing_plain_vs_all.py) | **Empirical Pricing** | Prices Plain Yogurt vs. All Else using calibrated Nielsen estimates, reproducing Table 3 of `mason_draft.pdf`. |

---

## 🚀 How to Run the Code

All scripts are completely self-contained and run with standard Python 3:

```bash
# Step 1: Generate synthetic household panel data
python3 dumbed_down_code/simulate_dataset.py

# Step 2: Estimate demand preferences from the synthetic data
python3 dumbed_down_code/run_estimation.py

# Step 3: Run the symbolic math derivations in SymPy
python3 dumbed_down_code/sympy_theory_derivations.py

# Step 4: Run the theoretical pricing simulation (Uniform vs Informed)
python3 dumbed_down_code/theory_pricing_simulation.py

# Step 5: Run the empirical pricing exercise (Plain Yogurt vs All Else)
python3 dumbed_down_code/empirical_pricing_plain_vs_all.py
```
