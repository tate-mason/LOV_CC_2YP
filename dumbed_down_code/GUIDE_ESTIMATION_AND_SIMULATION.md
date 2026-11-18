# Guide 2: Demand Estimation & Dataset Simulation

This guide provides the **econometric theory** and **computational documentation** for:
- [simulate_dataset.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/simulate_dataset.py) (Data Generator)
- [estimation_functions.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/estimation_functions.py) (Likelihood Functions)
- [run_estimation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/run_estimation.py) (Nelder-Mead Estimation)

---

## 1. The Demand Model Specification

Consumers make repeated choices over shopping weeks $t = 1, \dots, T$.
In any given week, a consumer chooses one of four mutually exclusive options:
- **Index 0: Other Yogurt** (Reference flavor, $\beta_{\text{other}} = 0$)
- **Index 1: Berry Yogurt** (Taste premium $\beta_{\text{berry}}$)
- **Index 2: Plain Yogurt** (Taste premium $\beta_{\text{plain}}$)
- **Index 3: Outside Option** (No yogurt, utility normalized to $0$)

### Utility Formulation
Consumer $i$'s utility from alternative $j \in \{0, 1, 2\}$ in week $t$ is:
$$U_{ijt} = V_{ijt} + \varepsilon_{ijt}$$
$$V_{ijt} = \beta_0 + \beta_j + \alpha_i p_{ijt} + \gamma_i N_{ijt}$$
where:
- $\beta_0$: Base inside intercept (utility from consuming any yogurt vs. outside option).
- $\beta_j$: Taste premium for flavor $j$ relative to Other ($\beta_{\text{other}} = 0$).
- $p_{ijt}$: Price of flavor $j$ faced by household $i$ in week $t$.
- $\alpha_i$: Household-specific price sensitivity ($\alpha_i < 0$).
- $N_{ijt}$: Novelty / product-switching measure.
- $\gamma_i$: Household-specific taste for novelty.
- $\varepsilon_{ijt}$: Independent standard Type-I Extreme Value shock.

For the outside option ($j = 3$, no yogurt):
$$U_{i, \text{outside}, t} = 0 + \varepsilon_{i, \text{outside}, t}$$

---

## 2. Novelty and Switching ($N_{ijt}$)

In the primary no-habit specification, switching preferences are captured by **novelty**:
$$N_{ijt} = 1 - w_{ij, t-1}$$
where $w_{ij, t-1}$ is the share of flavor $j$ in the household's basket in the previous week:
- If last week the household bought flavor $j$: $w_{ij, t-1} = 1 \implies N_{ijt} = 0.0$ (no novelty bonus).
- If last week the household bought a different flavor $k \ne j$: $w_{ij, t-1} = 0 \implies N_{ijt} = 1.0$ (full novelty bonus).
- If last week the household bought the outside option (no yogurt): $N_{ijt} = 0.0$ (no recent flavor history).

There is **no habit stock** ($\phi_i = 0$), so preferences depend only on the immediate previous purchase.

---

## 3. Unobserved Consumer Heterogeneity (Random Coefficients)

Different households have different price sensitivities and switching tastes:
1. **Price Sensitivity ($\alpha_i < 0$)**:
   To ensure downward-sloping demand ($\alpha_i < 0$ for all households), we assume $-\alpha_i$ follows a lognormal distribution:
   $$\alpha_i = -\exp(\mu_\alpha + \sigma_\alpha z_{\alpha, i}), \quad z_{\alpha, i} \sim \mathcal{N}(0, 1)$$
2. **Switching Taste ($\gamma_i$)**:
   Switching utility is unrestricted in sign and follows a normal distribution:
   $$\gamma_i = \mu_\gamma + \sigma_\gamma z_{\gamma, i}, \quad z_{\gamma, i} \sim \mathcal{N}(0, 1)$$

---

## 4. The Simulation Process (`simulate_dataset.py`)

[simulate_dataset.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/simulate_dataset.py) generates synthetic panel data so you can run the estimation locally without needing restricted research server files.

### Step-by-Step Simulation Logic:
1. **Fix Ground-Truth Parameters**:
   - $\beta_0 = 0.20, \beta_{\text{berry}} = -0.50, \beta_{\text{plain}} = -0.80$
   - $\mu_\alpha = 0.40, \sigma_\alpha = 0.20$
   - $\mu_\gamma = -0.80, \sigma_\gamma = 0.50$
2. **Draw Household Types**:
   For each household $i$, draw persistent shocks $z_{\alpha, i}$ and $z_{\gamma, i}$.
3. **Loop Over Weeks**:
   - Draw random quoted prices $p_{ijt} \in [\$1.80, \$3.20]$.
   - Calculate novelty $N_{ijt}$ based on the household's choice in week $t-1$.
   - Calculate deterministic utilities $V_{ijt}$.
   - Compute choice probabilities:
     $$P_{ijt} = \frac{\exp(V_{ijt})}{\sum_{k=0}^3 \exp(V_{ikt})}$$
   - Draw a random number $u \in [0, 1]$ to simulate which option the household buys.
   - Record the chosen product and update history for week $t+1$.
4. **Save to CSV**:
   Outputs `simulated_panel_data.csv` with columns: `household_id`, `week`, prices, novelties, and chosen product.

---

## 5. Simulated Maximum Likelihood (SML) Theory

Given observed choices $y_{it} \in \{0, 1, 2, 3\}$, how do we estimate the parameters $\theta = (\beta_0, \beta_{\text{berry}}, \beta_{\text{plain}}, \mu_\alpha, \sigma_\alpha, \mu_\gamma, \sigma_\gamma)$?

### 1. Choice Probability for One Draw
Conditional on a specific draw of $(\alpha, \gamma)$:
$$P_{i, y_{it}, t}(\alpha, \gamma) = \frac{\exp(V_{i, y_{it}, t})}{\sum_{k=0}^3 \exp(V_{ikt})}$$

### 2. Household Panel Likelihood
Because household preferences $(\alpha_i, \gamma_i)$ persist over time, the probability of observing household $i$'s entire sequence of choices over $T$ weeks is:
$$L_i(\alpha, \gamma) = \prod_{t=1}^T P_{i, y_{it}, t}(\alpha, \gamma)$$

### 3. Integrating Over Heterogeneity (Simulation)
Because we do not observe $(\alpha_i, \gamma_i)$, we integrate over their population distribution using $R$ simulation draws:
$$\widehat{L}_i(\theta) = \frac{1}{R} \sum_{r=1}^R L_i\left(\alpha(z_{\alpha, r}), \gamma(z_{\gamma, r})\right)$$

### 4. Antithetic Acceleration
Rather than drawing $R$ completely independent normals, we draw $R/2$ pairs $(z_\alpha, z_\gamma)$ and include their mirror reflections $(-z_\alpha, -z_\gamma)$. This antithetic pairing forces sample mean to zero and dramatically lowers simulation variance.

### 5. Negative Log-Likelihood Objective
We sum the log of simulated likelihoods across all households:
$$\text{NLL}(\theta) = -\sum_{i=1}^N \log \widehat{L}_i(\theta)$$
The parameter estimate $\widehat{\theta}$ minimizes $\text{NLL}(\theta)$.

---

## 6. Optimization via Pure-Python Nelder-Mead Simplex

In [run_estimation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/run_estimation.py), we implement the **Nelder-Mead simplex algorithm** in ~40 lines of pure Python:
1. **Initialize Simplex**: Create $D+1$ points in the 7-dimensional parameter space around the starting guess.
2. **Order Points**: Evaluate $\text{NLL}$ at each point, ranking them from best to worst.
3. **Compute Centroid**: Find the center of all points except the worst point.
4. **Reflection**: Reflect the worst point across the centroid into unexplored territory.
5. **Expansion / Contraction**:
   - If the reflected point is better than the best point, expand further.
   - If the reflected point is worse than the second-worst, contract back toward the centroid.
   - If contraction fails, shrink the entire simplex toward the best point.
6. **Convergence**: Stop after iterations are complete and report $\widehat{\theta}$.
