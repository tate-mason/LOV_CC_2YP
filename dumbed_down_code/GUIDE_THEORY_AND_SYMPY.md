# Guide 1: Dynamic Monopoly Pricing & SymPy Derivations

This guide provides the **economic theory** and **computational documentation** for:
- [sympy_theory_derivations.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/sympy_theory_derivations.py) (Symbolic Math)
- [theory_pricing_simulation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/theory_pricing_simulation.py) (Numerical Simulation)

---

## 1. The Economic Environment

We model a single monopolist firm selling a good $j$ against an outside alternative $0$ over two discrete time periods, $t \in \{1, 2\}$.

### Consumers and Preferences
A continuum of consumers live for two periods. Each consumer is characterized by:
1. **Initial History ($h_0 \in \{j, 0\}$)**: What product they purchased immediately before period 1.
2. **Switching Preference ($\gamma_i$)**: A persistent utility taste for variety/novelty.
   - $\gamma_i > 0$: The consumer enjoys variety and gains positive utility from switching products.
   - $\gamma_i < 0$: The consumer has habit/switching costs and dislikes changing products.
3. **Price Sensitivity ($\alpha < 0$)**: Marginal utility of money (homogeneous in the pure theory model).
4. **Base Quality ($\beta > 0$)**: The baseline intrinsic utility of good $j$.

### Consumer Utility Function
In period $t$, conditional on previous purchase $h_{t-1} \in \{j, 0\}$, consumer utility is:
$$u_t(h_t \mid h_{t-1}, \gamma) = \mathbbm{1}\{h_t = j\}(\beta + \alpha p_t) + \gamma \mathbbm{1}\{h_t \ne h_{t-1}\} + \varepsilon_{h_t, t}$$
where $\varepsilon_{h_t, t}$ is an independent Type-I Extreme Value (Gumbel) shock.

Let us evaluate the deterministic utilities for each possible previous choice:

| Previous Choice $h_{t-1}$ | Utility from Good $j$ ($V_j$) | Utility from Outside Option $0$ ($V_0$) | Difference $v = V_j - V_0$ |
|---|---|---|---|
| **Outside ($h_{t-1} = 0$)** | $\beta + \alpha p_t + \gamma$ (switched in) | $0$ (stayed outside) | $\beta + \alpha p_t + \gamma$ |
| **Inside ($h_{t-1} = j$)** | $\beta + \alpha p_t$ (stayed inside) | $\gamma$ (switched out) | $\beta + \alpha p_t - \gamma$ |

We can express the deterministic utility difference compactly using a history indicator $q(h)$:
$$v(p_t, h_{t-1}; \gamma) = \beta + \alpha p_t + q(h_{t-1})\gamma, \quad \text{where } q(0) = +1, \; q(j) = -1$$

### Choice Probabilities (Market Shares)
With Type-I extreme value shocks, the inside choice probability is the standard logistic function:
$$s(p_t, h_{t-1}; \gamma) = \frac{\exp(v)}{1 + \exp(v)} = \frac{1}{1 + \exp(-v)}$$
The outside share is $s_0 = 1 - s$.

### Demand Derivatives
Differentiating the logit share with respect to price and $\gamma$:
$$\frac{\partial s}{\partial p_t} = \alpha s (1 - s) < 0 \quad (\text{since } \alpha < 0)$$
$$\frac{\partial s}{\partial \gamma} = q(h_{t-1}) s (1 - s)$$
- When $h_{t-1} = 0$ ($q = +1$): $\frac{\partial s}{\partial \gamma} > 0$ (higher switching preference increases demand for entering good $j$).
- When $h_{t-1} = j$ ($q = -1$): $\frac{\partial s}{\partial \gamma} < 0$ (higher switching preference decreases demand for repeating good $j$).

---

## 2. Backward Induction and Dynamic Monopoly Pricing

The firm faces constant marginal cost $c$ (normalized to $0$ in the baseline) and discount factor $\delta \in (0, 1]$. Because the game ends at $t=2$, we solve by **backward induction**.

### Period 2 (Terminal Problem)
In period 2, there is no future. Given realized choice $h_1 \in \{j, 0\}$, the firm chooses $p_2 \ge 0$ to maximize static revenue:
$$\max_{p_2 \ge 0} \; s(p_2, h_1; \gamma)(p_2 - c)$$

**First-Order Condition (FOC)**:
$$s_2 + (p_2^* - c) \frac{\partial s_2}{\partial p_2} = 0 \implies s_2 + \alpha s_2(1 - s_2)(p_2^* - c) = 0$$
Dividing by $s_2$:
$$1 + \alpha (1 - s_2^*)(p_2^* - c) = 0 \implies p_2^*(h_1) = c - \frac{1}{\alpha [1 - s_2^*(h_1)]}$$
The resulting continuation profit is:
$$\pi_2^*(h_1) = s_2^*(p_2^* - c) = -\frac{s_2^*(h_1)}{\alpha [1 - s_2^*(h_1)]}$$

### Period 1 (Introductory Problem)
In period 1, the firm chooses $p_1 \ge 0$ knowing that today's purchase $h_1$ determines tomorrow's state:
- If the consumer buys good $j$ ($h_1 = j$, probability $s_1$), continuation profit is $\pi_2^*(j)$.
- If the consumer buys outside $0$ ($h_1 = 0$, probability $1 - s_1$), continuation profit is $\pi_2^*(0)$.

Total two-period expected profit is:
$$\Pi(p_1; h_0) = s_1(p_1 - c) + \delta \left[ s_1 \pi_2^*(j) + (1 - s_1) \pi_2^*(0) \right]$$
Define $\Delta \pi_2 = \pi_2^*(j) - \pi_2^*(0)$. We can rewrite total profit as:
$$\Pi(p_1; h_0) = \delta \pi_2^*(0) + s_1 \left[ p_1 - c + \delta \Delta \pi_2 \right]$$

**First-Order Condition (FOC)**:
$$\frac{\partial \Pi}{\partial p_1} = s_1 + \alpha s_1(1 - s_1) \left[ p_1^* - c + \delta \Delta \pi_2 \right] = 0$$
Dividing by $s_1$:
$$1 + \alpha(1 - s_1^*) \left[ p_1^* - c + \delta \Delta \pi_2 \right] = 0 \implies p_1^*(h_0) = c - \frac{1}{\alpha [1 - s_1^*(h_0)]} - \delta \Delta \pi_2$$

### Economic Intuition of the Opportunity Cost $\Delta \pi_2$:
- **When $\gamma < 0$ (Habit / Switching Cost)**:
  Consumers who buy inside today want to repeat tomorrow. Therefore, $\pi_2^*(j) > \pi_2^*(0) \implies \Delta \pi_2 > 0$.
  The term $-\delta \Delta \pi_2 < 0$ acts as an **introductory discount** ("bargain-then-rip-off") to lock the consumer in.
- **When $\gamma > 0$ (Variety-Seeking)**:
  Consumers who buy inside today will want to switch out tomorrow. Therefore, $\pi_2^*(j) < \pi_2^*(0) \implies \Delta \pi_2 < 0$.
  The term $-\delta \Delta \pi_2 > 0$ **increases** the first-period price: the firm harvests consumers today because they won't repeat tomorrow!

---

## 3. Comparative Statics: How Do Prices Change with $\gamma$?

By applying implicit differentiation to the first-order conditions:

### Period 2 Prices
$$\frac{dp_2^*(h_1)}{d\gamma} = -\frac{q(h_1) s_2^*(h_1)}{\alpha}$$
Because $\alpha < 0$:
- **After an outside purchase ($h_1 = 0, q = +1$)**: $\frac{dp_2^*(0)}{d\gamma} > 0$. Higher switching taste raises willingness to buy, so the firm raises price.
- **After an inside purchase ($h_1 = j, q = -1$)**: $\frac{dp_2^*(j)}{d\gamma} < 0$. Higher switching taste means the consumer wants to leave, so the firm must cut price to retain them.

### Period 1 Prices (with $\delta = 1$)
$$\frac{dp_1^*(h_0)}{d\gamma} = -\frac{q(h_0) s_1^*(h_0) + (1 - s_1^*(h_0)) [s_2^*(j) + s_2^*(0)]}{\alpha}$$
- **Initially Outside ($h_0 = 0, q = +1$)**: Both terms inside the numerator are strictly positive $\implies \frac{dp_1^*(0)}{d\gamma} > 0$ unambiguously.
- **Initially Inside ($h_0 = j, q = -1$)**: The first term is negative while the second is positive $\implies$ ambiguous net effect.

### Price Ratios $R = p_1^* / p_2^*$
For the path $(h_0 = 0, h_1 = j)$:
$p_1^*(0)$ increases with $\gamma$, while $p_2^*(j)$ decreases with $\gamma$.
Therefore, the price ratio $R(0, j) = p_1^*(0) / p_2^*(j)$ **unambiguously increases with $\gamma$**.

---

## 4. Uniform Pricing vs. Price Discrimination

In [theory_pricing_simulation.py](file:///Users/tate/LOV_CC_2YP/dumbed_down_code/theory_pricing_simulation.py), we simulate this model with:
$$\beta = 2.0, \quad \alpha = -1.0, \quad c = 0.0, \quad \delta = 1.0, \quad \gamma \sim \mathcal{N}(0, 1.25^2)$$
We partition the normal distribution into 5 equal-probability quintiles ($Q_1$ to $Q_5$).

### The Two Information Regimes:
1. **Uniform Pricing (Uninformed Baseline)**:
   - The firm observes history ($h_0$ at $t=1$, $h_1$ at $t=2$), but does not know which quintile the consumer belongs to.
   - It sets a single population-optimal price $p_1^*(h_0)$ and single terminal prices $p_2^*(j)$, $p_2^*(0)$.
2. **Price Discrimination (Informed Counterfactual)**:
   - The firm observes history AND the consumer's quintile $k \in \{1, 2, 3, 4, 5\}$.
   - It offers targeted price paths $(p_1^k, p_2^*(j, k), p_2^*(0, k))$ customized to each quintile.

### Key Numerical Results:
- **For initially outside consumers ($h_0 = 0$)**:
  - $Q_1$ (habitual outside) receives a massive introductory discount ($p_1 = \$0.41$ vs. Uniform $\$2.41$), leading to a **$+214\%$ increase in consumer surplus**!
  - Overall consumer surplus rises by **$+20.55\%$**, revenue rises by **$+15.80\%$**, and total welfare rises by **$+17.64\%$**.
- **For initially inside consumers ($h_0 = j$)**:
  - The firm extracts surplus from the tails ($Q_1$ and $Q_5$), so total welfare decreases for the tails but increases for middle quintiles ($Q_2, Q_3, Q_4$).
