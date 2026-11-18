# Random-coefficients choice model: start here

No prior Python or choice-model experience is assumed. The example is deliberately small, but it actually generates data and estimates a model.

## 1. What are we trying to explain?

Imagine 2000 people. Each chooses one of three products. We observe the price and quality of every product they could choose, and which product they chose.

We want to estimate:

1. How much people care about price **on average**.
2. How much price sensitivity **differs between people**.
3. How much people care about quality.

The model is called **mixed logit**, also called **random-coefficients logit**. “Random coefficient” means a preference varies across people according to a distribution. It does not mean Python randomly changes the estimated answer for no reason.

Our one random coefficient is the price effect. The quality coefficient is shared by everyone. You can have several random coefficients, but one makes a good starting point.

Files:

- `random_coefficients_logit.py`: executable example.
- `requirements.txt`: the two Python packages it needs.
- `GUIDE.md`: this explanation.

## 2. Run it

Open a terminal in this folder. Assuming Python 3 is installed, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 random_coefficients_logit.py
```

The first command creates a local environment for Python packages. The second activates it on macOS/Linux. The third installs NumPy and SciPy. The fourth runs the example. On Windows, use `.venv\Scripts\activate` to activate the environment and use `python` if your installation does not have a `python3` command.

The output reports whether the optimizer converged, the true values used to make the fake data, and the estimates. The estimates will not match the true values exactly: there are only finitely many people and we approximate an integral using simulation.

## 3. What does the data look like?

We use three arrays, which are containers for numbers:

| Python name | Shape | Meaning |
|---|---|---|
| `price` | `(2000, 3)` | One row per person, one column per option: its price |
| `quality` | `(2000, 3)` | Same layout: each option's quality |
| `choices` | `(2000,)` | One integer per person: the column they chose |

For example, one person's row could be:

```text
price:    [2.0, 4.0, 3.0]
quality:  [1.0, 3.0, 2.0]
choice:   1
```

Python counts from zero. A choice of `1` means the **second** option, with price 4 and quality 3. A choice of `0` means the first option.

Prices and qualities vary across people in the fake data. That gives the model many different choice situations to learn from. Every person must choose one of the available options; we do not include a “buy nothing” option.

## 4. The math, one piece at a time

### 4.1 Utility is a score

Let $i$ identify a person and $j$ identify an option. A higher utility means an option is more attractive:

$$
U_{ij}=\beta_i p_{ij}+\gamma q_{ij}+\varepsilon_{ij}.
$$

| Symbol | Meaning |
|---|---|
| $U_{ij}$ | Person $i$'s total utility for option $j$ |
| $p_{ij}$ | Its price |
| $q_{ij}$ | Its quality |
| $\beta_i$ | This person's price coefficient |
| $\gamma$ | Everyone's shared quality coefficient |
| $\varepsilon_{ij}$ | Unobserved factors affecting this choice |

The person chooses the option with the highest total utility. We cannot observe the last term, so we predict probabilities instead of perfectly predicting choices.

The part before the error is called **systematic utility**:

$$
V_{ij}=\beta_i p_{ij}+\gamma q_{ij}.
$$

For example, with $\beta_i=-1$ and $\gamma=0.8$, a product with price 2 and quality 3 gets score $-1\times2+0.8\times3=0.4$.

These scores are relative attractiveness, not probabilities or dollars. Only utility differences between options affect choices. We omit product-specific constants to keep the example small. A constant added equally to every option would cancel out of the probabilities anyway.

### 4.2 People have different price coefficients

We assume:

$$
\beta_i=\mu+\sigma z_i,\qquad z_i\sim N(0,1).
$$

Equivalently, $\beta_i\sim N(\mu,\sigma^2)$.

- $N(0,1)$ is the standard normal bell curve: mean zero, standard deviation one.
- $\mu$ is the mean price coefficient.
- $\sigma$ is its standard deviation, measuring variation between people.
- $\sigma^2$ is its variance. The code estimates the **standard deviation**, not the variance.

With $\mu=-1$ and $\sigma=0.5$, a person with $z_i=-1$ has coefficient $-1.5$, and someone with $z_i=1$ has coefficient $-0.5$. Both dislike higher prices, but the first person dislikes them more.

A normal distribution also permits positive price coefficients. This is an intentional simplification, not a guarantee that every person dislikes price. A model requiring all price effects to be negative could instead use $\beta_i=-\exp(\mu+\sigma z_i)$, but then $\mu$ and $\sigma$ describe the underlying normal variable, not the mean and standard deviation of the price coefficient itself. We do not implement that alternative here.

If $\sigma=0$, all people have the same price coefficient and the model reduces to ordinary multinomial logit.

### 4.3 If we knew someone's coefficient, we could calculate logit probabilities

Assume the errors $\varepsilon_{ij}$ are independent standard type-I extreme-value errors, also called Gumbel errors, and independent of observed attributes and the random coefficient. Their scale is fixed at one: we cannot separately identify the utility coefficients and an arbitrary overall error scale from choices alone.

Under this assumption, conditional on $\beta_i$:

$$
P_{ij}(\beta_i)=\frac{\exp(V_{ij})}{\sum_{k=1}^{J}\exp(V_{ik})}.
$$

Here $J=3$, $k$ runs through all the options, and $\exp(x)=e^x$ turns a score into a positive number. Divide each positive number by their total and you get probabilities that add to one.

For utilities `[0, 1, 2]`, the probabilities are approximately `[0.090, 0.245, 0.665]`. The highest-scoring option is most likely, but not certain, because of the unobserved errors.

### 4.4 We do not know each person's coefficient

We average conditional probabilities over the assumed distribution of price preferences:

$$
P_{ij}(\theta)=\int_{-\infty}^{\infty}
\frac{\exp((\mu+\sigma z)p_{ij}+\gamma q_{ij})}
{\sum_{k=1}^{J}\exp((\mu+\sigma z)p_{ik}+\gamma q_{ik})}
\phi(z)\,dz.
$$

Here $\phi(z)$ is the standard normal density, which gives different preference values their relative weight, and $\theta$ represents the parameters we want to estimate.

An integral is a continuous weighted average. This integral generally does not have a simple closed-form answer, so we approximate it by drawing many possible preferences:

$$
z_{ir}\sim N(0,1),\qquad \beta_{ir}=\mu+\sigma z_{ir},
$$

$$
\widehat P_{ij}(\theta)=\frac{1}{R}\sum_{r=1}^{R}P_{ij}(\beta_{ir}).
$$

The code uses $R=200$ draws per person. The index $r$ labels a simulation draw. These draws represent possible preferences we average over; they are not 200 extra people or 200 observed choices.

We average **probabilities**, not coefficients. Calculating a logit probability at the average coefficient generally gives a different answer.

### 4.5 Fit the model to the observed choices

Let $y_i$ be the option person $i$ actually chose. Assuming independent people, the simulated likelihood is:

$$
\widehat L(\theta)=\prod_{i=1}^{N}\widehat P_{i,y_i}(\theta).
$$

The product symbol means multiply the chosen-option probabilities across people. We want parameter values that make the observed choices likely.

Multiplying hundreds of small numbers is numerically awkward. Taking natural logarithms turns the product into a sum:

$$
\widehat\ell(\theta)=\sum_{i=1}^{N}\log\widehat P_{i,y_i}(\theta).
$$

SciPy's `minimize` finds a minimum, so we give it the negative:

$$
\operatorname{NLL}(\theta)=-\sum_{i=1}^{N}\log\widehat P_{i,y_i}(\theta).
$$

Lower NLL means a better fit to these data under the same simulation setup. This procedure is **maximum simulated likelihood**.

The order matters: first average probabilities over draws, **then** take the log. Averaging log probabilities would estimate a different objective.

### 4.6 Keep the standard deviation positive

Instead of directly optimizing $\sigma$, we optimize $a=\log\sigma$ and set $\sigma=\exp(a)$. The optimizer's parameter vector is therefore:

$$
\theta=(\mu,a,\gamma).
$$

Any finite $a$ gives a positive $\sigma$. The example additionally bounds $a$ between -5 and 2, putting $\sigma$ between about 0.0067 and 7.39. These are practical guardrails for this dataset, not universal economic restrictions. They exclude exactly zero. An estimate at a bound needs investigation; this skeleton does not provide a formal test of whether heterogeneity exists.

## 5. Every part of the Python code

### Imports and basic syntax

The opening triple-quoted string is a **docstring**: explanatory text, not a calculation. Lines starting with `#` are comments.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
```

- NumPy works with arrays and mathematical operations. `as np` gives it a short name.
- `minimize` searches for parameters that make a function small.
- `logsumexp` calculates logarithms of sums of exponentials safely.

`def` defines a function, which is a reusable recipe. Indentation identifies the instructions inside it. Arguments in parentheses are inputs. `return` sends the result back. `=` assigns a value to a name. A dot accesses a function or attribute belonging to an object, such as `price.shape`.

### `log_choice_probabilities`: convert scores into log probabilities

```python
utility = beta_price * price + beta_quality * quality
return utility - logsumexp(utility, axis=-1, keepdims=True)
```

The first line implements $V=\beta p+\gamma q$. NumPy's `*` multiplies matching entries here; it is not matrix multiplication.

The second implements:

$$
\log P_{ij}=V_{ij}-\log\sum_k\exp(V_{ik}).
$$

`axis=-1` means “combine the last dimension,” which always contains the alternatives. `keepdims=True` preserves that dimension with length one so subtraction lines up correctly.

Why logs? Directly computing exponentials of very large scores can overflow. `logsumexp` uses an equivalent stable calculation. For numbers $v_k$, let $m=\max_k v_k$; then:

$$
\log\sum_k e^{v_k}=m+\log\sum_k e^{v_k-m}.
$$

Subtracting the maximum prevents huge exponentials. This is the same mathematical probability formula, calculated more safely.

### `simulate_data`: make a practice dataset

The definition has default inputs `n_people=2000`, `n_alternatives=3`, and `seed=42`. Calling `simulate_data()` uses those defaults.

`rng = np.random.default_rng(seed)` creates a random-number generator. A seed makes the example reproducible in the same software environment.

`rng.uniform(1.0, 5.0, size=(n_people, n_alternatives))` fills a table with prices uniformly sampled between 1 and 5. The quality line does the same between 0 and 3. These are arbitrary teaching units.

The three `true_...` assignments choose the parameters used to generate data: $\mu=-1$, $\sigma=0.5$, $\gamma=0.8$. We know them because we invented the dataset. With real data, we would not know them.

`rng.normal(..., size=(n_people, 1))` draws one actual price coefficient per simulated person. Its single column lets NumPy use the same coefficient across all that person's alternatives. A person does not receive a separate price preference for each product.

`log_choice_probabilities(...)` computes conditional log probabilities using those preferences. `np.exp(log_p)` converts them back to ordinary probabilities.

```python
choices = np.array([
    rng.choice(n_alternatives, p=probabilities[i])
    for i in range(n_people)
])
```

This is a **list comprehension**, a compact loop. `range(n_people)` gives indices from 0 through 1999. For each person, `rng.choice` selects an option using their probabilities, and `np.array` turns the resulting list into an array. This samples the observed choice, not simply the option with the highest systematic score.

We do not explicitly draw Gumbel errors: sampling from the logit probabilities produces the same conditional choice distribution under our assumptions.

`return price, quality, choices` returns three arrays. It deliberately does not give the estimator the true person-specific coefficients.

### `negative_log_likelihood`: score a candidate parameter vector

```python
mean_price, log_sd_price, beta_quality = theta
sd_price = np.exp(log_sd_price)
n_people, n_draws = draws.shape
```

The first line **unpacks** the three numbers in `theta` into readable names. The second transforms $a$ into positive $\sigma$. The third reads the two dimensions of the draws table.

`beta_price = mean_price + sd_price * draws` implements $\beta_{ir}=\mu+\sigma z_{ir}$. Its shape is `(2000, 200)`.

Next we add dimensions so NumPy can calculate all people, all draws, and all options together:

| Expression | Shape | Role |
|---|---|---|
| `price[:, None, :]` | `(2000, 1, 3)` | Prices repeated conceptually across draws |
| `quality[:, None, :]` | `(2000, 1, 3)` | Qualities repeated conceptually across draws |
| `beta_price[:, :, None]` | `(2000, 200, 1)` | Each preference used across alternatives |
| returned `log_p` | `(2000, 200, 3)` | Log probability for every person/draw/option |

Inside brackets, `:` means “take everything on this dimension.” `None` inserts a dimension of length one. NumPy automatically expands compatible dimensions during calculations; this is called **broadcasting**. It avoids writing three nested Python loops.

```python
chosen_log_p = log_p[np.arange(n_people), :, choices]
```

`np.arange(n_people)` makes the array of person indices. NumPy pairs each person index with that person's chosen-option index, while `:` keeps all simulation draws. The result has shape `(2000, 200)`. Entry `[i, r]` is the log probability assigned to person `i`'s observed choice under draw `r`.

```python
log_average_p = logsumexp(chosen_log_p, axis=1) - np.log(n_draws)
return -np.sum(log_average_p)
```

Here `axis=1` is the simulation-draw dimension. If $b_{ir}$ is a chosen-option log probability, these lines calculate:

$$
\log\left(\frac{1}{R}\sum_r e^{b_{ir}}\right)
=\operatorname{logsumexp}_r(b_{ir})-\log R.
$$

There is one log average probability per person. `np.sum` adds them; the minus sign produces the NLL, a single number the optimizer can compare across parameter guesses.

### `main`: connect the pieces and estimate

`price, quality, choices = simulate_data()` creates and unpacks the fake dataset. `price.shape[0]` reads its number of rows. `n_draws = 200` sets the integration accuracy/cost tradeoff.

A second generator, with seed 123, creates standard normal estimation draws. These are separate from the true preferences used to generate choices. The estimator does not get to peek at the true preferences.

**The estimation draws stay fixed throughout optimization.** If we redrew them every time we scored parameters, changes in the objective would reflect both parameter changes and new simulation noise, making the search harder.

```python
initial_guess = np.array([-0.5, np.log(0.3), 0.5])
```

This starts the search at mean price effect -0.5, price-effect standard deviation 0.3, and quality effect 0.5. The middle input is the logarithm of 0.3 because that is the parameterization we optimize.

The `minimize` arguments mean:

| Argument | What it does |
|---|---|
| `negative_log_likelihood` | Function to make as small as possible; no parentheses because we pass the function itself |
| `x0=initial_guess` | Starting parameter vector |
| `args=(price, quality, choices, draws)` | Extra fixed inputs passed after `theta` |
| `method="L-BFGS-B"` | A numerical optimization method that supports bounds |
| `bounds=[...]` | Bounds in the same order as the three parameters |
| `options={"maxiter": 200}` | A dictionary setting the maximum number of optimizer iterations |

`(None, None)` means no lower or upper bound. Only the log standard deviation has finite bounds. We supply no derivative function, so SciPy approximates derivatives numerically. The optimizer repeatedly evaluates the objective and adjusts the three parameters.

`result.success` reports whether the optimizer met its stopping criteria. `result.message` explains its stopping reason. `if not result.success:` prints an extra caution if it failed. Convergence is not proof that the solution is the global best or that the economic assumptions are correct.

`result.x` contains the estimated parameters, and `result.fun` is the final NLL. We exponentiate the estimated log standard deviation before reporting it.

`print` displays text. `\n` inserts a new line. Strings beginning with `f` insert values inside braces. A format like `{mean_price:8.3f}` uses a field eight characters wide and three decimal places. The true-value column is hard-coded to match `simulate_data`; update it if you change the generating parameters.

```python
if __name__ == "__main__":
    main()
```

Python sets `__name__` to `"__main__"` when this file is run directly. This condition runs the example then, while allowing another script to import the functions without automatically running the estimation. `==` tests equality; it does not assign a value.

## 6. Reading the estimates

- A negative mean price effect means higher price reduces systematic utility on average, holding quality fixed.
- A positive price-effect standard deviation means the model allows price preferences to differ across people.
- A positive quality coefficient means higher quality increases systematic utility, holding price fixed.

Coefficients are not percentage changes in purchase probability. Probability changes depend on all options' attributes and the preference distribution.

We estimate three population parameters. We do **not** estimate a separate reliable coefficient for each of the 2000 people from their single choice.

## 7. Replace the fake data with your data

Inside `main`, replace `price, quality, choices = simulate_data()` with arrays in the same format. For example, this shows the required layout:

```python
price = np.array([
    [2.0, 4.0, 3.0],
    [3.0, 2.0, 5.0],
], dtype=float)
quality = np.array([
    [1.0, 3.0, 2.0],
    [2.0, 1.0, 3.0],
], dtype=float)
choices = np.array([1, 0], dtype=int)
```

`dtype=float` stores numerical measurements with decimals; `dtype=int` stores integer choice indices. These two rows illustrate formatting only: two observations are not adequate for this estimation.

You need attributes for **all available options**, not just the selected one. Keep columns aligned across prices, qualities, and choice indices. Use finite numbers with no missing values and integer choices between zero and the number of columns minus one. Every row must have the same number of available options for this skeleton.

With real data, remove the true-value column from the printing code. It refers only to our simulation.

## 8. What this skeleton assumes and leaves for later

This is a learning example, not a complete empirical analysis:

- **One choice per independent person.** Repeated choices by the same person need a panel likelihood if their coefficient persists. For that model, multiply the conditional probabilities across a person's occasions *within each draw*, then average over draws. Treating their rows as independent would describe a different model.
- **All options are available.** Different choice sets need availability handling in the probability denominator.
- **Attributes are exogenous.** We assume price and quality are independent of omitted utility shocks and of the unmodeled variation in preferences. Real prices may violate this. The skeleton does not establish a causal price effect.
- **Simple preference distribution.** Price sensitivity is normal, quality sensitivity is fixed, and there are no product-specific constants or demographic interactions.
- **Approximate integration.** More draws generally improve the approximation and increase runtime. For serious work, compare results using larger draw counts and different fixed simulation seeds. Finite-draw log likelihoods have simulation error; taking a log of an averaged probability also introduces simulation bias.
- **One starting point.** Try multiple starting vectors before trusting an empirical fit. A convergence flag alone does not establish a global optimum.
- **No standard errors.** This example reports point estimates only. It does not calculate confidence intervals or statistical significance.

A useful first experiment is to increase `n_people` to 4000 and `n_draws` to 500. More people provide more information; more draws improve the integral approximation. Those are different sources of improvement, and neither guarantees that every estimate moves closer to its true value in any one run.
