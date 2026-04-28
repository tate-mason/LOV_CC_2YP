# Literature Review: Love of Variety, Consumer Choice, and Firm Advertising

**Tate Mason**  
*April 2026*

---

## 1. The Love of Variety Concept

The foundational treatment of love of variety in economics is **Dixit and Stiglitz (1977)**, "Monopolistic Competition and Optimum Product Diversity," *American Economic Review* 67(3): 297–308. Their model formalizes consumer preferences for variety using a CES utility function in a monopolistically competitive market. Crucially, the love of variety arises from the curvature of preferences: a consumer with convex preferences strictly prefers an average of any two bundles over either extreme. While Dixit-Stiglitz variety is defined over distinct goods simultaneously, their framework motivates the fundamental idea that the marginal utility of any single good is diminishing in the breadth of consumption — directly analogous to the $\gamma(\Sigma_j)^2$ sameness penalty in this model.

A more recent treatment is **Fally and Faber (2022)**, "Love of Variety and Gains from Trade," *European Economic Review* 158, which revisits the empirical content of love of variety preferences and discusses when such preferences generate welfare gains from access to new products. Their decomposition of variety gains is useful for thinking about how consumer welfare is affected as the firm's offered set $C_t$ expands or contracts.

---

## 2. Variety-Seeking Behavior in Consumer Choice

### 2.1 Conceptual Foundations

**McAlister and Pessemier (1982)**, "Variety Seeking Behavior: An Interdisciplinary Review," *Journal of Consumer Research* 9(3): 311–322, provide the foundational taxonomy of variety-seeking. They distinguish *derived* variety-seeking (seeking variety for instrumental reasons, e.g., stocking up) from *direct* variety-seeking, where the desire for novelty is intrinsic. Their framework is the closest conceptual antecedent to the $\gamma_i$ parameter in this model: a consumer with high $\gamma_i$ derives direct disutility from attribute sameness.

**Givon (1984)**, "Variety Seeking Through Brand Switching," *Marketing Science* 3(1): 1–22, develops a stochastic brand-choice model in which variety-seeking is identified by the propensity to switch brands even absent price changes or stock-outs. Givon estimates individual-level variety-seeking parameters from scanner data — a direct empirical precedent for the heterogeneous $\gamma_i$ in this model.

**McAlister (1982)**, "A Dynamic Attribute Satiation Model for Choices Across Time," *Journal of Consumer Research* 9(2): 141–150, models attribute-level satiation directly: each attribute accumulated through past consumption reduces the marginal utility of that attribute from future products. This is structurally very close to the $\Sigma_j$ accumulation in this model, where the sum of chosen attributes penalizes future utility.

### 2.2 Reviews and Synthesis

**Kahn (1995)**, "Consumer Variety-Seeking Among Goods and Services: An Integrative Review," *Journal of Retailing and Consumer Services* 2(3): 139–148, synthesizes the marketing literature on variety-seeking motivations into three categories: (1) satiation/stimulation — repeating consumption reduces marginal utility and causes boredom; (2) external situational factors — social contexts drive switching; and (3) future preference uncertainty — consumers hedge against uncertain future tastes. The satiation/stimulation channel directly motivates the $\gamma(\Sigma_j)^2$ specification.

**Sevilla, Zhang, and Kahn (2019)**, "Variety-Seeking, Satiation, and Maximizing Enjoyment Over Time," *Journal of Consumer Psychology* 29(2): 225–235, provide a modern psychological treatment. They show that variety-seeking and satiation interact in complex ways: consumers who seek variety tend to maximize hedonic experience across time in a manner consistent with a forward-looking but satiation-aware utility function. This is consistent with the long-run utility decline seen in this model's simulations.

**Inman (2001)**, "The Role of Sensory-Specific Satiety in Attribute-Level Variety Seeking," *Journal of Consumer Research* 28(1): 105–120, establishes that variety-seeking operates primarily at the attribute level rather than the brand level — supporting the use of scalar product attributes $X_{jt}$ rather than brand dummies as the fundamental object of variety in this model.

---

## 3. Discrete Choice Models

### 3.1 The Logit Foundation

**McFadden (1974)**, "Conditional Logit Analysis of Qualitative Choice Behavior," in P. Zarembka (ed.), *Frontiers in Econometrics*, Academic Press, pp. 105–142, derives the multinomial logit from random utility maximization with T1EV errors — the exact distributional assumption used in this model's $\epsilon_{ijt}$. The model implies the closed-form choice probability $P_{ijt} = e^{U_{ijt}} / \sum_k e^{U_{ikt}}$, which is equation (2) of this project. McFadden's framework is the foundational workhorse of all structural discrete-choice IO models.

### 3.2 Random Coefficients and Heterogeneity

**Berry, Levinsohn, and Pakes (1995)** (BLP), "Automobile Prices in Market Equilibrium," *Econometrica* 63(4): 841–890, extend McFadden's logit to allow random coefficients on product characteristics, accommodating the consumer heterogeneity in $\beta_i$ and $\gamma_i$ in this model. BLP's contraction mapping for inverting market shares to recover mean utilities is the standard methodology for estimating demand in differentiated-product markets. If this model were taken to data, BLP-style estimation with random coefficients on $\gamma_i$ would be the natural approach.

---

## 4. State Dependence, Habit Formation, and Dynamic Choice

### 4.1 State Dependence vs. Heterogeneity

**Heckman (1981)**, "Heterogeneity and State Dependence," in S. Rosen (ed.), *Studies in Labor Markets*, University of Chicago Press, pp. 91–139, establishes the fundamental identification problem in dynamic discrete choice: persistence in choices can arise from either *structural state dependence* (lagged choice directly enters current utility) or *spurious state dependence* (persistent unobserved heterogeneity). The $\Sigma_j$ term in this model is explicitly a structural state dependence mechanism — past consumption history directly depresses current utility through the sameness penalty.

### 4.2 Consumer Learning

**Erdem and Keane (1996)**, "Decision-Making Under Uncertainty: Capturing Dynamic Brand Choice Processes in Turbulent Consumer Goods Markets," *Marketing Science* 15(1): 1–20, develop a Bayesian learning model where consumers update beliefs about brand quality via usage experience and advertising exposure. Their forward-looking dynamic model is a direct structural precedent for the firm's belief-updating rule $\lambda_t(\gamma) \sim \mathcal{N}(1/\Sigma_{jt}, \sigma_\lambda)$ in this project, where the firm updates its belief about consumer type $\gamma_i$ using the accumulated purchase history. The key difference is that here the *firm* (not the consumer) is doing the learning.

---

## 5. Advertising in IO Models

### 5.1 Informative Advertising

**Nelson (1970)**, "Information and Consumer Behavior," *Journal of Political Economy* 78(2): 311–329, distinguishes search and experience goods and argues that advertising serves an informational role, signaling product quality. This is the classic starting point for the IO literature on advertising.

**Grossman and Shapiro (1984)**, "Informative Advertising with Differentiated Products," *Review of Economic Studies* 51(1): 63–81, develop a model of oligopolistic competition where advertising reaches consumers who would otherwise be unaware of the product. Their key result — that advertising is socially excessive in oligopoly — is directly relevant to the advertising incentive results in this model. Under both the fixed and Markov specifications, the firm's incentive to advertise ($V(\text{ad}) - V(\text{no ad})$) turns out to be negative in the fixed case and positive in the Markov case, which maps onto the debate over whether advertising is welfare-enhancing or excessive.

### 5.2 Advertising Dynamics

**Dubé, Hitsch, and Manchanda (2005)**, "An Empirical Model of Advertising Dynamics," *Quantitative Marketing and Economics* 3(2): 107–144, develop a dynamic discrete-choice model of advertising competition in which advertising generates goodwill that depreciates over time. Their estimated weekly carryover parameter of 0.89 and 12% weekly decay of goodwill establish empirical baselines for the persistence of advertising effects. The Markov advertising rule in this project — where advertising is a function of the firm's current belief about consumer differentiation — is analogous to their dynamic advertising scheduling problem.

### 5.3 Targeted Advertising

**Bergemann and Bonatti (2011)**, "Targeting in Advertising Markets: Implications for Offline Versus Online Media," *RAND Journal of Economics* 42(3): 417–443, develop a model where advertisers send messages to consumer segments with different match values. Their key finding — that the equilibrium price of advertising is non-monotonic in targeting precision — provides a benchmark for thinking about the firm's Markov advertising rule in this model, where higher precision in inferring $\gamma_i$ affects the optimal advertising level.

**Shin and Yu (2021)**, "Targeted Advertising and Consumer Inference," *Marketing Science* 40(6): 1030–1051, develop a model where consumers make inferences about product quality *from the fact of being targeted*. This creates an advertising spillover: targeting one consumer makes others more likely to engage. The belief-updating mechanism in this project — where the firm updates its estimate of $\gamma_i$ from purchase history — is a supply-side analog of Shin and Yu's consumer inference.

---

## 6. Recommendation Algorithms, Platforms, and Filter Bubbles

### 6.1 Platform Economics and Recommender Systems

**Hosanagar, Fleder, Lee, and Buja (2014)**, "Will the Global Village Fracture Into Tribes? Recommender Systems and Their Effects on Consumer Fragmentation," *Management Science* 60(4): 805–823, show that recommender systems simultaneously increase sales volume and reduce individual-level consumption diversity (homogeneity bias) while increasing aggregate variety. This is a direct empirical counterpart to the model's firm side: the firm's choice of $\sigma_x$ (the spread of products offered) maps onto the recommender system's effective targeting precision, with similar tradeoffs between individual and aggregate variety.

**Aguiar, Waldfogel, and Waldfogel (2021)**, "Playlists and Platforms: Evidence from Spotify," *Journal of Industrial Economics* 69(4), provide causal evidence on how algorithmic curation on music platforms affects listening diversity. Their finding that platform recommendations narrow individual variety while increasing aggregate discovery is consistent with the dynamics in this model under low-$\sigma_x$ regimes.

**Papanastasiou, Bimpikis, and Savva (2018)**, "Crowdsourcing Exploration," *Management Science* 64(4): 1727–1746, study the platform's tradeoff between exploiting consumer preferences and exploring new products, which directly motivates the $\sigma_x$ parameter in this model as a choice variable: a higher $\sigma_x$ corresponds to more exploration.

### 6.2 Filter Bubbles

**Flaxman, Goel, and Rao (2016)**, "Filter Bubbles, Echo Chambers, and Online News Consumption," *Public Opinion Quarterly* 80(S1): 298–320, provide empirical evidence that algorithmic filtering leads to reduced exposure to cross-cutting content. In the context of this model, filter bubble formation corresponds to $\sigma_x \to 0$: the firm converges on a narrow set of products centered around $\bar{X}_{jt}$, reducing consumer utility through the sameness penalty.

**Dubé, Fang, Fong, and Luo (2022)**, "Competitive Price Targeting with Smartphone Coupons," *Marketing Science* 41(6), demonstrate that firms can use individual-level behavioral data to fine-tune targeting, which connects to the long-run objective here of making $\sigma_x$ and $a_{jt}$ endogenous to the firm's inferred $\gamma_i$.

---

## Summary Table

| Paper | Relevance |
|---|---|
| Dixit & Stiglitz (1977) | Foundational love of variety; CES utility |
| McAlister & Pessemier (1982) | Taxonomy of variety-seeking; direct vs. derived |
| McAlister (1982) | Attribute-level satiation; $\Sigma_j$ accumulation |
| Givon (1984) | Stochastic model of variety-seeking; individual $\gamma$ |
| McFadden (1974) | Logit foundation; T1EV errors; choice probabilities |
| Heckman (1981) | State dependence vs. heterogeneity; $\Sigma_j$ as structural state |
| Berry, Levinsohn & Pakes (1995) | Random coefficients logit; heterogeneous $\beta_i, \gamma_i$ |
| Erdem & Keane (1996) | Dynamic brand choice; Bayesian learning analog |
| Kahn (1995) | Satiation/stimulation; variety-seeking review |
| Nelson (1970) | Informative advertising; experience goods |
| Grossman & Shapiro (1984) | Informative advertising; social excess |
| Dubé, Hitsch & Manchanda (2005) | Advertising dynamics; goodwill depreciation |
| Bergemann & Bonatti (2011) | Targeting precision; advertising markets |
| Shin & Yu (2021) | Targeted advertising and consumer inference |
| Hosanagar et al. (2014) | Recommender systems; homogeneity bias |
| Fally & Faber (2022) | Love of variety; welfare measurement |
