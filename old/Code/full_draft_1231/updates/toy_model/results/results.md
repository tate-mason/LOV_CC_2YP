# Free prices with random initial histories

Initial state is j, k, or 0, each with probability 1/3 independently of type. Initial 0 means no prior product and no switching bonus. Initial state is observed. Period-one outside choices exit permanently.
Prices for both goods in both periods are freely optimized, with no coupon restriction, price floor, or price ceiling. Period-two menus can depend on both initial state and period-one purchase. Consumers are myopic. All menus are committed before choices; endogenous selection of types into purchase histories enters the objective. First purchases can now be informative about type.

Parameters: beta=(2.2,2), costs=(1.2,0.8), alpha=-1.5, delta=.95. Good 2 costs 0.8 versus 1.2 for good 1; both prices remain freely optimized.
50/50 common profit: 1.08744099; observed-type profit: 1.09721954; information gain: 0.00977855.

The gamma_H sweep uses gamma_H=1.5 to 10 in steps of 0.5, holds gamma_L=1 and Pr(H)=0.5 fixed and reoptimizes all common prices for every gamma_H. It reports each true type's period-one profit, undiscounted period-two profit per arriving consumer, total discounted value, survival probability, and period-two profit per survivor. The known-type comparison separately optimizes each type. Induction sacrifice compares known-type dynamic period-one profit with its optimized one-period benchmark; it is not a causal effect of a coupon.

The value distribution has two atoms with masses 0.5, rather than a continuous density. value_distribution.csv records these atoms. A continuous value density would require a continuous distribution over gamma. The distribution is of expected customer values, not realized transaction profits.

Known-type prices use backward interior equations. Heterogeneous common-price optimization uses analytic gradients and three unconstrained BFGS starts per initial state, accepting stationary solutions with maximum gradient below 2e-6. Multistart checks provide numerical evidence, not a global theorem. With known type, interior terminal margins are equal across products at each history; a higher-margin destination is no longer maintained.
