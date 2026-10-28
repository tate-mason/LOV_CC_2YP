# Model and derivative conventions

Utilities are U_t = beta + gamma z_t + alpha p_t, with an outside option of utility zero. Shares are logit probabilities; s_0t = 1-s_jt. Profit is Pi = s_j1 [(p1-c) + delta s_j2 (p2-c)]. Thus period-2 sales require a period-1 purchase. The indicators z1,z2 are fixed history labels, not endogenous probabilities. The plotted histories are (0,0), (1,0), (0,1).

Prices are independent decision variables. Period-2 utility has no p1 dependence. The period-1 FOC differentiates profit holding p2 fixed; when p2 is optimized first, its FOC supplies the envelope argument for the same result. Period-2's implemented FOC is normalized by the strictly positive delta*s_j1.

For comparative statics in alpha, beta, gamma, and delta, both prices are re-optimized, with c and the other parameters and history indicators fixed. Every share responds through its own utility and own price. In particular, ds_j2/dgamma = s_j2*s_02*(z2 + alpha*dp2/dgamma). The old unqualified share coefficient is now named dsj2_dutility; the total gamma derivative lives in comparative_statics/gamma_p2.py. No imports use the old misleading name.

The delta check using R2 holds R2=s_j2/s_02 fixed. This agrees with the full model here because period-2 pricing and utility do not depend on delta. It would not apply unchanged to a model with delta-dependent period-2 demand or cost.

Assumptions: alpha<0, 0<delta<1, c>=0, positive prices, interior shares. The assumptions dictionary is descriptive, not a SymPy assumption context. Sign checks use explicit positive symbols and alpha=-a. Numerical code rejects parameter violations and solutions outside the positive-price, interior-share region. The implicit pricing Jacobian is triangular with diagonal 1/s_01 and 1/s_02, so it is nonsingular for interior shares.

# Validation and corrections

Run `python3 run_all.py`. Checks cover both profit-derived FOCs, collected form, both price equations, implicit differentiation for all four parameters, the beta/gamma quotient rules, gamma history signs, the profit envelope, and numerical comparisons. Numerical validation uses 27 parameter/history/gamma combinations, checks FOC residuals and negative definite joint profit Hessians, and compares analytic formulas with central finite differences that re-solve both prices. It also checks the mixed beta-gamma price derivative. These are local optimality checks and sampled numerical evidence, not a claim about every possible calibration.

Validation corrected the alpha derivative, the mixed beta-gamma derivative, and the profit envelope. In particular, the full profit derivative is

    dPi*/dgamma = -s_j1 * (z1 + delta*z2*s_j2) / alpha.

The previous formula omitted s_j1 on the continuation-profit contribution. The stand-alone symbolic script now imports the modular formulas to avoid divergent copies.

# Reproducible figures

Run `python3 make_comparative_static_figures.py`. It solves both pricing equations at each of 201 gamma values in [0,3], for each of the three histories. Parameters are alpha=-2, beta=-1, delta=0.7, c=1. Outputs in `figures/` include PNG and PDF versions and `solutions.csv`. Prices and shares adjust at each point; the graphs are not integrations of fixed-share slopes. The ratio derivative is zero for (0,0), positive for (1,0), and negative for (0,1). History labels are shown without imposing an additional interpretation of z beyond the stated utilities.

For a paper, use the price-ratio curve as the main result and the derivative curve as its sign verification. Profit is a complementary result. Add separate p1 and p2 panels to show the mechanism, and s_j1 and s_j2 panels to explain demand adjustment. For robustness, vary beta and delta over a grid, re-solve the FOCs, and show heatmaps of derivative magnitudes; mask calibrations violating feasibility. State the calibration, history, fixed parameters, and whether a plot shows levels or total derivatives in every caption. Parameters here are illustrative, not estimated.
