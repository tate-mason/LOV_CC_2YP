# Toy model: no initial purchase state

Consumers begin with no prior purchase. There is no initial state distribution
or averaging over initial histories. Period-one utility is

$$V_{rj1}=\beta_j+\alpha p_{j1}.$$

First-period shares are identical across H and L. Variety utility begins in
period two, relative to the product purchased in period one:

$$V_{rj2}(h)=\beta_j+\gamma_r\mathbf{1}\{j\ne h\}+\alpha p_{j2}(h).$$

The firm knows the fixed type masses $\lambda=\Pr(H)$ and $1-\lambda=\Pr(L)$.
It commits to the full price menu in advance. There is no learning or posterior
updating. Consumers are myopic; choosing outside in period one exits permanently.
Prices may vary by product, period, and the period-one purchase. Uniform prices
are common across types. The observed-type benchmark gives each type its own menu.

For first-period shares $b_h(p_1)$ and conditional terminal profits $w_r(h)$,
expected discounted profit per initial consumer is

$$\Pi=\sum_h b_h(p_1)\left[p_{h1}-c_h+
\delta\{\lambda w_H(h)+(1-\lambda)w_L(h)\}\right].$$

Since first-period shares do not depend on type, terminal prices can be solved
using the known type masses directly. `solve_common` optimizes the full menu;
`solve_newborn` and `evaluate_newborn` are compatibility names for this same model.

The three fixed menus are optimized for H, L, and a 50/50 mix, respectively.
Their prices remain fixed as the type probability varies. Each menu is evaluated
on both true types, and the upper envelope gives the preferred fixed menu.
The probability in this comparison is a type share, not an initial purchase state.
See `updates/toy_model/results/fixed_menu_results.md` for regenerated cutoffs.

The fixed terminal-discount comparison uses the same consumers. It holds the
known-H first-period prices fixed and replaces terminal prices with the known-L
optimum. Its probability variable is Pr(L); see `newborn_results.md` in the same
results directory. These file names are retained for compatibility.

Run `.venv/bin/python toy_model.py` to regenerate tables, figures, and reports.
The derivation is in `updates/toy_model/toy_model.tex` and its compiled PDF.
Known-type terminal interior optima have equal product margins at each history;
this specification does not establish switching toward a higher-margin good.
