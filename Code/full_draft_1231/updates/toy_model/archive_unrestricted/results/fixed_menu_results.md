# Three fixed pricing menus

H and L menus maximize profit for known H and known L respectively. The uniform menu maximizes profit for a 50/50 mix. Each menu allows product, period, and terminal purchase-history variation. All three menus remain fixed as q = Pr(H) varies. Consumers have no prior purchase; variety utility begins in period two.

Expected profit for menu m is q*profit_m(H) + (1-q)*profit_m(L). For two menus a and b, let D_H=profit_a(H)-profit_b(H), and similarly D_L. Their crossing is q=-D_L/(D_H-D_L). Only crossings on the upper envelope change the optimal menu.

- 0.00000000 < Pr(H) < 0.20708781: low menu.
- 0.20708781 < Pr(H) < 0.75402175: uniform_50_50 menu.
- 0.75402175 < Pr(H) < 1.00000000: high menu.

Adjacent menus tie at each internal boundary; endpoint choices follow the adjacent region. There is no targeting cost. With type observed, choose H at q=1 and L at q=0. With unrestricted reoptimization at every q, the optimal common menu weakly dominates these three candidates; these thresholds describe a restricted choice among fixed menus.

Both known-type terminal optima have equal product margins at each history: p_rk2(h)-c_k = -1/(alpha*s_r02(h)) for interior prices. Thus a higher-margin destination good is not implied by this model. The H menu has higher terminal prices than the L menu in this calibration; both are monopoly optima for their respective types. A switch from the uniform menu can raise some prices and lower others.

See prices_50_50.csv for the full menus and fixed_menu_type_profits.csv for the cross-type profit evaluations.
