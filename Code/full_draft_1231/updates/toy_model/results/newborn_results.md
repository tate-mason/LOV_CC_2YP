# Newborn discount decision

The newborn has no previous product. Their type is unobserved and Pr(L)=q, Pr(H)=1-q. The first purchase has no switching bonus and therefore does not reveal type. Type remains unobserved in period two.

The regular menu is the two-period policy optimized for a known H type. The discount menu replaces terminal prices with the known-L optimum while keeping the regular first-period price. All its prices are verified to be weakly below regular prices. Both menus are fixed while q varies. Period-two offers depend on the first product purchased. The firm knows both types exogenously for purposes of constructing these benchmark menus.

Profit change from discounting an H newborn: -0.04484198

Profit change from discounting an L newborn: 0.02410654

Break-even Pr(L): 0.65036900

Break-even Pr(H): 0.34963100

50/50 regular profit: 0.67420560

50/50 discount profit: 0.66383788

50/50 reoptimized common profit: 0.68367095

Expected gain = (1-q)*Delta_H + q*Delta_L. The exact cutoff is -Delta_H/(Delta_L-Delta_H). Offer the fixed discount when the expected gain is positive; at zero the firm is indifferent. No fixed adoption cost or classifier error is added.

The CSV sweeps q from 0.01 through 0.99 and also solves freely reoptimized common prices at every q. The fixed-menu cutoff does not imply a discontinuity in those freely optimized prices. It depends on the specified regular and discount menus.

If the probability being varied instead means the random offer probability a, expected profit is regular_profit + a*(discount_profit-regular_profit). Then the optimum is a=0 or a=1 (or any a at indifference), rather than an interior threshold in a.
