# Correction: discounting toward the higher-margin good

The intended mechanism is a targeted discount that moves demand toward good 2,
which has a higher regular-price unit margin. The unrestricted-price calculation
in `toy_model.py` does not establish this mechanism: its known-type interior
terminal optimum has equal margins across goods. Lowering good 2's cost alone
cannot fix that issue when both prices remain freely adjustable.

The manuscript's regular-price/discount distinction must be made an actual
restriction on the firm's choices. Holding regular prices fixed, consider a
period-two discount on good 2 after purchasing good 1:

\[
p_1=\bar p_1,\qquad p_2(d)=\bar p_2-d,\qquad d\geq0.
\]

The unit margins are

\[
m_1=\bar p_1-c_1,\qquad m_2(d)=\bar p_2-c_2-d.
\]

A higher-margin switching destination requires

\[
\bar p_2-c_2>\bar p_1-c_1.
\]

This holds automatically with equal regular prices and lower cost for good 2.
Discounting reduces that margin advantage; it must be evaluated at the net
transaction prices, rather than inferred from costs alone.

For true type r, conditional terminal profit is

\[
w_r(d\mid h=1)=s_{r1}(d)m_1+s_{r2}(d)m_2(d).
\]

Let a = -alpha > 0. The logit derivative is

\[
\frac{dw_r}{dd}
=s_{r2}\left[-1+a\left\{s_{r0}m_2+s_{r1}(m_2-m_1)\right\}\right].
\]

The -1 term is the discount paid on purchases of good 2. The outside-option
term captures additional purchases; the margin-difference term captures demand
redirected from good 1. Thus the calculation explicitly values steering toward
the higher-margin product. At an interior optimum the bracket is zero; the
boundary d=0 must also be checked.

With an assumed H probability q, choose the discount to maximize

\[
q w_H(d)+(1-q)w_L(d).
\]

To measure mispricing, hold the actual type fixed and evaluate w_r(d*(q)). To
compare observed and unobserved types, use the same fixed regular prices and
feasible discounts for both regimes. Regular prices must not be reoptimized
separately for every assumed probability, which would confound the comparison.

Two qualifications matter for implementation:

- Finite switching bonuses and logit shocks do not guarantee that H switches.
  A smaller optimal H discount than L discount must be checked numerically;
  no discount for H is not an automatic theorem.
- After a purchase of good 2, switching goes toward good 1. A discount on good 2
  then encourages retention, not switching. History-conditional results should
  make this distinction explicit.

The selected specification keeps common price sensitivity and holds regular prices fixed only during the targeted-offer stage. The monopolist may set or revise regular prices beforehand. The active exercise is conditional on posted regular prices (2,2), with costs (1.2,0.2), and is not a claim about the jointly optimal upstream regular prices. Current results and slides now use this offer-stage specification. The old unrestricted model and outputs are preserved in `archive_unrestricted/`.
