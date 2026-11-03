"""Two-period symmetric switching model: solve, differentiate, verify, plot.

Run: python structural_switching_reference.py
Requires: numpy, scipy, matplotlib.  Outputs to ./outputs.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

BETA, ALPHA, DELTA, COST = 2.0, -1.0, 0.92, 0.0
GAMMAS = np.linspace(-3.0, 3.0, 601)
OUTPUT = Path('outputs')


def expit_safe(x):
    from scipy.special import expit
    return expit(x)


def inside_share(price, gamma, state, beta=BETA, alpha=ALPHA):
    """state='j': existing customer, state='0': outside previously."""
    q = -1 if state == 'j' else +1 if state == '0' else None
    if q is None:
        raise ValueError("state must be 'j' or '0'")
    return expit_safe(beta + q * gamma + alpha * price)


def p2_opt(gamma, state):
    """Unique finite price solving p=c-1/[alpha*(1-s)] (interior)."""
    f = lambda p: p - COST + 1 / (ALPHA * (1 - inside_share(p, gamma, state)))
    return brentq(f, -30.0, 40.0, xtol=1e-13)


def evaluate(gamma, initial_state):
    p2j, p20 = p2_opt(gamma, 'j'), p2_opt(gamma, '0')
    u, v = inside_share(p2j, gamma, 'j'), inside_share(p20, gamma, '0')
    Vj, V0 = (p2j-COST)*u, (p20-COST)*v
    dV = Vj - V0
    q = -1 if initial_state == 'j' else 1
    f = lambda p: p-COST+1/(ALPHA*(1-inside_share(p, gamma, initial_state)))+DELTA*dV
    p1 = brentq(f, -30.0, 40.0, xtol=1e-13)
    s = inside_share(p1, gamma, initial_state)
    r = 1-s
    pi = s*(p1-COST)+DELTA*(s*Vj+r*V0)
    static_piece = COST - 1/(ALPHA*r)
    dynamic_piece = -DELTA*dV
    # Envelope derivatives: Vj'=u/alpha, V0'=-v/alpha.
    d_pi = (-q*s + DELTA*(s*u-r*v))/ALPHA
    dp1 = -(q*s + DELTA*r*(u+v))/ALPHA
    # ds/dgamma = s*r*(q + alpha*dp1) = s*r^2*(q - delta*(u+v)).
    ds = s*r*r*(q-DELTA*(u+v))
    dB = -u*(1-u)**2 + v*(1-v)**2
    d2p1 = -(s*r*r*(q-DELTA*(u+v))**2 + DELTA*r*dB)/ALPHA
    dp2j = -(-1)*u/ALPHA
    dp20 = -(+1)*v/ALPHA
    matched_p2 = p2j if initial_state == 'j' else p20
    matched_dp2 = dp2j if initial_state == 'j' else dp20
    d_matched_ratio = (dp1*matched_p2-p1*matched_dp2)/matched_p2**2
    d_induce_ratio = (dp1*p2j - p1*dp2j)/p2j**2 if initial_state == '0' else np.nan
    assert np.isclose(p1, static_piece+dynamic_piece, atol=1e-9)
    return dict(gamma=gamma, initial_state=initial_state, p1=p1, p2j=p2j,
                p20=p20, profit=pi, V2j=Vj, V20=V0, DeltaV=dV,
                static_component=static_piece, continuation_adjustment=dynamic_piece,
                dp1=dp1, d2p1=d2p1, dprofit=d_pi,
                d_matched_ratio=d_matched_ratio, d_induce_ratio=d_induce_ratio)


def verify():
    h = 1e-4
    for g in [-2.4, -1.2, 0, 0.7, 2.6]:
        for state in ['j', '0']:
            result = evaluate(g, state)
            plus, minus = evaluate(g+h, state), evaluate(g-h, state)
            for value, key in [('dp1', 'p1'), ('dprofit', 'profit')]:
                numerical = (plus[key]-minus[key])/(2*h)
                assert np.isclose(result[value], numerical, atol=2e-7), (g,state,value,result[value],numerical)
            numerical2 = (plus['dp1']-minus['dp1'])/(2*h)
            assert np.isclose(result['d2p1'], numerical2, atol=2e-7), (g,state,'d2p1')
            numerical_ratio = (plus['p1']/ (plus['p2j'] if state=='j' else plus['p20']) -
                               minus['p1']/(minus['p2j'] if state=='j' else minus['p20']))/(2*h)
            assert np.isclose(result['d_matched_ratio'], numerical_ratio, atol=2e-7)
            if state == '0':
                numerical_ih = (plus['p1']/plus['p2j'] - minus['p1']/minus['p2j'])/(2*h)
                assert np.isclose(result['d_induce_ratio'], numerical_ih, atol=2e-7)
    print('All analytical derivative checks passed.')


def plot_series(rows, key, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for state,label in [('j',r'Initially inside ($x_0=j$)'),('0',r'Initially outside ($x_0=0$)')]:
        ax.plot(GAMMAS, [r[key] for r in rows[state]], label=label, linewidth=2)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xlabel=r'$\gamma$', ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT/filename, dpi=180)
    plt.close(fig)


def main():
    verify()
    OUTPUT.mkdir(exist_ok=True)
    rows = {state:[evaluate(float(g),state) for g in GAMMAS] for state in ['j','0']}
    with (OUTPUT/'model_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows['j'][0]))
        writer.writeheader()
        for state in ['j','0']:
            writer.writerows(rows[state])
    plot_series(rows, 'p1', r'$p_1^*$', 'price1.png', 'Optimal period-1 prices')
    plot_series(rows, 'dp1', r'$dp_1^*/d\gamma$', 'dp1.png', 'First derivative of period-1 price')
    plot_series(rows, 'd2p1', r'$d^2p_1^*/d\gamma^2$', 'd2p1.png', 'Second derivative of period-1 price (analytic)')
    plot_series(rows, 'dprofit', r'$d\Pi^*/d\gamma$', 'dprofit.png', 'Optimized profit derivative')
    plot_series(rows, 'd_matched_ratio', r'$d(p_1/p_2)/d\gamma$', 'matched_price_ratio.png', 'State-matched price-ratio derivatives')
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    ax.plot(GAMMAS, [r['d_induce_ratio'] for r in rows['0']], linewidth=2)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set(xlabel=r'$\gamma$', ylabel=r'$d[p_1(0)/p_2(j)]/d\gamma$', title='Induce-then-harvest price-ratio derivative')
    fig.tight_layout();fig.savefig(OUTPUT/'induce_harvest_ratio.png',dpi=180);plt.close(fig)
    for state in ['j','0']:
        fig,ax=plt.subplots(figsize=(8.6,5.2))
        for key,label in [('p1',r'$p_1^*$'),('static_component','Static FOC component'),('continuation_adjustment',r'$-\delta\Delta V_2$')]:
            ax.plot(GAMMAS,[r[key] for r in rows[state]],label=label,linewidth=2)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(xlabel=r'$\gamma$',ylabel='Price / component',title=f'First-period decomposition: initial state {state}')
        ax.legend();fig.tight_layout();fig.savefig(OUTPUT/f'decomposition_{state}.png',dpi=180);plt.close(fig)
    print('Saved outputs to', OUTPUT.resolve())

if __name__ == '__main__':
    main()
