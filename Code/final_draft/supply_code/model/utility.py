import sympy as sp
from model.symbols import *


def signed_state(state):
    if state == "j":
        return -1
    if state == "0":
        return 1
    raise ValueError("state must be 'j' or '0'")


def raw_utilities(price, signed_history):
    return (beta + alpha*price + gamma*(1+signed_history)/2,
            gamma*(1-signed_history)/2)

raw_inside1, raw_outside1 = raw_utilities(p1, q)
raw_inside2, raw_outside2 = raw_utilities(p2, q2)
U1 = sp.expand(raw_inside1 - raw_outside1)
U2 = sp.expand(raw_inside2 - raw_outside2)
