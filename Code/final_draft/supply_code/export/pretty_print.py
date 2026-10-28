import sympy as sp


def show(title, expr):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    sp.pprint(expr)
