"""Export the model's actual SymPy expressions as display equations."""
import sympy as sp


def display_equation(lhs, expression):
    """Keep the mathematical right-hand side sourced directly from SymPy."""
    return '\\[\n' + lhs + ' = ' + sp.latex(expression) + '\n\\]\n'
