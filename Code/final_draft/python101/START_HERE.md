# Python 101: learning to read this research code

This is a separate companion to the economic and pipeline guides. Those guides
explain the models. This book teaches you to read their implementation, beginning
with Python punctuation and ending with each function and script in `data_code`,
`est_code`, `supply_code`, and `pricing_exercises`, together with the current
readable demand workflow in `final_simple/demand`. You do not need to understand all the economics
before learning what a line of Python does.

The book contains the actual source, source line references, function-specific
explanations, statement-by-statement reading notes, alternatives, and small
examples. Nested helper functions and test methods are included. Scripts with
no functions receive the same treatment at module level. Empty package files
are identified explicitly. Source excerpts are a snapshot of the working tree
at build time; they are not edited versions of the executable research code.

**Current edition, November 6, 2026.** For the revised empirical exercise, read
worked lessons 11–16 after the foundations, then the chapters for
`../final_simple/demand/estimation.py` and `split_pricing.py`. The split model
estimates independent novelty and own-stock effects; novelty defines reporting
groups while both effects enter demand. All three flavors are priced separately
as the focal inside product. The regeneration and distribution-export chapters
explain how to recover the deleted weekly file on the server using the saved fit.
The original combined-gamma examples remain useful comparisons.

Read the foundations first, then the small `make_draws` and `signed_state`
functions. After that, try `add_stocks`, `resample_households`, and
`optimal_price`. Return to the long preparation functions after you can recognize
loops, columns, masks, and returned values. The table of contents and PDF
bookmarks let you find any function by file and name. Appendix tests are useful
worked examples of how to call the code without the private datasets.

## 1. A program is a sequence of instructions

Python executes statements in order. An assignment evaluates its right-hand
side, then binds its result to the name on its left:

```python
packages = 3
price = 1.50
spending = packages * price
```

`spending` is now 4.5. The equals sign is an instruction to store a result;
it is not a mathematical claim to solve. `spending == 4.5` instead asks a
question and produces the Boolean value `True`.

Names such as `weekly` and `frame` are ordinary variable names chosen by the
author. They have no special meaning to Python. A name can later refer to a
different object. `frame = frame.copy()` first copies the old table and then
rebinds `frame` to the new one. Assigning `other = frame` does not copy the
table; it gives the same object another name.

`#` begins a comment outside a string. Python ignores the rest of that line.
Triple-quoted text at the start of a module, class, or function is a docstring:
it is documentation retrievable with `help`, not a hidden calculation.

## 2. Values, types, and containers

| Example | Type | How this code uses it |
|---|---|---|
| `3` | Integer, `int` | Counts, loop limits, seeds |
| `3.0`, `.7`, `1e-6` | Floating-point number, `float` | Prices, coefficients, tolerances |
| `"berry"` | String, `str` | Column labels, model names, paths |
| `True`, `False` | Boolean, `bool` | Settings and validation |
| `None` | The special absence value | Optional input not supplied |
| `["berry", "plain"]` | List | Ordered, changeable sequence |
| `("j", "0")` | Tuple | Ordered collection, often used for fixed choices |
| `{"price": 1.5}` | Dictionary, `dict` | Named values looked up by key |
| `{"upc", "flavor"}` | Set | Unordered distinct labels and membership checks |

Square brackets have several jobs. In `[1, 2]` they create a list. In
`items[0]` they select an item. In `frame["price"]` they select a named table
column. Read the object to the left before deciding what the brackets mean.

Python indexes start at zero. For `names = ["other", "berry", "plain"]`,
`names[0]` is other and `names[2]` is plain. `names[:2]` selects positions zero
and one; slice stops are excluded. `names[1:]` starts at berry and goes to the
end. A NumPy slice such as `prices[:, j]` means every row of column j.

`dict(zip(names, theta))` pairs labels with values. If the labels are
`["beta", "gamma"]` and values `[2.0, -.5]`, the result is
`{"beta": 2.0, "gamma": -.5}`. Ordinary `zip` stops at the shorter input;
the surrounding code is responsible for consistent lengths. `p["gamma"]`
requires the key; `p.get("gamma", 0)` supplies zero if it is absent.

`a, b = (10, 20)` unpacks two returned values. `_` is a conventional name for
an ignored value; Python does not magically discard it. In `fit, _, _ = ...`,
the author keeps the fit and does not use the other returned objects.

## 3. Functions: inputs, work, and outputs

```python
def package_cost(quantity, price=1.0):
    total = quantity * price
    return total

answer = package_cost(3, price=1.5)
```

`def` defines a function. The name is `package_cost`; parentheses contain its
parameters. The colon introduces the body. Indentation determines which
statements belong to that body. Defining the function does not yet execute
its body. Calling it with parentheses does.

In the call, 3 is a positional argument and `price=1.5` is a keyword argument.
The `=` inside a call names an argument; it is not assignment to a variable
outside the function. `price=1.0` in the definition supplies a default when
the caller omits that argument. `return` ends this call and gives its value
back to the caller. Printing 4.5 displays text; returning 4.5 lets another
calculation use the number. A function that falls off its end returns `None`.

`def solve(*, initial_state="0", **parameters)` contains two more pieces of
syntax: bare `*` makes following arguments keyword-only; `**parameters` collects
additional named arguments into a dictionary. In a call,
`evaluate(gamma, state, **parameters)` expands such a dictionary back into
named arguments. `*command` in a list expands a sequence into its elements.

Functions defined inside another function are nested functions. They can read
names from the surrounding call. In `optimize_demand`, the nested `objective`
remembers `data`, `draws`, and `args`; SciPy only has to pass the changing
parameter vector. This is a closure. It avoids global variables, whose contents
could accidentally affect another fit.

`lambda p: p * p` is a compact function returning the square of p. It is handy
for a one-expression root equation. A named `def` is clearer if the calculation
needs several steps, comments, or reuse. A function object can itself be passed
as an argument: `brentq(f, lo, hi)` asks a solver to call f repeatedly.

## 4. Branches, loops, and scope

```python
if quantity <= 0:
    raise ValueError("Quantity must be positive")
else:
    print("Usable quantity")
```

`if` evaluates a condition, `elif` tries another condition when the previous
one was false, and `else` handles the remaining case. `raise` stops normal
execution by creating an exception. Rejecting unusable counts is preferable
to quietly turning them into valid-looking zeros.

`for j, name in enumerate(FLAVORS):` repeats a block once for each flavor,
providing both its numeric position and its label. It is equivalent to looping
over positions and looking up labels, but makes the correspondence explicit.
`range(3)` produces 0, 1, 2. A `while` loop instead keeps going while its condition
is true; the price solver uses it because it does not know in advance how many
bracket expansions are needed.

`["price_" + f for f in FLAVORS]` is a list comprehension. A slower-to-write
but equally valid spelling is:

```python
columns = []
for flavor in FLAVORS:
    columns.append("price_" + flavor)
```

An optional `if` at the end filters elements. `{key: value for ...}` builds a
dictionary. `(value for ...)` is a generator expression: it yields values as
needed rather than first making a complete list. `next(generator, None)` takes
the first yielded value, or returns `None` if no match exists.

`try` executes code that might fail. A matching `except` handles named errors.
The bootstrap records identification failures rather than accepting them as
estimates. Catching every possible exception would also hide programming bugs,
so it catches specific expected failure types.

## 5. Arrays: many numbers with one operation

NumPy arrays are rectangular numerical containers. Their `shape` is a tuple
of axis lengths. A three-week, three-flavor table has shape `(3, 3)`. Axis zero
indexes rows; axis one indexes columns. Consequently, `.sum(axis=1)` adds across
flavors for each week, while `.sum(axis=0)` adds across weeks for each flavor.

```python
import numpy as np
counts = np.array([[2., 0., 1.], [0., 4., 0.]])
totals = counts.sum(axis=1)             # [3., 4.]
shares = counts / totals[:, None]
```

`totals[:, None]` changes a shape `(2,)` vector into a `(2, 1)` column. NumPy
then broadcasts each row total across its three columns. Without that extra
axis, `(2,3) / (2,)` cannot broadcast: the trailing dimensions 3 and 2 conflict.
Broadcasting is controlled shape matching, not a guess at intended economics.

Similarly, adding a three-element beta vector to a W-by-3 utility matrix adds
the flavor intercepts to every row. Array multiplication `*` is elementwise.
Matrix multiplication `@` sums products over the matching inner dimension.
For one-dimensional vectors it is the dot product, e.g. `[1,2] @ [3,4] = 11`.
`**2` squares a value; `^` is not Python's exponentiation operator.

`np.zeros((W,3))` allocates an array of zeros. `np.zeros_like(prices)` matches
another array's shape and dtype. Preallocating avoids repeatedly growing an
array inside a long loop. `np.column_stack` joins columns; `np.vstack` stacks
rows. `np.bincount(hh, weights=row_ll)` sums contributions by nonnegative integer
household label, avoiding a slower Python loop over each household.

`np.nan` represents a missing floating-point value; `np.inf` represents positive
infinity. NaN is not equal to itself, so use `np.isnan` or table missing-value
methods rather than `x == np.nan`. `np.isfinite` rejects both NaN and infinities.

## 6. Masks and comparisons

```python
covered = np.isfinite(prices) & (prices > 0)
clean_prices = np.where(covered, prices, 0.)
```

A mask is an array of True/False values. `&` combines array conditions with
elementwise AND; `|` is elementwise OR; `~` reverses Booleans. Put each comparison
in parentheses because Python operator precedence otherwise changes the grouping.
`and`, `or`, and `not` instead combine scalar truth values and short-circuit;
they generally cannot combine an entire pandas Series unambiguously.

`.any()` asks whether at least one entry is true; `.all()` asks whether every
entry is true. `available.any(axis=1)` asks separately for each row. A useful
translation is: `((counts == 0) | available).all(axis=1)` means "every flavor
was either unpurchased or available." It rejects a purchased flavor without a price.

`np.where` selects values element by element. It does not lazily evaluate its
two alternatives: both argument expressions are evaluated before the call.
Use safe denominators or explicit indexing when the unused branch would raise
or perform an invalid calculation. A zero price placeholder is not a free good
when the corresponding alternative is subsequently masked out of utility.

## 7. pandas tables and Polars query expressions

A pandas `DataFrame` is a table with named columns and row labels; a `Series`
is a single labeled column. `frame["price"]` selects one Series;
`frame[["price", "quantity"]]` selects a two-column DataFrame.
`frame.loc[mask, "price"]` selects specified rows and one column.
`frame.iloc[0]` selects by numeric row position. `.copy()` makes an independent
table before modifications, avoiding accidental changes to the original.

`.groupby(keys)["quantity"].sum()` splits rows by common keys and sums within
each group. `.transform("nunique")` broadcasts a group's distinct count back
to every original row. In contrast, an ordinary group aggregation returns one
row per group. `.unstack` and `.pivot` turn flavor labels into columns;
`.reindex(columns=FLAVORS)` makes their ordering explicit for numerical arrays.

A merge is a lookup using matching key columns. `how="left"` keeps left rows,
`inner` keeps matches, `semi` keeps only matching left rows without adding right
columns, and `anti` keeps unmatched left rows. A duplicate right key can multiply
rows. `validate="many_to_one"` checks the right side is unique; it does not
force an invalid table to become unique. Deduplication and validation are
different decisions.

Polars uses `pl.col("price")` to build a column expression. Think of it as a
recipe, not yet a column of numbers. `pl.lit(0)` is a literal-valued expression;
`.alias("price")` names an expression's output. In `pl.when(condition).then(a)
.otherwise(b)`, each row chooses an outcome. `.with_columns` adds or replaces
columns, while `.select` projects only the requested columns/expressions.

`pl.scan_parquet(path)` builds a lazy plan. `.filter`, `.select`, and `.join`
extend that plan. `.collect()` executes it and materializes a table in memory;
`.sink_parquet(path)` executes it while writing output. This allows early
projection/filtering and streaming execution. It does not guarantee tiny memory:
joins, sorts, and exact distinct counts still need state. `.to_pandas()` converts
to an eager pandas table, appropriate for the later NumPy computations.

Method chaining means "take the result so far, then call the next method":

```python
cleaned = frame.dropna().sort_values("week").reset_index(drop=True)
```

The code often places the chain inside parentheses so it can span physical
lines. Leading dots on later lines continue the same expression; they are not
new statements. Nested function calls evaluate their arguments before the outer
call. Starting with the innermost parentheses is often the easiest reading strategy.

## 8. Strings, regular expressions, and paths

`"price_" + flavor` concatenates text. `f"Start {s + 1}"` inserts an expression
into a formatted string. `:.4f` means four digits after the decimal point,
`:,` adds thousands separators, and `:.1%` displays a fraction as a percentage.
`flush=True` asks printed progress to reach the log promptly instead of waiting
in a buffer.

Regexes describe string patterns. In `r"[0-9]+(?:\.0+)?"`, `[0-9]+` means
one or more digits, `(?:...)` groups without capturing, `\.` is a literal dot,
and `?` makes the decimal-zero suffix optional. The raw-string prefix r keeps
Python from processing backslashes first. `^` and `$` anchor start and end.
The UPC normalization removes only a numeric `.0` suffix, not every decimal
point or arbitrary letters. String handling avoids rounding a large UPC through
a floating-point number.

`Path("/scratch") / "Data"` joins filesystem components. The slash is overloaded
by `Path`; here it does not divide numbers. `.exists()` checks presence,
`.mkdir(parents=True, exist_ok=True)` creates needed directories, and `.resolve()`
normalizes a path for comparisons. A `with TemporaryDirectory() as tmp:` block
cleans up the temporary directory on exit, including ordinary exception exits.
`with path.open("w") as file:` similarly ensures the file is closed.

## 9. Numerical optimization and symbolic algebra are different

`minimize(objective, initial, ...)` repeatedly calls a numerical function,
changing an array of parameter values to lower its returned loss. `jac=True`
means this objective supplies both the number and its derivative vector.
`brentq(f, lo, hi)` finds a zero of a continuous scalar function with a sign-changing
bracket; it does not minimize the function. A root of a profit derivative is a
candidate optimum, whose optimality needs the argument in the economic guide.

`sp.symbols("alpha beta")` creates symbolic mathematical placeholders. It does
not create estimated numerical values. `sp.diff(expr, alpha)` differentiates
algebra; `.subs({alpha: -1})` substitutes; `sp.simplify` tries to reduce an
expression. `sp.lambdify(arguments, expression, "numpy")` turns that symbolic
formula into a callable numerical function with an explicit argument order.

A crucial trap: SymPy treats distinct symbols as independent unless dependence
is encoded. If `s` is a separate symbol, differentiating `s*p` with respect to p
gives s, not `s + p*ds/dp`. The supply code's total-derivative helper manually
adds that chain-rule contribution for equilibrium shares and prices.

For likelihood calculations, `logsumexp` computes the logarithm of a sum of
exponentials stably. `np.log(np.exp(x).sum())` can overflow even when its final
mathematical answer is finite. The optimizer would then receive unusable values.
The analytic gradient is exact for the fixed simulated draws, while the
simulation integral itself still approximates the population integral.

## 10. Modules, entry points, and tests

`import numpy as np` imports a library and gives it a short local name.
`from price_control import check_dma` imports one object from a neighboring
module. Imports can execute a module's top-level code. That is why the legacy
cleaning scripts should not be imported merely for inspection: they perform
data work at module level.

```python
if __name__ == "__main__":
    main()
```

When Python runs a file directly, its `__name__` is `"__main__"`, so this block
calls its entry point. When another file imports it, the guard is false and
the guarded work is skipped. ArgumentParser translates command-line text into
attributes such as `args.draws`; `type=int` converts text to integers and
`action="store_true"` creates a Boolean flag.

A class groups behavior. `class Checks(unittest.TestCase):` inherits the testing
tools. A method is a function attached to a class; `self` refers to the current
instance and is supplied automatically in calls such as `self.frame()`.
`self.assertEqual(actual, expected)` fails the test if the values differ.
Floating-point results need tolerances: `.1 + .2` is generally not represented
exactly as .3. `assert_allclose` compares arrays element by element with stated
absolute/relative tolerances. A passing test confirms its constructed case,
not every possible input or the truth of an economic assumption.

## 11. A tiny trace you can do on paper

```python
stock = np.zeros(3)
counts = np.array([0., 2., 0.])
share = counts / counts.sum()
before = stock.copy()
stock = stock + share
```

Initially stock is `[0,0,0]`. Counts sum to 2, so share is `[0,1,0]`.
`before` preserves zero history; stock becomes `[0,1,0]`. The order explains
why the current purchase does not enter its own pre-choice state. Reversing the
last two statements would change the statistical model, even though both
versions are valid Python.

Try these questions before looking back: what does `np.arange(4)` return? Why
is `frame[["price"]]` different from `frame["price"]`? When does a nested
objective run: when it is defined or when SciPy calls it? Does `new = old`
copy a table? Does `.all(axis=1)` summarize rows or columns? The answers are
0–3, table versus Series, when called, no, and one result per row after
combining columns. These are the building blocks used repeatedly below.
