"""Syntax-reading helpers for the documentation builder; never executes source."""
import ast


def short(node, limit=100):
    text = ast.unparse(node).replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def code(text):
    return "`" + text.replace("`", "'") + "`"


def value_description(node):
    """Describe this expression's concrete construction, without executing it."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            if len(node.value) > 140:
                return "a multiline text template; its contents are data, not Python instructions"
            return "the literal text " + code(repr(node.value)[:65])
        return "the literal value " + code(repr(node.value))
    if isinstance(node, ast.Dict):
        keys = [str(k.value) for k in node.keys if isinstance(k, ast.Constant)]
        return "a dictionary" + (" with named entries " + ", ".join(code(k) for k in keys[:8]) if keys else " from the displayed keys and values")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        kind = {ast.List: "list", ast.Tuple: "tuple", ast.Set: "set"}[type(node)]
        if len(node.elts) <= 4 and all(isinstance(x, (ast.Constant, ast.Name)) for x in node.elts):
            return f"the {kind} " + code(short(node, 60))
        return f"a {kind} containing {len(node.elts)} displayed entries"
    if isinstance(node, ast.ListComp):
        return "a list computed by the displayed for-expression" + (", retaining only entries passing its if filter" if any(x.ifs for x in node.generators) else "")
    if isinstance(node, ast.DictComp):
        return "a dictionary computed one key/value pair at a time by the comprehension"
    if isinstance(node, ast.GeneratorExp):
        return "a lazy iterator from the generator expression"
    if isinstance(node, ast.Lambda):
        return "a one-expression callable, with input " + code(short(node.args, 35)) + " and returned expression " + code(short(node.body, 55))
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        # Method chains can be hundreds of characters. The complete expression
        # is immediately above; identify its last operation here.
        chained = any(isinstance(n, ast.Call) for n in ast.walk(node.func))
        label = name if len(name) < 40 and not chained else "." + name.rsplit(".", 1)[-1]
        return "the result of calling " + code(label + "(...)") + "; the arguments and preceding method chain are shown above"
    if isinstance(node, ast.Subscript):
        return "the entry/slice selected by " + code(short(node, 60)) + "; brackets select from the object to their left"
    if isinstance(node, ast.Attribute):
        return "the attribute " + code(short(node, 60)) + " of the object before the dot"
    if isinstance(node, ast.Name):
        return "the object currently named " + code(node.id) + "; plain assignment does not copy it"
    if isinstance(node, ast.BinOp):
        op = {ast.Add: "addition/concatenation", ast.Sub: "subtraction", ast.Mult: "multiplication", ast.Div: "division", ast.MatMult: "matrix multiplication", ast.Pow: "exponentiation", ast.Mod: "remainder", ast.FloorDiv: "floor division", ast.BitAnd: "elementwise AND", ast.BitOr: "elementwise OR"}.get(type(node.op), "the displayed operation")
        return op + " of the displayed operands; parentheses control which subexpressions are evaluated together"
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return "the Boolean condition " + code(short(node, 65)) + "; it can be a scalar or a row/array mask depending on its inputs"
    if isinstance(node, ast.IfExp):
        return "the true-branch value if " + code(short(node.test, 50)) + " holds, otherwise the else-branch value"
    if isinstance(node, ast.JoinedStr):
        return "formatted text, substituting the brace-enclosed values into the f-string"
    if isinstance(node, ast.UnaryOp):
        return "the negated or sign-adjusted expression " + code(short(node, 60))
    return "the displayed right-hand expression"


METHODS = {
    "scan_parquet": "Build a lazy Parquet read plan; rows are not yet materialized in a Python table.",
    "scan_csv": "Build a lazy delimited-text read plan; parser options determine types and missing values.",
    "read_csv": "Read a CSV into an eager table; unlike a lazy scan, the requested data are loaded now.",
    "read_parquet": "Read Parquet into an eager table, rather than retaining only a deferred plan.",
    "collect_schema": "Inspect column names/types of a lazy plan without collecting its entire result table.",
    "names": "Retrieve schema column names as a sequence, useful for checking optional fields.",
    "select": "Choose output columns or evaluate selected expressions; unselected columns do not automatically remain.",
    "with_columns": "Add or replace named columns while keeping the other columns in the table.",
    "filter": "Retain rows whose Boolean condition is true; missing-condition behavior depends on the table library.",
    "col": "Build a Polars column-expression recipe; the string identifies a column rather than a literal value.",
    "lit": "Build a literal-valued Polars expression, distinguishing constant text/numbers from column names.",
    "when": "Start a Polars row-wise conditional expression, completed by then and otherwise.",
    "then": "Supply the value/expression selected where the preceding Polars condition is true.",
    "otherwise": "Supply the fallback expression for rows not selected by the preceding condition.",
    "alias": "Name a Polars expression's result; this often determines which column with_columns replaces.",
    "cast": "Request a dtype conversion; strict=False can turn invalid conversions into nulls for later validation.",
    "astype": "Convert the pandas/NumPy object's dtype; this changes representation, not merely its displayed label.",
    "to_numpy": "Convert table/Series values to a numerical array for subsequent array arithmetic.",
    "to_pandas": "Convert a Polars table to pandas, crossing from query/table preparation to eager labeled analysis.",
    "from_pandas": "Convert a pandas object into a Polars table; calling lazy afterward makes a query plan from it.",
    "lazy": "Wrap an eager Polars table as a lazy query so later operations can be planned together.",
    "collect": "Execute a lazy Polars query and materialize its resulting table; this can use substantial memory.",
    "sink_parquet": "Execute/write a lazy result to Parquet instead of first keeping its whole result in a Python table.",
    "sink_csv": "Execute/write the query result as CSV; unlike select/filter, this is a filesystem side effect.",
    "write_csv": "Write an eager Polars table to CSV; it does not return a newly analyzed dataset.",
    "to_csv": "Serialize a pandas table as CSV; index=False avoids writing row labels as an extra column.",
    "to_latex": "Serialize a pandas table as LaTeX source, which needs a separate TeX compilation to become PDF.",
    "write_parquet": "Persist an eager table as Parquet, often used for small synthetic fixtures.",
    "groupby": "Partition pandas rows by key values; the following aggregation/transform specifies what each group produces.",
    "group_by": "Partition Polars rows by key expressions before evaluating per-group aggregations.",
    "agg": "Evaluate aggregations within the current groups; the selected expressions define output columns.",
    "sum": "Sum values along the specified axis or within each group; check the receiver and axis to identify the unit.",
    "mean": "Compute an arithmetic mean over the selected group/axis, not automatically a weighted mean.",
    "median": "Compute the middle-value statistic rather than a mean; global medians cannot generally be averaged across batches.",
    "std": "Compute standard deviation; ddof=1 requests sample SD rather than dividing squared deviations by the full count.",
    "min": "Take the smallest value, or a per-group/axis minimum when used on a table expression.",
    "max": "Take the largest value, or a per-group/axis maximum when used on a table expression.",
    "nunique": "Count distinct pandas values, rather than counting repeated purchase rows as separate identities.",
    "n_unique": "Count distinct Polars values within the relevant aggregation scope.",
    "unique": "Retain distinct rows/values; subset and keep determine the keys and duplicate-resolution rule.",
    "drop_duplicates": "Retain one row per selected key; keep='first' makes earlier sorting/order economically consequential.",
    "duplicated": "Flag repeated keys/rows so validation can detect ambiguity instead of resolving it silently.",
    "merge": "Join pandas tables through matching keys; how controls retention and validate checks intended key cardinality.",
    "join": "Join table/query rows on keys; left, inner, semi, and anti joins have different retention meanings.",
    "rename": "Change column labels (or apply a renaming function); this does not change their underlying values.",
    "drop": "Remove the specified columns/labels; inspect the receiver because list and table methods have different APIs.",
    "sort": "Order values/rows; sorting before first-row selection determines which record wins a tie.",
    "sort_values": "Order pandas rows by values; ascending can supply one direction for each listed sort key.",
    "sort_by": "Order values within an expression/group by another column before selecting a first value.",
    "reset_index": "Turn the index back into ordinary row numbering; drop=True discards the old index rather than saving it as a column.",
    "set_index": "Use one or more columns as pandas row labels, making named lookup convenient.",
    "reindex": "Align to an explicitly requested row/column order, adding missing labels with a chosen fill value if provided.",
    "unstack": "Move an index level into columns, such as turning long flavor counts into wide weekly outcomes.",
    "pivot": "Reshape long labels into matrix columns; index/columns/values specify their roles.",
    "transform": "Compute a pandas group result broadcast back to original rows, unlike a reduced one-row-per-group aggregation.",
    "factorize": "Encode labels as dense integer positions and return the original-label lookup alongside those positions.",
    "fillna": "Replace pandas missing entries with the provided fallback; it can change downstream denominators or validity checks.",
    "fill_null": "Replace Polars nulls with a chosen value/expression; null and floating-point NaN are distinct representations.",
    "dropna": "Remove pandas missing observations before the following calculation.",
    "drop_nulls": "Remove null values from the Polars expression before selecting a first observed value.",
    "isna": "Create a pandas mask marking missing values; compare masks rather than using equality to NaN.",
    "notna": "Create a pandas mask marking nonmissing values; it does not establish numeric positivity or finiteness.",
    "is_null": "Create a Polars null-value mask, useful when validating joins and conversions.",
    "is_not_null": "Create a Polars nonnull mask; additional range checks can still be needed.",
    "isfinite": "Test for finite numbers, rejecting both NaN and positive/negative infinity.",
    "is_finite": "Build the Polars finite-number test used to reject missing/infinite quotes.",
    "isnan": "Identify floating-point NaNs; NaN must not be tested with ordinary equality.",
    "isin": "Test each pandas entry's membership in an allowed sequence, producing a Boolean Series.",
    "is_in": "Build a Polars membership expression, often used for eligible household IDs or allowed labels.",
    "is_between": "Build an inclusive interval check by default, such as selecting the requested panel-year range.",
    "where": "For pandas receivers keep original values where the condition is true and use the fallback elsewhere; np.where instead takes explicit true/false branches.",
    "copy": "Make an independent container before changing it; plain assignment would give the original object another name.",
    "append": "Add one item to the end of a Python list; it mutates the list and normally returns None.",
    "extend": "Append every element from an iterable, unlike append which would add that iterable as one item.",
    "update": "Add/replace dictionary entries in place; unrelated entries remain available.",
    "get": "Retrieve a dictionary/Series entry with a fallback for an absent key; direct bracket lookup would require it.",
    "pop": "Retrieve and remove a dictionary entry, using the supplied default if it is absent.",
    "items": "Iterate through dictionary key/value pairs together, allowing two-name unpacking in a loop.",
    "tolist": "Convert a NumPy/pandas sequence to ordinary Python list values, useful for JSON output or comparisons.",
    "item": "Extract a scalar from a one-element array/one-cell result rather than passing a table wrapper onward.",
    "to_dict": "Convert table contents to a Python dictionary; output orientation/options determine its nested layout.",
    "strip": "Remove surrounding string whitespace; it does not remove internal spaces.",
    "strip_chars": "Trim surrounding characters/whitespace in a Polars string expression.",
    "lstrip": "Remove any leading occurrences of the specified character set, such as zeros in numeric UPC text.",
    "strip_chars_start": "Trim characters from the start of each Polars string; it does not remove matching characters inside it.",
    "lower": "Convert pandas/Python text to lowercase so comparisons need not list every capitalization.",
    "to_lowercase": "Build a lowercase Polars string expression for reproducible label comparisons.",
    "fullmatch": "Require the entire string to match a regex; contains would allow a partial match inside unrelated text.",
    "contains": "Test whether text contains the supplied regex pattern; anchors can make this an entire-string test.",
    "replace": "Replace matched values/text; pandas values, regex string namespaces, and Python strings have different replacement semantics.",
    "replace_all": "Replace every regex match within each string, such as collapsing repeated whitespace.",
    "slice": "Take a substring starting at a position for a given length, here often the first ten date characters.",
    "to_date": "Parse text into calendar-date values before weekday extraction and date arithmetic.",
    "to_datetime": "Parse/convert dates; an explicit format avoids interpreting numeric YYYYMMDD values as timestamp offsets.",
    "weekday": "Extract a Polars weekday number, Monday=1 through Sunday=7, for the Saturday offset calculation.",
    "duration": "Build a date/time duration from the named unit, allowing a different day offset per row.",
    "strftime": "Format a parsed date according to directives, e.g. %Y%m%d for an eight-digit week-ending key.",
    "total_days": "Convert a duration expression into its elapsed number of days before comparing weekly gaps.",
    "shift": "Move values by one row by default; over(household) prevents the shift crossing household boundaries.",
    "over": "Evaluate a Polars expression within the specified group/window, preserving row-level results.",
    "diff": "For SymPy differentiate algebra; for a Polars/pandas date/value expression calculate successive differences. The receiver decides which operation this name means.",
    "cum_sum": "Accumulate running totals, used to turn spell-start flags into spell IDs.",
    "first": "Select the first value in the current expression/group; preceding ordering and null filtering determine which one.",
    "pow": "Raise each expression value to a power, such as squaring quantities for concentration calculations.",
    "div": "Divide a pandas object with explicit alignment options; axis=0 aligns the denominator by rows.",
    "eq": "Compare elementwise for equality, the method equivalent of == for these pandas values.",
    "any": "Reduce Boolean values to whether at least one is true; an axis can request a result for every row/column.",
    "all": "Reduce Boolean values to whether every value is true; with axis=1 the result has one entry per row.",
    "array": "Create a NumPy array; nested equal-length lists form rows and columns.",
    "asarray": "View/convert data as an array without demanding a copy when the representation already fits.",
    "zeros": "Allocate zero-filled numerical storage of the requested shape before accumulation.",
    "zeros_like": "Create zeros matching another array's shape/dtype instead of manually repeating dimensions.",
    "full": "Allocate an array filled with one value, such as equal histogram weights.",
    "arange": "Create evenly stepped values with an excluded stop, commonly row positions 0 through n-1.",
    "linspace": "Create the requested number of evenly spaced points, normally including both endpoints.",
    "vstack": "Stack input arrays vertically, adding rows while preserving their common column layout.",
    "column_stack": "Place one-dimensional vectors beside each other as columns of a matrix.",
    "bincount": "Accumulate frequencies or weighted sums by dense nonnegative integer labels; repeated household codes are added together.",
    "add.at": "Accumulate in place at possibly repeated indices; this avoids the lost-repeat behavior of buffered fancy-index updates.",
    "exp": "Apply the exponential, elementwise for NumPy and symbolically for SymPy; it is not multiplication by a constant.",
    "log": "Take the natural logarithm, used here for log likelihoods and integration normalization.",
    "clip": "Cap values at lower/upper limits; here it keeps starts feasible or exponentials within floating-point range.",
    "maximum": "Take an elementwise maximum, e.g. max(capacity minus packages, zero) independently each week.",
    "round": "Round numeric values for a check/display rule; rounding quantities is compared to the original before acceptance.",
    "floor": "Take the greatest integer no larger than each value; equality with it tests whether a positive code is integral.",
    "abs": "Take absolute magnitude, as needed for tolerance tests and bound distances.",
    "allclose": "Check approximate numerical equality with tolerances, appropriate for floating-point arrays.",
    "isclose": "Create approximate-equality flags rather than requiring exact floating-point representation.",
    "einsum": "Perform an explicitly labeled sum of products; rh,rhp->p sums draws/households and retains the parameter axis.",
    "lstsq": "Solve least squares without explicitly inverting a cross-product matrix; [0] keeps its coefficient vector.",
    "cov": "Compute covariance; rowvar=False means columns represent parameters rather than treating rows as variables.",
    "quantile": "Find empirical quantiles, supplying percentile bootstrap endpoints rather than normal-Wald formulas.",
    "ceil": "Round upward, so the minimum usable-bootstrap threshold meets the requested success fraction.",
    "default_rng": "Create a local seeded random Generator, avoiding dependence on a shared global random stream.",
    "standard_normal": "Draw independent standard-normal values in the requested shape, before coefficient transformations.",
    "normal": "Draw normal random values with the supplied location/scale, used for starts or synthetic data.",
    "integers": "Draw integer positions with replacement by default; the high endpoint is excluded.",
    "uniform": "Draw values from a continuous uniform interval for controlled synthetic inputs.",
    "multinomial": "Draw a vector of category counts summing to the specified total, given category probabilities.",
    "repeat": "Repeat each element consecutively, e.g. each household ID across all its periods.",
    "tile": "Repeat a complete sequence, e.g. the period-date sequence for every household.",
    "symbols": "Create SymPy symbols, which stand for algebraic variables rather than observed numbers.",
    "expand": "Distribute products and combine terms in a symbolic expression without solving its unknowns.",
    "simplify": "Try algebraic simplification; it does not numerically optimize a price or establish every economic sign.",
    "factor": "Factor common algebraic pieces to make a symbolic derivative easier to inspect.",
    "subs": "Substitute symbols/expressions in symbolic algebra; this does not by itself solve an equilibrium.",
    "lambdify": "Compile a symbolic expression into a numerical callable with an explicitly ordered argument list.",
    "latex": "Serialize symbolic algebra as LaTeX source rather than recomputing the mathematical formula manually.",
    "pprint": "Pretty-print symbolic mathematics to the terminal for human inspection.",
    "Eq": "Create a symbolic equality object; unlike assignment it does not bind its left side to its right side.",
    "subplots": "Create a Figure plus one or more Axes; the Axes hold plots and the Figure holds page-level layout.",
    "plot": "Draw a line from supplied x/y values on the receiving plotting axes.",
    "bar": "Draw bar heights from supplied categories and values; it does not compute the underlying shares.",
    "hist": "Bin observed values, with optional weights controlling whether heights are counts, percentages, or another unit.",
    "step": "Draw an empirical step curve; where='post' extends each value to the right.",
    "imshow": "Display a numerical matrix as colored cells; row/column label alignment remains the caller's responsibility.",
    "fill_between": "Shade the area between curves/zero, optionally only where a Boolean condition is true.",
    "axhline": "Add a horizontal reference line, commonly zero for a derivative plot.",
    "axvline": "Add a vertical reference line, such as zero gamma or a point-mass type.",
    "set": "Set multiple plot properties using named arguments rather than separate setter calls.",
    "set_ylabel": "Set the vertical-axis label, whose unit should match the plotted values.",
    "set_xlim": "Set the visible x-axis limits; this changes presentation rather than the calculated results.",
    "set_yticks": "Choose which vertical tick marks to show, including an empty list for a point-mass display.",
    "legend": "Render labels previously attached to plotted series, identifying curves.",
    "grid": "Add visual grid lines without changing underlying numerical results.",
    "tick_params": "Adjust axis tick appearance, for example rotating long category labels.",
    "text": "Place a text label on the axes/Figure at the supplied coordinates.",
    "colorbar": "Add a legend relating matrix colors to numerical values.",
    "tight_layout": "Adjust plot spacing to reduce overlap between labels and content.",
    "savefig": "Write the Figure to a file whose extension selects PDF/PNG; this is a filesystem side effect.",
    "close": "Release a Figure or file resource, according to the receiving object.",
    "use": "Select a Matplotlib backend; Agg renders to files without opening a GUI window.",
    "pdf": "Evaluate a probability density (here a normal distribution), not a PDF document file.",
    "cdf": "Evaluate cumulative probability below a threshold, used for the population share with negative gamma.",
    "sf": "Evaluate survival probability above a threshold, useful for a two-sided t diagnostic.",
    "exists": "Test whether a filesystem path exists; this is not a schema/content validity check.",
    "mkdir": "Create a directory; parents=True creates ancestors and exist_ok=True accepts an existing directory.",
    "glob": "Find filesystem paths matching a wildcard pattern, which may include stale files unless the folder is controlled.",
    "resolve": "Normalize a path to an absolute resolved location before locating siblings or comparing paths.",
    "with_name": "Construct a sibling Path with a replacement filename, preserving its parent directory.",
    "open": "Open a file in the requested mode; using a with block ensures it closes on exit.",
    "write_text": "Write a string into a text file, replacing an existing file at that path.",
    "getsize": "Read filesystem byte size, later divided by powers of 1024 for MiB/GiB display.",
    "remove": "Delete the named file; in these scripts it cleans generated temporary year chunks.",
    "rmdir": "Remove an empty directory after its temporary files have been deleted.",
    "monotonic": "Read a monotonic clock for elapsed durations rather than a wall-clock date/time.",
    "dumps": "Serialize Python dictionaries/lists/scalars into a JSON string, with optional readable indentation.",
    "add_argument": "Register a command-line option and its type/default/choices/flag behavior before parsing.",
    "parse_args": "Read command-line arguments into a Namespace of named attributes.",
    "error": "Report invalid command-line input and exit through argparse; it is not a recoverable data-cleaning step here.",
    "ArgumentParser": "Create the command-line parser object that will register and validate options.",
    "TemporaryDirectory": "Create temporary working storage whose context manager cleans it up when the block exits.",
    "DictWriter": "Create a CSV writer mapping dictionary keys to named columns in an explicit field order.",
    "writeheader": "Write the CSV's named column header before writing data rows.",
    "writerows": "Write a sequence of dictionaries/rows using the configured CSV schema.",
    "run": "For subprocess.run, execute an argument-list command; check=True raises if its exit status is nonzero.",
    "redirect_stdout": "Temporarily capture printed output, keeping tests quiet without changing the tested computation.",
    "StringIO": "Create an in-memory text buffer instead of writing captured output to a real file.",
    "get_dummies": "Construct indicator columns for categories, used as a small explicit fixed-effect benchmark in tests.",
    "assertEqual": "Fail this unittest method unless actual and expected objects are equal.",
    "assertAlmostEqual": "Fail unless two numbers agree to the requested decimal places or absolute delta.",
    "assertTrue": "Fail unless the supplied condition is true; array conditions normally need any/all first.",
    "assertFalse": "Fail unless the supplied condition is false, checking an expected absence/invalidity.",
    "assertLess": "Fail unless the first value is strictly smaller than the second.",
    "assertRaisesRegex": "Expect a particular exception and matching error text inside the with block; no exception is a test failure.",
    "assert_frame_equal": "Compare pandas tables including values, labels, and dtype conventions rather than only row counts.",
    "assert_allclose": "Compare arrays numerically with tolerances rather than requiring bit-for-bit floating-point equality.",
    "assert_array_equal": "Check that corresponding array entries agree exactly, appropriate for integer identities/counts.",
}

BUILTINS = {
    "len": "Count elements/rows rather than summing their numerical values.",
    "range": "Create a loop range whose stop is excluded; range(3) visits 0, 1, 2.",
    "enumerate": "Yield position/value pairs, allowing an index and label to be named together.",
    "zip": "Pair entries from sequences in order; by default iteration stops at the shortest input.",
    "dict": "Build a key/value mapping, often by pairing parameter names with numeric coefficients.",
    "list": "Materialize iterable entries as an ordered Python list.",
    "set": "Build a collection of distinct values, used for column membership and missing-field checks.",
    "sorted": "Return entries in deterministic sorted order, with optional key and reverse direction.",
    "next": "Take the first item from an iterator, or the supplied default when no item matches.",
    "getattr": "Retrieve an attribute by its string name, optionally supplying a default if it is absent.",
    "hasattr": "Check that an object exposes a named attribute before using an optional/version-dependent API.",
    "isinstance": "Check a runtime type to select the appropriate conversion path.",
    "all": "Require every generated condition to be true; this reduces a sequence to one Boolean.",
    "any": "Require at least one condition to be true; this reduces a sequence to one Boolean.",
    "float": "Convert a scalar to an ordinary Python floating-point value, useful for portable outputs.",
    "int": "Convert to an integer; it does not by itself validate that a source value was already integral.",
    "bool": "Convert scalar truthiness to an explicit Boolean flag.",
    "str": "Convert to text for paths, messages, or serialization.",
    "min": "Select the smallest entry; key= can select by objective value rather than object identity.",
    "max": "Select the largest entry, used for widths, tolerances, and diagnostic residuals.",
    "abs": "Take magnitude, removing a sign before a distance/tolerance comparison.",
    "sum": "Accumulate a sequence, including generated chain-rule terms or counts.",
    "map": "Apply a callable to each iterable entry lazily, such as converting result scalars to Python float.",
    "vars": "Expose an object's attribute dictionary, here the parsed settings saved with the result.",
    "print": "Display/log a value; this differs from returning it to another calculation. flush=True sends progress promptly.",
    "Path": "Construct a pathlib filesystem-path object, enabling / joining and path methods.",
    "logsumexp": "Compute log(sum(exp(values))) stably; axis selects alternatives, draws, or another reduction dimension.",
    "expit": "Evaluate the logistic function stably, avoiding overflow from a naive exp(x)/(1+exp(x)) formula.",
    "minimize": "Run iterative numerical minimization; jac=True expects the objective to return value plus gradient.",
    "brentq": "Find a scalar zero within a sign-changing bracket, rather than minimize the function itself.",
    "minimize_scalar": "Numerically minimize a one-variable objective, used as an independent check of price optimality.",
    "patch": "Temporarily replace/wrap a callable in a test so its invocations can be observed.",
    "OptimizeResult": "Construct a SciPy-style result object; test fixtures can provide one without a genuine optimizer run.",
    "SimpleNamespace": "Construct a lightweight object with named attributes, useful for args-like synthetic test settings.",
}


def call_name(node):
    return ast.unparse(node.func)


def operations(node):
    """Explain library calls and distinctive punctuation used by one code block."""
    notes = []
    seen = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = call_name(child)
        last = name.rsplit(".", 1)[-1]
        key = "add.at" if name.endswith("add.at") else last
        explanation = BUILTINS.get(name) if isinstance(child.func, ast.Name) else METHODS.get(key)
        if explanation is None and isinstance(child.func, ast.Name):
            if name in ("ValueError", "FileNotFoundError", "RuntimeError", "AssertionError"):
                explanation = "Construct an exception describing why this operation cannot proceed normally."
            else:
                explanation = "Call this project helper/callable with the displayed arguments; its own walkthrough gives the input/output contract."
        if explanation and (key, explanation) not in seen:
            # A complex receiver is already visible in the source; repeat only its method label here.
            label = name if len(name) < 65 else "." + last + "(...)"
            # A consistent method label also enables deduplication between
            # separate statements of the same function.
            label = name if isinstance(child.func, ast.Name) else "." + key
            notes.append(f"{code(label)}: {explanation}")
            seen.add((key, explanation))
    kinds = {type(n) for n in ast.walk(node)}
    syntax = [
        (ast.ListComp, "The bracketed for-expression is a list comprehension: one computed entry per loop item, optionally filtered by its final if."),
        (ast.DictComp, "The brace-enclosed key:value for-expression builds a dictionary, rather than an ordinary list."),
        (ast.GeneratorExp, "The parenthesized for-expression yields items as requested; it does not first allocate a complete list."),
        (ast.Lambda, "lambda introduces a short callable. Its parameter is left of the colon; its one returned expression is right of it."),
        (ast.MatMult, "The @ operator is matrix multiplication/dot product. It sums matching products; * would multiply elementwise."),
        (ast.Pow, "The ** operator raises to a power. Python's ^ operator would mean bitwise XOR instead."),
        (ast.FloorDiv, "The // operator is floor division, used here for integer counts or calendar-year extraction."),
        (ast.Mod, "The % operator takes a remainder. It is used for parity, repeating schedules, calendar offsets, or hash buckets."),
        (ast.BitAnd, "The & operator combines Boolean masks element by element; parenthesize the separate comparisons."),
        (ast.BitOr, "The | operator accepts either elementwise condition; Python's scalar or is generally unsuitable for an entire table column."),
        (ast.Invert, "The ~ operator reverses a Boolean mask here, changing true entries to false and vice versa."),
        (ast.IfExp, "The expression a if condition else b chooses one value; unlike a full if statement it fits inside another expression."),
        (ast.JoinedStr, "The f-prefixed string inserts expressions in braces; format suffixes control display without changing stored coefficients."),
        (ast.Starred, "A starred entry expands an iterable into separate elements, here within a list/tuple or call."),
    ]
    notes += [text for kind, text in syntax if kind in kinds]
    return notes


def describe(node):
    """Describe statements, retaining their original source line for lookup."""
    where = f"Line {node.lineno}"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if isinstance(node, ast.ImportFrom):
            names = ", ".join(x.name + (f" as {x.asname}" if x.asname else "") for x in node.names)
            return f"{where}: from {code(node.module or '')}, import {code(names)} into this module's namespace. Importing can execute the imported module's top-level setup."
        names = ", ".join(x.name + (f" as {x.asname}" if x.asname else "") for x in node.names)
        return f"{where}: import {code(names)}. An as-name is a convenient local alias, not another library."
    if isinstance(node, ast.Assign):
        targets = ", ".join(short(x, 65) for x in node.targets)
        if any(isinstance(x, (ast.Tuple, ast.List)) for x in node.targets):
            action = "Unpack " + value_description(node.value) + " into these targets in order."
        elif any(isinstance(x, ast.Subscript) for x in node.targets):
            action = "Store " + value_description(node.value) + " in the selected entry/column."
        else:
            action = "Set this name to " + value_description(node.value) + "."
        return f"{where}, {code(targets)}: {action}"
    if isinstance(node, ast.AugAssign):
        op = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=", ast.BitAnd: "&="}.get(type(node.op), "augmented assignment")
        return f"{where}: update {code(short(node.target, 65))} using {code(op)}. Numeric arrays update elementwise; lists extend under +=; strings concatenate under +=. Read the receiving object's type."
    if isinstance(node, ast.Return):
        if node.value is None:
            return f"{where}: bare return ends this call immediately with None. Later statements in the function are skipped."
        if isinstance(node.value, ast.Tuple):
            return f"{where}: return a tuple of {len(node.value.elts)} results. The caller can unpack them; return ends the call, unlike print."
        return f"{where}: return " + value_description(node.value) + ". This ends the call."
    if isinstance(node, ast.If):
        condition = short(node.test, 80)
        return f"{where}: test {code(condition)}. Execute the indented branch only when it is true; an else/elif branch handles its alternative."
    if isinstance(node, ast.For):
        return f"{where}: for each entry yielded by {code(short(node.iter, 65))}, bind/unpack it into {code(short(node.target, 45))} and run the indented body. Changes inside an iteration can feed subsequent iterations."
    if isinstance(node, ast.While):
        return f"{where}: repeat while {code(short(node.test, 70))} remains true. Unlike for/range, this loop need not know the iteration count beforehand; its body must eventually change the condition."
    if isinstance(node, ast.With):
        return f"{where}: enter the displayed context manager(s). An as-name receives the temporary resource; leaving the block restores/closes/cleans it according to that manager, including ordinary exception exits."
    if isinstance(node, ast.Try):
        return f"{where}: try the indented operation; matching except blocks handle specified exceptions. Unhandled error types still propagate instead of being silently treated as success."
    if isinstance(node, ast.ExceptHandler):
        return f"{where}: catch {code(short(node.type, 65)) if node.type else 'any exception'} from the preceding try; the optional as-name holds the error object for logging."
    if isinstance(node, ast.Raise):
        return f"{where}: raise the displayed exception, stopping normal flow because continuing would produce misleading or unusable results. A surrounding matching except may handle it."
    if isinstance(node, ast.Assert):
        return f"{where}: require {code(short(node.test, 85))} to be true or raise AssertionError. Assertions document checks, but Python can disable plain assert statements in optimized mode; explicit ValueError checks are preferable for essential input validation."
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return f"{where}: this string is documentation when placed first in the body. It describes behavior but does not itself perform the stated calculation."
        if isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            label = name if len(name) <= 55 else "." + name.rsplit(".", 1)[-1]
            return f"{where}: call {code(label)} with the displayed arguments. No assignment stores its return here; its documented side effect or assertion is what matters."
        return f"{where}: evaluate this expression without assigning its returned value."
    if isinstance(node, ast.FunctionDef):
        return f"{where}: define nested callable {code(node.name)}. Its body runs when it is called, and it can read enclosing local names. It has its own dedicated walkthrough below."
    if isinstance(node, ast.ClassDef):
        return f"{where}: define class {code(node.name)} and its methods. TestCase inheritance provides the self.assert... checking methods."
    if isinstance(node, ast.Continue):
        return f"{where}: skip the rest of this iteration and continue the nearest enclosing loop."
    if isinstance(node, ast.Break):
        return f"{where}: exit the nearest enclosing loop immediately, rather than merely skipping one iteration."
    if isinstance(node, ast.Pass):
        return f"{where}: pass is a placeholder that performs no operation."
    return f"{where}: {type(node).__name__} statement; read its literal source and the enclosing function's contract together."


def nested_statements(node):
    """Do not descend into functions/classes: each gets an independent section."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return
    for field in ("body", "orelse", "handlers", "finalbody"):
        for child in getattr(node, field, []):
            if isinstance(child, (ast.stmt, ast.ExceptHandler)):
                yield from nested_statements(child)
