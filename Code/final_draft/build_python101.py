"""Build one source-linked Python 101 walkthrough PDF without executing research code.

python3 final_draft/build_python101.py
python3 final_draft/build_python101.py --markdown-only

The editable lessons live in python101/. Source snippets are extracted using
Python's AST, checked against a coverage manifest, and combined with those lessons.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python101"))
from explain_syntax import describe, operations, nested_statements
from build_code_guides import HEADER


def load_notes(path):
    parts = re.split(r"^## (.+)$", path.read_text(), flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


TESTS = {
    "test_history_excludes_current_counts_and_accumulates_all_occasions": "Checks pre-choice timing: changing a current basket must not rewrite the stock already available for that choice. It compares explicit stock vectors, not just output dimensions.",
    "test_xi_sign_and_unlimited_history": "Checks numerical Xi signs and persistence across long gaps. The explicit -.7/.3 expectations distinguish own-stock penalties from recent-other rewards.",
    "test_parquet_pipeline_and_fit": "Builds a tiny real Parquet input and runs preparation plus a deliberately short fit. It checks all-store counts, exposure-store menus, missing alternatives, exported files, and own-DMA exclusion. Its low iteration limit is a smoke check, not proof of statistical convergence.",
    "test_unavailable_flavor_is_excluded_from_logit_denominator": "Removes plain prices and independently computes the remaining two-flavor likelihood. It also requires a ValueError if an unavailable flavor is actually purchased. This catches treating a missing-price placeholder as an available free good.",
    "test_persistent_household_integration": "Independently computes a draw's full household history before averaging draws. This catches a subtle wrong model that integrates each week separately.",
    "test_analytic_gradient": "Perturbs each coefficient and compares centered finite-difference objective slopes with the supplied analytic gradient. This checks derivative values rather than just whether optimization returns a result.",
    "test_four_model_specifications": "Checks parameter layouts/likelihood behavior for the four supported specifications. The labels define vector meaning; silently changed ordering would corrupt estimates even when vector lengths still match.",
    "test_mixed_naive_integrates_only_alpha": "Checks that mixed-naive choice probabilities vary with alpha draws while excluding a gamma effect. It independently constructs the corresponding integrated likelihood.",
    "test_zero_heterogeneity_nests_homogeneous_model": "Sets random-coefficient SDs to zero and compares against the common-coefficient model. A nesting identity is a strong check on parameter transforms and integration.",
    "test_excludes_consumer_dma_even_when_exposure_store_elsewhere": "Exercises the retained UPC-matched IV helper with explicit tiny prices. It excludes the consumer DMA rather than assuming exposure store and consumer are in the same market, and checks incomplete baskets become missing.",
    "test_absorption_matches_explicit_dummy_ols": "Compares repeated demeaning to a small regression on explicit week/DMA dummy columns. The dummy matrix is feasible in a fixture and supplies a transparent independent alternative to the production memory-saving algorithm.",
    "test_first_stage_residuals_orthogonal_and_in_utility": "Checks positive first-stage relevance in controlled data, residual orthogonality, zero group means, and that the demand arrays actually contain the computed controls.",
    "test_first_stage_uses_only_covered_flavor_prices": "Moves purchases away from unavailable alternatives, removes their prices, and checks each first stage uses only its covered rows. It also verifies zero residual placeholders remain finite and masked.",
    "test_resampling_preserves_histories_and_relabels_duplicates": "Resamples a 60-household/eight-period fixture and verifies there are 60 newly labeled histories, each with eight rows and no repeated new-household/week key. Sampling whole groups is compared with the behavior a row bootstrap would break.",
    "test_bootstrap_refits_first_stage_and_reports_empirical_sd": "Wraps the first-stage callable with a mock to count 20 refits, supplies a cheap residual-dependent fake estimator, and compares output SEs with SDs calculated from saved bootstrap draws. This isolates two-stage orchestration from optimization.",
    "test_controlled_demand_optimizer_runs": "Fits a small controlled naive demand model and checks convergence, a four-coefficient vector, and finite objective. This is a controlled optimizer smoke test, not an assertion about private-data fit quality.",
    "test_demographics_and_mixed_baskets": "Uses tiny tables to check household demographic selection, missing categories, and mixed-basket summary behavior. Explicit source values make it possible to verify denominators by hand.",
    "test_batches_preserve_global_statistics": "Runs a small complete summary pipeline with one and three partitions, sorts outputs, and compares their actual contents. It verifies partitioning changes execution/storage rather than statistical results and preserves original metadata in the prepared sample.",
    "test_demographics_match_previous_fieldwise_sort": "Compares the optimized shared demographic-history sort to separate field-by-field selection on a synthetic example. This checks that a memory/runtime improvement preserves earliest-nonmissing semantics.",
    "test_recovery_preserves_existing_labels_and_uses_earliest_year": "Constructs reference and annual metadata cases to check that existing labels are retained and absent UPCs take earliest usable metadata. It verifies a substantive recovery precedence rule.",
    "test_raw_normalization_and_both_states": "Evaluates both signed histories and symbolically verifies subtracting raw outside utility yields beta+alpha*price+q*gamma. This catches accidentally deleting outside switching utility.",
    "test_focs_from_actual_profit": "Differentiates literal symbolic profits and compares them to stored FOCs, including both continuation weights. It checks the objective behind the equations rather than merely rearranging a copied FOC.",
    "test_envelope_and_implicit_gamma_derivatives": "Verifies explicit gamma price/profit/curvature identities and total derivatives of the implicit FOC. Algebraic comparison catches missing chain-rule and continuation terms.",
    "test_all_parameter_implicit_derivatives": "Loops through alpha, beta, delta, and cost and verifies that differentiating the equilibrium pricing identity along each solution gives zero. This extends checks beyond the headline gamma plots.",
    "test_reference_on_entire_benchmark_grid": "Solves both histories at all 601 gamma grid points and compares named numerical outputs against the independently coded reference. It also checks original FOC residual magnitudes.",
    "test_independent_reoptimization_and_derivatives": "At several calibrations compares analytic slopes and curvature with finite differences of fully reoptimized solutions, ratio derivatives with recomputed ratios, and p1 with a separate direct profit optimizer. Different computational routes can catch shared-expression mistakes.",
    "test_all_parameter_and_mixed_derivatives_numerically": "Lambdifies general parameter derivatives, perturbs alpha/beta/delta/c separately, and compares finite differences. It also checks a beta-gamma mixed derivative using reoptimized results.",
    "test_zero_switching_static_case_and_symmetry": "Checks gamma=0 branch equality, delta=0 equality with the true static counterfactual, terminal history/sign symmetry, and an admissible negative introductory price. These limit cases catch state and price-floor regressions.",
}


def function_inventory(tree):
    found = []

    def visit(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*parents, child.name))
                found.append((qualified, child))
                visit(child, (*parents, child.name))
            elif isinstance(child, ast.ClassDef):
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)
    visit(tree)
    return sorted(found, key=lambda x: x[1].lineno)


def test_note(name):
    last = name.split(".")[-1]
    if last not in TESTS:
        return None
    return ("**What this test establishes.** " + TESTS[last] +
            "\n\n**Reading its syntax.** This method is called by unittest on a TestCase instance. "
            "The first parameter self provides the assertion methods. Setup assignments create known inputs, "
            "the helper call produces actual results, and assertion lines compare them with expectations. "
            "The method normally returns None; a failed assertion raises and marks the test as failed. "
            "The following line notes spell out its particular operations and comparisons. "
            "Printing results instead would require manual inspection and would not automatically fail a regression check.")


def excerpt(lines, start, end):
    output = []
    # Long raw document strings are still reproduced completely, in manageable
    # continued listings. Textwrap removes only common indentation for reading.
    for first in range(start, end + 1, 42):
        last = min(first + 41, end)
        text = textwrap.dedent("\n".join(lines[first - 1:last]))
        output.append(f"Source lines {first}–{last}:\n\n"
                      f"```{{.python .numberLines startFrom=\"{first}\"}}\n{text}\n```\n")
    return "\n".join(output)


def statements_for_module(tree):
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.ClassDef):
            # The class header appears in its own short listing; methods are
            # handled individually. Keep any nonmethod class-level statements.
            out.append((node, True))
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((child, False))
        else:
            out.append((node, False))
    return out


def block(node, lines, seen_operations, covered, header_only=False):
    end = node.lineno if header_only else node.end_lineno
    result = [excerpt(lines, node.lineno, end), "**Read this block:**\n"]
    descendants = [node] if header_only else list(nested_statements(node))
    for item in descendants:
        covered.add(item.lineno)
        result.append("- " + describe(item))
    # Avoid explaining methods hidden in nested callable bodies at definition time.
    ops = [] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else operations(node)
    fresh = [note for note in ops if note not in seen_operations]
    seen_operations.update(fresh)
    if fresh:
        result += ["\n**Syntax and calls used here:**\n", *["- " + note for note in fresh]]
    return "\n".join(result) + "\n\n"


def grouped_blocks(nodes, lines, seen_operations, covered):
    """Keep short adjacent statements together while annotating every statement.

    This avoids forcing the reader through a separate layout heading for each
    import or print. Control-flow blocks stay intact and retain nested notes.
    """
    groups = []
    pending = []
    for node in nodes:
        simple = isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.Expr))
        if pending and (not simple or node.end_lineno - pending[0].lineno >= 15 or len(pending) >= 6):
            groups.append(pending)
            pending = []
        if simple and node.end_lineno - node.lineno < 15:
            pending.append(node)
        else:
            groups.append([node])
    if pending:
        groups.append(pending)
    results = []
    for group in groups:
        if len(group) == 1:
            results.append(block(group[0], lines, seen_operations, covered))
            continue
        result = [excerpt(lines, group[0].lineno, group[-1].end_lineno), "**Read this block:**\n"]
        fresh = []
        for node in group:
            for item in nested_statements(node):
                covered.add(item.lineno)
                result.append("- " + describe(item))
            for note in operations(node):
                if note not in seen_operations:
                    fresh.append(note)
                    seen_operations.add(note)
        if fresh:
            result += ["\n**Syntax and calls used here:**\n", *["- " + note for note in fresh]]
        results.append("\n".join(result) + "\n\n")
    return results


def file_intro(rel, notes, tree):
    if rel in notes:
        return notes[rel]
    if rel.endswith("/__init__.py"):
        return ("This empty file marks an importable package directory. It defines no functions "
                "and performs no data calculations. Python can support some namespace packages "
                "without it, but an explicit marker makes the intended package structure clear.")
    if not tree.body:
        return "This file is empty in the source snapshot: it contains no function or executable data-processing block to explain."
    if "/comparative_statics/" in rel:
        return ("This small module gives an existing canonical derivative a convenient public name, "
                "or combines canonical derivatives into the displayed expression. It does not fit a model. "
                "The literal imports and assignment below show whether it calls a generic helper or aliases "
                "an already derived expression. Reusing the canonical implementation avoids maintaining "
                "separate competing versions of the same comparative static.")
    if "/checks/check_" in rel:
        return ("This is a compatibility entry point for the canonical unittest checks. Importing a "
                "TestCase class makes its methods discoverable to unittest; the guarded unittest.main() "
                "runs the selected checks when invoked directly. It does not define another derivative model.")
    return ("Module-level imports and constants set up the functions below. Definitions are registered "
            "when imported, while function bodies wait until called. The __main__ guard, if present, "
            "selects direct execution. Each function has its own input/output and syntax discussion below.")


def generate():
    function_notes = load_notes(ROOT / "python101/FUNCTION_NOTES.md")
    file_notes = load_notes(ROOT / "python101/FILE_NOTES.md")
    files = []
    # Put production workflow first and tests in an appendix.
    for folder in ("data_code", "est_code", "supply_code", "pricing_exercises"):
        paths = sorted((ROOT / folder).rglob("*.py"))
        files.extend(p for p in paths if "__pycache__" not in p.parts)
    # Include the readable current demand workflow alongside its original sources.
    simple = ROOT.parent / "final_simple"
    files.extend(simple / "demand" / name for name in
                 ("estimation.py", "split_pricing.py", "plain_berry_pricing.py", "regenerate_weekly_inputs.py", "history_distributions.py"))
    files.extend(simple / "checks" / name for name in
                 ("test_split_pricing.py", "test_regenerate_weekly_inputs.py"))
    is_test = lambda p: p.name.startswith("test_") or "checks" in p.parts
    ordered = [p for p in files if not is_test(p)] + [p for p in files if is_test(p)]
    chunks = [(ROOT / "python101/START_HERE.md").read_text(),
              "\\clearpage\n\n" + (ROOT / "python101/WORKED_LESSONS.md").read_text()]
    records = []
    missing = []
    last_group = None
    for path in ordered:
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else "../" + path.relative_to(ROOT.parent).as_posix()
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source, filename=rel)
        group = "Tests and verification: worked calling examples" if is_test(path) else ("Current readable demand workflow (final_simple)" if rel.startswith("../final_simple/") else rel.split("/")[0] + ": source walkthrough")
        if group != last_group:
            chunks.append("\n\\clearpage\n\n# " + group + "\n")
            last_group = group
        chunks.append("\n## " + rel.replace("_", "\\_") + "\n")
        chunks.append(file_intro(rel, file_notes, tree) + "\n")
        functions = function_inventory(tree)
        covered = set()
        chunks.append("\n### Module setup and direct-execution blocks\n")
        if not tree.body:
            chunks.append("There are no statements in this file.\n")
        seen = set()
        pending = []
        for node, header_only in statements_for_module(tree):
            if header_only:
                chunks.extend(grouped_blocks(pending, lines, seen, covered))
                pending = []
                chunks.append(block(node, lines, seen, covered, True))
            else:
                # Definitions skipped in module setup must not accidentally
                # reappear in a group spanning their source-line gap.
                if pending and node.lineno - pending[-1].end_lineno > 3:
                    chunks.extend(grouped_blocks(pending, lines, seen, covered))
                    pending = []
                pending.append(node)
        chunks.extend(grouped_blocks(pending, lines, seen, covered))
        fnrecords = []
        for name, fn in functions:
            key = rel + "::" + name
            note = function_notes.get(key) or test_note(name)
            if note is None:
                missing.append(key)
                note = "MISSING EDITORIAL NOTE"
            chunks.append("\n### " + name.replace("_", " ") + "\n")
            chunks.append(f"Original function: `{name}`. Source begins at line {fn.lineno}.\n\n")
            chunks.append(note + "\n\n")
            first_body = fn.body[0].lineno
            header_end = first_body - 1
            has_doc = isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str)
            if has_doc:
                header_end = fn.body[0].end_lineno
                covered.add(fn.body[0].lineno)
            # Definitions with a one-line body are not used in the snapshot.
            header_end = max(fn.lineno, header_end)
            chunks.append("**Definition and stated intent:**\n\n" + excerpt(lines, fn.lineno, header_end))
            args = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
            chunks.append("`def` creates this callable; the parentheses name its inputs (" +
                          ", ".join("`" + a + "`" for a in args) +
                          "). A displayed `=value` supplies a default, and the colon begins its indented body.\n\n")
            if fn.args.kwarg:
                chunks.append(f"`**{fn.args.kwarg.arg}` collects additional keyword arguments in a dictionary.\n\n")
            covered.add(fn.lineno)
            seen = set()
            chunks.extend(grouped_blocks(fn.body[1:] if has_doc else fn.body, lines, seen, covered))
            if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
                chunks.append("**Output reminder:** reaching the end returns `None`. The useful result here is assertions, printing, plotting, writing, or another documented side effect.\n")
            fnrecords.append(dict(name=name, line=fn.lineno, end_line=fn.end_lineno,
                                  editorial_note=key if key in function_notes else "test lesson: " + fn.name))
        expected = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt)}
        unaccounted = expected - covered
        if unaccounted:
            raise RuntimeError(f"Unexplained statements in {rel}: {sorted(unaccounted)}")
        records.append(dict(file=rel, sha256=hashlib.sha256(source.encode()).hexdigest(),
                            source_lines=len(lines), functions=fnrecords,
                            statement_start_lines=sorted(expected), covered_statement_start_lines=sorted(covered)))
    if missing:
        raise RuntimeError("Missing function teaching notes:\n" + "\n".join(missing))
    count = sum(len(r["functions"]) for r in records)
    chunks.append("\n\\clearpage\n\n# Coverage, rebuilding, and next steps\n")
    chunks.append(f"This edition includes {len(records)} Python files, {count} named functions/methods (including nested helpers), "
                  f"and {sum(r['source_lines'] for r in records):,} source lines. "
                  "Every AST statement start is accounted for in the annotated listings. Empty files and top-level scripts are included. "
                  "Lambda expressions are explained at their use sites rather than counted as named functions. "
                  "Large raw LaTeX strings are reproduced as data; their mathematical content is discussed in the separate economic guide.\n\n")
    chunks.append("The build reads Python files as text and parses their syntax; it never imports or executes the private-data processing scripts. "
                  "The companion JSON manifest records source hashes, function names/lines, and statement coverage. "
                  "If source changes, rebuild the book so printed line references match the new snapshot. "
                  "A newly added function requires a teaching note before the build succeeds.\n\n")
    chunks.append("Rebuild from the repository's Code directory:\n\n```bash\npython3 final_draft/build_python101.py\n```\n\n"
                  "Use `--markdown-only` to regenerate the readable source/coverage manifest without invoking Pandoc/XeLaTeX. "
                  "The source lessons are in `final_draft/python101/`. The generated Markdown is `final_draft/PYTHON101_WALKTHROUGH.md`.\n\n"
                  "Keep this book beside the separate `complete_code_guides.pdf`: use that guide for model derivations and this one "
                  "for what the corresponding Python actually says. Try changing tiny test fixtures or tracing values on paper before editing the large server pipelines.\n")
    book = ROOT / "PYTHON101_WALKTHROUGH.md"
    book.write_text("\n\n".join(chunks))
    manifest = ROOT / "python101/coverage_manifest.json"
    manifest.write_text(json.dumps(dict(files=len(records), named_functions=count, sources=records), indent=2) + "\n")
    return book, len(records), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-only", action="store_true")
    args = parser.parse_args()
    book, files, functions = generate()
    print(f"Wrote {book}: {files} files, {functions} named functions/methods", flush=True)
    if args.markdown_only:
        return
    out = ROOT / "code_guides_pdf/python101_walkthrough.pdf"
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="python101-") as directory:
        header = Path(directory) / "header.tex"
        header.write_text(HEADER.replace("Code guides", "Python 101 code walkthrough") +
                          "\n\\usepackage{xurl,seqsplit}\n\\fvset{numbersep=3pt}\n")
        pdf_source = Path(directory) / "edition.md"
        # FE0F only selects emoji presentation in two legacy log strings;
        # omit that invisible selector in typesetting, preserving the source
        # verbatim in the generated Markdown and manifest.
        pdf_source.write_text(book.read_text().replace("\ufe0f", ""))
        command = ["pandoc", str(pdf_source), "--from=markdown+tex_math_single_backslash",
                   "--pdf-engine=xelatex", "--standalone", "--toc", "--toc-depth=3",
                   "--syntax-highlighting=tango", "--include-in-header", str(header),
                   "--lua-filter", str(ROOT / "python101/wrap_inline_code.lua"),
                   "-V", "geometry:margin=0.85in", "-V", "fontsize=10pt",
                   "-V", "monofont=Menlo",
                   "-V", "documentclass=report", "-V", "colorlinks=true",
                   "-M", "title=Python 101: A Function-by-Function Code Walkthrough",
                   "-M", "subtitle=Data preparation, demand estimation, and dynamic pricing",
                   "-M", "date=November 6, 2026", "-o", str(out)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = ROOT / "python101/pdf_build.log"
        log.write_text(result.stdout)
        if result.returncode:
            print(result.stdout[-8000:])
            result.check_returncode()
    print(f"Built {out}", flush=True)


if __name__ == "__main__":
    main()
