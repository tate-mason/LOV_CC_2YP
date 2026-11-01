"""Build the collected current-guide PDF, optionally individual chapter PDFs.

Run: python3 final_draft/build_code_guides.py
Requires pandoc and xelatex on PATH. No private data are read.
"""
from pathlib import Path
import argparse
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "code_guides_pdf"
CHAPTERS = [
    (ROOT / "CODE_GUIDES.md", "00_overview.pdf"),
    (ROOT / "data_code/CODE_GUIDE.md", "01_data_code_guide.pdf"),
    (ROOT / "est_code/CODE_GUIDE.md", "02_estimation_code_guide.pdf"),
    (ROOT / "supply_code/CODE_GUIDE.md", "03_supply_code_guide.pdf"),
]

HEADER = r"""\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\small}
\fvset{breaklines=true,breakanywhere=true}
\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Code guides}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
"""

DIAGRAM = """```text
Panel TSVs -> conversion/concatenation/merge -> attach week_end --+
                                                              |
Retail TSVs -> conversion/concatenation/merge ------------------+
                                                              v
                                          master_dat.py: matched purchases
                                                              |
                                          summ_stats.py: prepared data
                                                              |
                                          est.py: likelihood and inference

Fixed type + calibration -> supply backward induction -> derivatives/figures
```"""


def edition_text(source, combined=False):
    text = source.read_text()
    text = re.sub(r"```mermaid\n.*?```", DIAGRAM, text, flags=re.S)
    text = text.replace("(code_guides_pdf/", "(")
    # PDF editions link to the other standalone PDFs in the same directory.
    targets = {
        "data_code/CODE_GUIDE.md": "01_data_code_guide.pdf",
        "est_code/CODE_GUIDE.md": "02_estimation_code_guide.pdf",
        "supply_code/CODE_GUIDE.md": "03_supply_code_guide.pdf",
    }
    if combined:
        targets = {
            "data_code/CODE_GUIDE.md": "#data-construction-and-descriptive-statistics",
            "est_code/CODE_GUIDE.md": "#demand-estimation-from-purchase-histories-to-a-simulated-likelihood",
            "supply_code/CODE_GUIDE.md": "#dynamic-pricing-a-two-period-symmetric-switching-model",
        }
    for target, pdf in targets.items():
        text = text.replace(f"({target})", f"({pdf})")
        text = text.replace(f"(../{target})", f"({pdf})")
    # The chapters already have hand-written reading outlines; PDF bookmarks
    # and a generated contents page provide page-aware navigation as well.
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual", action="store_true", help="Also build individual chapter PDFs")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="code-guides-") as directory:
        work = Path(directory)
        header = work / "header.tex"
        header.write_text(HEADER)
        common = [
            "pandoc", "--from=markdown+tex_math_single_backslash",
            "--pdf-engine=xelatex", "--standalone", "--toc", "--toc-depth=2",
            "--syntax-highlighting=tango", "--include-in-header", str(header),
            "-V", "geometry:margin=0.8in", "-V", "fontsize=10pt",
            "-V", "documentclass=report", "-V", "colorlinks=true",
            "-V", "date=November 1, 2026",
        ]
        parts = []
        for index, (source, filename) in enumerate(CHAPTERS):
            text = edition_text(source)
            parts.append(edition_text(source, combined=True))
            md = work / f"chapter{index}.md"
            md.write_text(text)
            if args.individual:
                subprocess.run(common + [str(md), "-o", str(OUT / filename)], check=True)
                print(f"Built {OUT / filename}", flush=True)
        combined = work / "complete.md"
        combined.write_text("\n\n\\clearpage\n\n".join(parts))
        subprocess.run(common + [str(combined), "-o", str(OUT / "complete_code_guides.pdf")], check=True)
        print(f"Built {OUT / 'complete_code_guides.pdf'}", flush=True)


if __name__ == "__main__":
    main()
