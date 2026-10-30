"""Reproducibility entry point: tests, requested figures, symbolic LaTeX."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for command in [
    ['-m','unittest','-v','checks.test_symmetric'],
    ['make_comparative_static_figures.py'],
    ['make_latex_document.py'],
]:
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
print('All checks passed; figures and LaTeX regenerated. Compile the TeX with latexmk.')
