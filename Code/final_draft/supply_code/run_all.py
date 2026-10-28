import subprocess
import sys

checks = [
    "checks.check_focs",
    "checks.check_gamma",
    "checks.check_beta",
    "checks.check_alpha",
    "checks.check_delta",
    "checks.check_pricing_derivatives",
    "checks.check_numerical",
]

for check in checks:
    print(f"\nRunning {check}", flush=True)

    subprocess.run([sys.executable, "-m", check], check=True)

print("\nAll checks passed!")

print("\nDone!")
