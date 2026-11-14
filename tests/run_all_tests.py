
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_FILES = [
    "smoke_test_api.py",
    "test_explain_analytics.py",
]

print("\n======================================")
print(" SOFTWARE DEFECT PREDICTOR - TEST SUITE")
print("======================================\n")

failed_tests = []

for test_file in TEST_FILES:
    test_path = ROOT_DIR / "tests" / test_file

    print(f"\nRunning: {test_file}")
    print("-" * 45, flush=True)

    result = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=ROOT_DIR
    )

    if result.returncode != 0:
        failed_tests.append(test_file)

print("\n======================================")
print(" TEST SUMMARY")
print("======================================")

if failed_tests:
    print("FAILED TESTS:")
    for test in failed_tests:
        print(f" - {test}")
    sys.exit(1)

print("ALL TESTS PASSED SUCCESSFULLY!")
print(f"Total test files: {len(TEST_FILES)}")
