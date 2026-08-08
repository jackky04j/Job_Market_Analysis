"""
Master script to run all forecast accuracy tests.
Run from the repo root:

    python forecasting/run_forecast_tests.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import subprocess

from config import DATA_DIR, GRAPH_DIR, OUTPUT_DIR, REPO_ROOT


def run_script(rel_path: str, description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / rel_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            print("SUCCESS")
            if result.stdout:
                print(result.stdout)
        else:
            print("FAILED")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
    except subprocess.TimeoutExpired:
        print("TIMEOUT - Script took too long to run")
    except Exception as exc:
        print(f"ERROR: {exc}")


def main() -> None:
    print("FORECAST ACCURACY TESTING SUITE")
    print("=" * 60)

    required = [
        DATA_DIR / "linkedin_no_skills_cleaned.csv",
        REPO_ROOT / "forecasting" / "simple_forecast_test.py",
        REPO_ROOT / "forecasting" / "test_forecast_accuracy.py",
        REPO_ROOT / "forecasting" / "advanced_forecast_testing.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"\nMissing required files: {missing}")
        print("Place LinkedIn cleaned data in dat_dir/ (see dat_dir/README.md).")
        return

    print("\nAll required files found")

    run_script(
        "forecasting/simple_forecast_test.py",
        "Simple Forecast Test (Best for Small Datasets)",
    )
    run_script(
        "forecasting/test_forecast_accuracy.py",
        "Basic Forecast Accuracy Test (ARIMA Cross-Validation)",
    )
    run_script(
        "forecasting/advanced_forecast_testing.py",
        "Advanced Forecast Testing (Multiple Models & Metrics)",
    )

    print(f"\n{'=' * 60}")
    print("GENERATED FILES CHECK")
    print(f"{'=' * 60}")

    expected = [
        OUTPUT_DIR / "simple_forecast_results.csv",
        OUTPUT_DIR / "forecast_accuracy_results.csv",
        OUTPUT_DIR / "forecast_performance_summary.csv",
        OUTPUT_DIR / "advanced_forecast_results.csv",
        OUTPUT_DIR / "model_comparison_summary.csv",
        GRAPH_DIR / "simple_forecast_results.png",
        GRAPH_DIR / "forecast_accuracy_analysis.png",
        GRAPH_DIR / "prediction_vs_actual.png",
        GRAPH_DIR / "model_comparison.png",
    ]

    for path in expected:
        print(f"{'OK' if path.exists() else 'MISSING'}  {path.relative_to(REPO_ROOT)}")

    print(f"\n{'=' * 60}")
    print("TESTING COMPLETE")
    print(f"{'=' * 60}")
    print("CSVs -> output_dir/   |   graphs -> graph_dir/")


if __name__ == "__main__":
    main()
