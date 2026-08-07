"""
run_pipeline.py
================
Runs the full reproducible pipeline end-to-end, in order. This is the
single entry point graders can use to regenerate every figure, metric,
and forecast file in outputs/.

NOTE on runtime: the SARIMAX grid search (Part 4) and rolling-origin
backtest are the slow steps (~30-40 min total on 1 CPU core, since the
brief requires an exhaustive AIC search over p=[0,6], d=[0,2], q=[0,6]
= 147 SARIMAX fits, plus 14 more refits for the backtest). Every
expensive step checkpoints its progress to outputs/metrics/ so a
re-run resumes instead of restarting from scratch.

The foundation model (Part 7, Chronos) is NOT run by this script: this
project's environment has no internet access to Hugging Face. Run
src/foundation_model.py separately in Google Colab (see the docstring
in that file for exact steps), then drop the resulting
outputs/metrics/chronos_forecasts.parquet into this project before
re-running Part 8 (compare_models.py) to include it in the comparison.

Usage:
    python src/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent

STEPS = [
    ("Part 1: data preparation", "data_prep.py"),
    ("Part 1: EDA & stationarity tests", "eda.py"),
    ("Part 2: (problem definition - see evaluation.py docstring, no output)", None),
    ("Part 3: benchmark models", "benchmarks.py"),
    ("Part 4: SARIMAX (slow - grid search + rolling backtest)", "sarimax.py"),
    ("Part 5: feature engineering (smoke test - see features.py)", "features.py"),
    ("Part 6: feature-based ML model (XGBoost)", "ml_model.py"),
    ("Part 8: model comparison & evaluation", "compare_models.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        if script is None:
            continue
        result = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC)
        if result.returncode != 0:
            print(f"FAILED at {script} (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)
    print("\nPipeline complete. See outputs/figures and outputs/metrics.")
    print("Remember: run src/foundation_model.py separately in Colab for Part 7, "
          "then re-run compare_models.py to include Chronos in the final comparison.")


if __name__ == "__main__":
    main()
