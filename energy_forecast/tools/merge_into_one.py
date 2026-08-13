"""Merges data_prep.py, evaluation.py, eda.py, features.py, benchmarks.py,
sarimax.py, ml_model.py, foundation_model.py, compare_models.py into a
single all_in_one.py, in dependency order, stripping internal cross-file
imports (everything ends up in one namespace) and renaming each module's
`main()` into a unique `run_partN_*()` function, called in order from one
unified `main()` at the bottom."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent

FILES_IN_ORDER = [
    ("data_prep.py", "run_part1_data_prep"),
    ("evaluation.py", None),  # no main() in this one
    ("eda.py", "run_part1_eda"),
    ("features.py", None),
    ("benchmarks.py", "run_part3_benchmarks"),
    ("sarimax.py", "run_part4_sarimax"),
    ("ml_model.py", "run_part6_ml_model"),
    ("foundation_model.py", "run_part7_foundation_model"),
    ("compare_models.py", "run_part8_compare_models"),
]

INTERNAL_MODULE_NAMES = {
    "data_prep", "evaluation", "eda", "features", "benchmarks",
    "sarimax", "ml_model", "foundation_model", "compare_models",
}

HEADER = '''"""
all_in_one.py
=============
The ENTIRE project pipeline (Parts 1, 2, 3, 4, 5, 6, 7, 8) in a single file,
merged from the original modular scripts (data_prep.py, evaluation.py,
eda.py, features.py, benchmarks.py, sarimax.py, ml_model.py,
foundation_model.py, compare_models.py) for anyone who wants one file to
run instead of several.

HOW TO RUN
----------
    python all_in_one.py

This runs Parts 1, 3, 4, 5, 6, 8 in order (skipping Part 7/Chronos by
default, since it needs internet access to Hugging Face - see below).
Figures and metrics are written to ../outputs/, exactly as with the
original modular scripts. Part 4 (SARIMAX) is the slow step (~30-40 min
on 1 CPU core, since the assignment requires an exhaustive AIC grid
search); it checkpoints progress to ../outputs/metrics/ so a re-run
resumes instead of restarting.

To ALSO run Part 7 (Chronos foundation model) - only possible in an
environment with internet access to huggingface.co, e.g. Google Colab:
    python all_in_one.py --with-chronos

Every function below is still organised by the part of the assignment it
belongs to (see the "===== PART N =====" banners), and is unchanged from
the original modular files other than removing the `from X import Y`
lines between them, since everything now lives in one shared namespace.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

'''

FOOTER_TEMPLATE = '''

# ============================================================
# UNIFIED PIPELINE ENTRY POINT
# ============================================================
def main(with_chronos: bool = False):
    steps = [
        ("PART 1: Data preparation", run_part1_data_prep),
        ("PART 1: EDA & stationarity tests", run_part1_eda),
        ("PART 3: Benchmark models", run_part3_benchmarks),
        ("PART 4: SARIMAX (slow - grid search + rolling backtest)", run_part4_sarimax),
        ("PART 6: Feature-based ML model (XGBoost)", run_part6_ml_model),
    ]
    if with_chronos:
        steps.append(("PART 7: Foundation model (Chronos)", run_part7_foundation_model))
    steps.append(("PART 8: Model comparison & evaluation", run_part8_compare_models))

    for label, fn in steps:
        print(f"\\n{'=' * 70}\\n{label}\\n{'=' * 70}")
        fn()

    print("\\nPipeline complete. See ../outputs/figures and ../outputs/metrics.")
    if not with_chronos:
        print("NOTE: Part 7 (Chronos) was skipped (needs internet access to Hugging Face). "
              "Run with --with-chronos in an internet-connected environment (e.g. Colab) to include it.")


if __name__ == "__main__":
    main(with_chronos="--with-chronos" in sys.argv)
'''


def process_file(filename: str, main_rename: str | None) -> str:
    text = (SRC / filename).read_text()

    # Drop the module docstring (its content is folded into the section
    # banner comment instead, to avoid triple-quote collisions when files
    # are concatenated).
    text = re.sub(r'^""".*?"""\n', "", text, count=1, flags=re.DOTALL)

    lines = text.split("\n")
    out_lines = []
    skip_continuation = False
    for line in lines:
        stripped = line.strip()
        if skip_continuation:
            # keep dropping lines until the closing paren of a multi-line import
            if stripped.endswith(")"):
                skip_continuation = False
            continue
        # Drop internal cross-module imports (everything now shares one namespace)
        if stripped.startswith("from ") and stripped.split()[1] in INTERNAL_MODULE_NAMES:
            if "(" in stripped and not stripped.endswith(")"):
                skip_continuation = True
            continue
        # Drop this file's own __main__ guard (folded into the unified main() instead)
        if stripped.startswith('if __name__ == "__main__"'):
            break
        out_lines.append(line)
    text = "\n".join(out_lines).rstrip() + "\n"

    if main_rename:
        text = re.sub(r"\bdef main\(\)", f"def {main_rename}()", text, count=1)

    banner = f"\n\n# {'=' * 60}\n# SOURCE: {filename}\n# {'=' * 60}\n"
    return banner + text


def main():
    parts = [HEADER]
    for filename, main_rename in FILES_IN_ORDER:
        parts.append(process_file(filename, main_rename))
    parts.append(FOOTER_TEMPLATE)

    out_path = SRC / "all_in_one.py"
    out_path.write_text("".join(parts))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
