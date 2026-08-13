"""
Builds notebooks/analysis_standalone.ipynb: a self-contained notebook that
contains the actual analysis code directly in its cells (functions defined
and called inline), rather than importing from src/all_in_one.py. Fast
parts (data prep, EDA, benchmarks) run live; slow parts (SARIMAX grid
search + rolling backtest, XGBoost rolling backtest) load their
already-computed, checkpointed results from ../outputs/metrics/ so the
whole notebook still executes end-to-end in well under a minute.

Warnings (e.g. the KPSS "p-value outside lookup table" InterpolationWarning)
are suppressed globally in the first code cell.
"""
import re
import nbformat as nbf
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "all_in_one.py"
text = SRC.read_text()

# Split the merged file back into its named sections using the banner comments.
sections = {}
matches = list(re.finditer(r"# SOURCE: (\S+)\n# =+\n", text))
for i, m in enumerate(matches):
    name = m.group(1)
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    section_text = text[start:end].strip("\n")
    # __file__ doesn't exist inside a notebook cell (only in real .py files);
    # replace the file-relative path base with a notebook-relative one
    # (this notebook lives in notebooks/, one level below the project root,
    # same depth as src/ was, so the relative path is identical).
    section_text = section_text.replace(
        "Path(__file__).resolve().parents[1]", 'Path("..")'
    )
    sections[name] = section_text

# The last section (compare_models.py) runs to end-of-file, which also
# swept up all_in_one.py's own appended unified-pipeline footer (main(),
# argv handling, etc.) - strip that back off since it doesn't belong to
# any individual part and isn't meaningful inside a notebook cell.
sections["compare_models.py"] = sections["compare_models.py"].split(
    "# UNIFIED PIPELINE ENTRY POINT"
)[0].rstrip().rstrip("#").rstrip()

nb = nbf.v4.new_notebook()
cells = []

def md(t):
    cells.append(nbf.v4.new_markdown_cell(t))

def code(t):
    cells.append(nbf.v4.new_code_cell(t))

# ---------------------------------------------------------------- Title --
md("""# Appliance Energy Demand Forecasting — Standalone Analysis Notebook

This notebook contains the full analysis **directly in its own cells** (not
imported from an external `all_in_one.py` module) — every function is
defined and run right here. Fast steps (data preparation, EDA, stationarity
tests, benchmark models) execute live; the two slow steps (the exhaustive
SARIMAX AIC grid search + 14 rolling refits, and the 14 XGBoost
rolling-origin fits) load their already-computed, checkpointed results from
`../outputs/metrics/` instead of re-running ~30-40 minutes of computation,
so the whole notebook runs end-to-end in under a minute.

See `../reports/report.docx` for the full written report and discussion of
all 6 required questions.
""")

code("""import warnings
warnings.filterwarnings("ignore")  # suppresses harmless statistical warnings (e.g. KPSS p-value out of lookup-table range)

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from xgboost import XGBRegressor

pd.set_option("display.width", 120)
%matplotlib inline

DATA_DIR = Path("../data")
FIG_DIR = Path("../outputs/figures")
METRICS_DIR = Path("../outputs/metrics")
""")

# ---------------------------------------------------------------- Part 1 --
md("## Part 1 — Data Preparation\n\nLoad the raw 10-minute data, check for missing values, and resample to hourly resolution.")
code(sections["data_prep.py"])
code("""df_raw = load_raw()
print("Raw (10-min) shape:", df_raw.shape)
missing = check_missing(df_raw)

df_hourly = resample_hourly(df_raw)
df_hourly = add_time_features(df_hourly)
print("\\nHourly shape:", df_hourly.shape)
df_hourly.head()
""")

md("## Part 1 — EDA and Stationarity Testing")
code(sections["eda.py"])
code("""print("Summary statistics for Appliances (Wh/hour):")
print(df_hourly["Appliances"].describe())
""")
code("""display(Image("../outputs/figures/01_series_overview.png", width=700))
display(Image("../outputs/figures/03_weekly_hourly_profile.png", width=600))
display(Image("../outputs/figures/02_decomposition_daily.png", width=600))
""")
md("Both ADF and KPSS agree the raw hourly series is already stationary (run live below — fast).")
code("""import warnings
warnings.filterwarnings("ignore")  # re-assert: Jupyter can reset filters between cells

adf_test(df_hourly["Appliances"], "Appliances (raw)")
kpss_test(df_hourly["Appliances"], "Appliances (raw)")
""")
code("""display(Image("../outputs/figures/04_acf_pacf_raw.png", width=750))""")

# ---------------------------------------------------------------- Part 2 --
md("## Part 2 — Forecasting Problem Definition and Evaluation Harness\n\n"
   "- **Target:** `Appliances` (Wh/hour)\n"
   "- **Horizon:** 24 hours ahead\n"
   "- **Evaluation:** rolling-origin backtest over the last 14 days (14 non-overlapping 24h blocks, 336 pooled predictions)\n"
   "- **Metrics:** RMSE, MAE, MAPE, sMAPE")
code(sections["evaluation.py"])
code("""df = load_hourly()
train, test = train_test_split(df)
origins = get_origins(df)
print(f"Full data: {len(df)}h | Train: {len(train)}h | Test: {len(test)}h")
print(f"Rolling-origin blocks: {len(origins)} x {HORIZON}h")
""")

# ---------------------------------------------------------------- Part 3 --
md("## Part 3 — Benchmark Models\n\nMean, Naive, Daily/Weekly Seasonal Naive, Drift. Fast — run live.")
code(sections["benchmarks.py"])
code("""bench_results = run_benchmarks()
bench_summary = summarize(bench_results)
bench_summary.round(2)
""")
code("""display(Image("../outputs/figures/07_benchmark_example_block.png", width=650))""")

# ---------------------------------------------------------------- Part 4 --
md("""## Part 4 — SARIMAX

The full function definitions (grid search, residual diagnostics, rolling
backtest) are below and are exactly what produced the cached results —
but the search itself (147 fits + 14 refits, ~30-40 min on 1 CPU core) is
**not re-run live here**; we load its checkpointed output instead.""")
code(sections["sarimax.py"])
code("""grid = pd.read_csv(METRICS_DIR / "sarimax_grid_search.csv").sort_values("aic")
print("Top 10 orders by AIC (non-seasonal grid, seasonal order fixed at (0,1,1,24)):")
grid.head(10)
""")
code("""sarimax_forecasts = pd.read_parquet(METRICS_DIR / "sarimax_forecasts.parquet")
sarimax_summary = summarize(sarimax_forecasts)
sarimax_summary.round(2)
""")
code("""display(Image("../outputs/figures/09_sarimax_residual_diagnostics.png", width=750))""")
md("Ljung-Box test on the full-training-set model's residuals: p=0.44 at 24 lags, p=0.34 at 48 lags — "
   "residuals are statistically indistinguishable from white noise.")

# ---------------------------------------------------------------- Part 5/6 --
md("## Part 5 — Feature Engineering\n\nFast — run live.")
code(sections["features.py"])
code("""feat_df = build_feature_frame(df)
feature_cols = get_feature_columns()
print(f"Built {len(feature_cols)} features from sensor/weather/time/lag/rolling groups.")
feat_df[feature_cols].tail(3).T.head(15)
""")

md("""## Part 6 — Feature-Based ML Model (XGBoost)

Function definitions below; the 14-origin rolling backtest itself (~2 min)
is loaded from its cached output rather than re-run live.""")
code(sections["ml_model.py"])
code("""xgb_forecasts = pd.read_parquet(METRICS_DIR / "xgboost_forecasts.parquet")
xgb_summary = summarize(xgb_forecasts)
xgb_summary.round(2)
""")
code("""importance = pd.read_csv(METRICS_DIR / "xgboost_feature_importance.csv", index_col=0)
display(Image("../outputs/figures/10_xgb_feature_importance.png", width=550))
""")

md("""### Part 6b — Realistic weather (Section 9, Q5 follow-up)

`build_feature_frame` and `recursive_forecast` both take a `weather_mode`
argument: `"conditional"` (default, used above) takes the true future
weather values, which is optimistic — a real deployment would only have a
weather *forecast*. `"realistic"` instead substitutes each weather column's
value from 24 hours earlier (genuinely available at the forecast origin).
Re-running with `weather_mode="realistic"` directly measures how much the
conditional-forecast assumption is worth (~2 min, run live since it's a
second, smaller backtest).""")
code("""run_part6b_ml_model_realistic_weather()
""")

# ---------------------------------------------------------------- Part 7 --
md("""## Part 7 — Foundation Model (Chronos)

Run this notebook's Part 7 cells in an internet-connected environment (e.g.
Google Colab, free tier, ~2 min) to execute Chronos for real — this sandbox
has no internet access to Hugging Face. Install the extra dependency first:
`!pip install -q chronos-forecasting torch`, then run the cells below, then
copy the resulting `chronos_forecasts.parquet` into `../outputs/metrics/`
to include it in Part 8 automatically.""")
code(sections["foundation_model.py"])
code("""# In Colab (with internet access), uncomment and run the next two lines:
# import torch  # noqa: F401 (imported inside run_chronos_backtest; listed here as a dependency reminder)
# run_part7_foundation_model()

chronos_path = METRICS_DIR / "chronos_forecasts.parquet"
if chronos_path.exists():
    chronos_forecasts = pd.read_parquet(chronos_path)
    print(summarize(chronos_forecasts).round(2))
else:
    print("chronos_forecasts.parquet not found - run this notebook's Part 7 code in Colab first "
          "(see instructions above; this project's real Chronos result was RMSE=428.5, MAE=209.9, "
          "MAPE=23.0%, sMAPE=28.0% - see reports/report.docx Section 8-9 for the full discussion).")
""")

# ---------------------------------------------------------------- Part 8 --
md("## Part 8 — Model Comparison and Evaluation")
code(sections["compare_models.py"])
code("""all_results = load_all_results()
full_summary = comparison_table(all_results)
summary_with_improvement, strongest = compare_to_strongest_benchmark(full_summary)
summary_with_improvement.round(2)
""")
code("""display(Image("../outputs/figures/12_rmse_by_horizon_all_models.png", width=650))
display(Image("../outputs/figures/13_error_diagnostics_SARIMAX.png", width=650))
""")

# ---------------------------------------------------------------- Part 9 --
md("""## Part 9 — Discussion Questions

Full answers (with supporting evidence and figure references) are in
`../reports/report.docx`, Section 10. Summary:

1. **Strongest benchmark:** split result — Mean wins on RMSE, SeasonalNaive_Weekly wins on MAE/MAPE/sMAPE.
2. **SARIMAX vs. benchmarks:** yes on every metric; residuals are white noise (Ljung-Box p=0.44/0.34).
3. **XGBoost feature value:** beats benchmarks but not SARIMAX on RMSE — recursive multi-step error compounding and limited training data are likely causes.
4. **Foundation model:** Chronos, run zero-shot in Colab, had the highest RMSE (428.5) but the lowest MAE/MAPE/sMAPE (209.9/23.0%/28.0%) of every model tested — a genuine worst-case-vs-typical-case split, not a simple win or loss.
5. **Forecast conditionality:** SARIMAX/XGBoost use real future weather (and, for XGBoost, real future indoor sensor readings) — conditional, not true unconditional, forecasts. Tested directly: a 24h-lagged weather proxy gives slightly worse RMSE but better MAE/MAPE/sMAPE than true future weather (Part 6b above).
6. **Recommendation:** SARIMAX for this single house — best worst-case accuracy, native uncertainty quantification, most interpretable, cheapest to run; Chronos is a genuine alternative at multi-building or cold-start scale.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "analysis_standalone.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
