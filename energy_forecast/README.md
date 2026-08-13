# Appliance Energy Demand Forecasting — Time Series Case Study

Hourly forecasting of household appliance electricity demand using benchmark
models, SARIMAX, a feature-based gradient-boosting model (XGBoost), and a
time-series foundation model (Chronos), evaluated with a rolling-origin
backtest over 24-hour horizons.

## Dataset

[UCI "Appliances Energy Prediction" dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction):
10-minute readings of appliance energy use, indoor sensor data (temperature/
humidity across 9 rooms), and outdoor weather, for a low-energy house in
Stambruges, Belgium, from 11 Jan 2016 to 27 May 2016 (19,735 rows).

## Problem definition

- **Target:** `Appliances` — hourly appliance energy use (Wh), resampled from
  the raw 10-minute data (summed per hour).
- **Forecast horizon:** 24 hours ahead.
- **Evaluation design:** rolling-origin ("walk-forward") backtest over the
  last 14 days of data. The test period is split into 14 consecutive,
  non-overlapping 24h blocks; each model forecasts each block using only
  data available up to that block's start, then all 14×24=336 predictions
  are pooled for the final metrics. See `src/all_in_one.py` for full detail
  and rationale.
- **Metrics:** RMSE, MAE, MAPE, sMAPE — pooled, and broken down by horizon
  step to see how error grows with lead time.

## Repository structure

```
energy_forecast/
├── data/
│   ├── raw/                  # energydata_complete.csv (downloaded, gitignored if large)
│   └── processed/            # energy_hourly.parquet (resampled, feature-ready)
├── src/
│   └── all_in_one.py         # THE ENTIRE PIPELINE (Parts 1-8) in a single file
├── notebooks/
│   ├── analysis.ipynb              # narrative walkthrough, imports src/all_in_one.py
│   ├── analysis_standalone.ipynb   # same walkthrough, code embedded directly (no import needed)
│   └── analysis_colab_executed.ipynb  # the actual Colab run, WITH Chronos + real results saved inside
├── outputs/
│   ├── figures/                # all generated plots (numbered by part/order produced)
│   └── metrics/                # CSV/parquet: grid search results, forecasts, comparison table
├── reports/
│   ├── report.docx             # 8-page written report
│   └── report.pdf
├── tools/                    # authoring/build scripts (not part of the pipeline itself)
│   ├── merge_into_one.py       # regenerates src/all_in_one.py from source, if ever needed
│   ├── build_notebook.py       # regenerates notebooks/analysis.ipynb
│   └── build_standalone_notebook.py  # regenerates notebooks/analysis_standalone.ipynb
├── requirements.txt
└── README.md
```

`src/all_in_one.py` is organised internally by assignment part (look for
the `# ===== SOURCE: ... =====` banners), with one function per part:
`run_part1_data_prep`, `run_part1_eda`, `run_part3_benchmarks`,
`run_part4_sarimax`, `run_part6_ml_model`, `run_part6b_ml_model_realistic_weather`,
`run_part7_foundation_model`, `run_part8_compare_models`, called in order
from a single `main()` at the bottom.




## How to reproduce

```bash
pip install -r requirements.txt
python src/all_in_one.py
```

This downloads/prepares the data (place `energydata_complete.csv` in
`data/raw/` first, or point `run_part1_data_prep()` at the UCI URL if your
network allows it), then runs Parts 1, 3, 4, 5, 6, and 8 end to end. **Part
4 (SARIMAX) is the slow step** (~30-40 minutes on 1 CPU core) because the
assignment requires an exhaustive AIC grid search over p∈[0,6], d∈[0,2],
q∈[0,6] (147 fits) plus 14 further refits for the rolling backtest; every
expensive step checkpoints to `outputs/metrics/` so an interrupted run can
be resumed by simply re-running the same command.

**Part 7 (Chronos foundation model) must be run separately**, in an
environment with internet access to Hugging Face (e.g. Google Colab):

```bash
pip install chronos-forecasting torch
python src/all_in_one.py --with-chronos
```

Copy the resulting `outputs/metrics/chronos_forecasts.parquet` into this
project and re-run `python src/all_in_one.py` (Part 8 will pick it up and
fold it into the final comparison table and plots automatically).

## How to run the notebooks

There are two notebooks in `notebooks/`:

- **`analysis.ipynb`** — imports and calls functions from `src/all_in_one.py`
- **`analysis_standalone.ipynb`** — the same analysis, but with every
  function's code written directly in its own cells (no import from
  `src/all_in_one.py` needed)

Both notebooks already have their outputs saved inside them, so you can
just open and read them with no setup at all — GitHub, Colab, and VS Code
all render the saved tables and plots automatically. You only need the
steps below if you want to **re-run** a notebook's cells yourself.

### What the notebooks need, and why

Both notebooks use **relative paths** (e.g. `../data`, `../outputs/figures`,
`../outputs/metrics`) that assume the notebook stays inside the
`notebooks/` folder, sitting next to the rest of the project exactly as it
is in this repository:

```
energy_forecast/
├── data/
│   ├── raw/energydata_complete.csv        ← notebooks read this
│   └── processed/energy_hourly.parquet    ← notebooks read this
├── notebooks/
│   ├── analysis.ipynb                     ← run from inside here
│   └── analysis_standalone.ipynb
├── outputs/
│   ├── figures/*.png                      ← notebooks display these images
│   └── metrics/*.csv, *.parquet           ← notebooks load these cached results
└── src/all_in_one.py
```

**Do not move or upload the `.ipynb` file on its own** — if you drag just
the notebook into Colab by itself, none of the `../data` or `../outputs`
paths will resolve and every cell that loads a figure or a cached result
will fail. You need the whole `energy_forecast` folder together.

### Step by step

**Option A — Locally (VS Code, Jupyter, or GitHub Codespaces)**
1. Make sure the full `energy_forecast` folder (with `data/`, `outputs/`,
   `src/`, `notebooks/` all present) is on disk / in your Codespace — this
   is exactly what you get by cloning the repo or unzipping the project.
2. Install dependencies once: `pip install -r requirements.txt jupyter`
3. Open the notebook and run it **from inside the `notebooks/` folder**
   (this matters for the relative paths):
   ```bash
   cd energy_forecast/notebooks
   jupyter notebook analysis_standalone.ipynb
   ```
   or just open the `.ipynb` file directly in VS Code / Codespaces — the
   working directory is set automatically to the notebook's own folder.
4. Run all cells: Run → Run All Cells (or Restart Kernel and Run All).

**Option B — Google Colab**
1. Go to colab.research.google.com → File → Upload notebook, and pick
   `analysis_standalone.ipynb` (this version needs no `src/` import).
2. Colab starts in an empty environment, so you must upload the supporting
   files/folders it will read via `../data` and `../outputs`. In the left
   sidebar, click the folder icon → upload icon, and recreate this
   structure under `/content/`:
   ```
   /content/data/raw/energydata_complete.csv
   /content/data/processed/energy_hourly.parquet
   /content/outputs/figures/            (all the .png files)
   /content/outputs/metrics/            (all the .csv and .parquet files)
   /content/notebooks/analysis_standalone.ipynb   ← your notebook, one level below the above
   ```
   Easiest way to get all of that in one go: zip your local `energy_forecast`
   folder, upload the single `.zip` to Colab's file pane, then run this in
   a Colab code cell to unzip it in place:
   ```python
   !unzip -q energy_forecast.zip -d /content/
   ```
   Then open/run `/content/energy_forecast/notebooks/analysis_standalone.ipynb`.
3. Install packages Colab doesn't already have:
   ```python
   !pip install -q statsmodels xgboost pyarrow
   ```
4. Run all cells: Runtime → Run all.

If a cell errors with something like `FileNotFoundError:
../outputs/figures/01_series_overview.png`, it means the notebook isn't
sitting in the right place relative to `data/` and `outputs/` — re-check
the folder layout above rather than editing the notebook's code.

## Key results (pooled RMSE/MAE over 14×24h test blocks)

| Model | RMSE | MAE | MAPE | sMAPE |
|---|---|---|---|---|
| **SARIMAX** | **381.6** | 215.7 | 33.3% | 30.0% |
| XGBoost | 390.2 | 253.7 | 44.5% | 35.2% |
| Chronos (foundation model) | 428.5 | **209.9** | **23.0%** | **28.0%** |
| Mean (strongest RMSE benchmark) | 436.3 | 296.1 | 53.6% | 46.0% |
| SeasonalNaive_Weekly (strongest MAE benchmark) | 474.7 | 254.6 | 37.1% | 32.5% |
| SeasonalNaive_Daily | 510.7 | 288.6 | 43.8% | 35.6% |
| Naive | 660.3 | 512.0 | 113.4% | 65.4% |
| Drift | 662.1 | 513.6 | 113.8% | 65.5% |

SARIMAX(1,0,6)(0,1,1)[24] with exogenous outdoor temperature/humidity had
the best RMSE overall (+12.5% over the strongest benchmark), narrowly ahead
of XGBoost. **Chronos** — run zero-shot in Colab with no fine-tuning and no
access to weather/sensor data — had a higher RMSE but the **best MAE, MAPE,
and sMAPE of every model tested**, a genuine worst-case-vs-typical-case
split rather than a simple ranking. Full discussion, limitations, and the
Q5 realistic-weather follow-up are in `reports/report.docx`.

## Notes on methodology / caveats

- **Conditional forecasts:** SARIMAX and XGBoost use future outdoor weather
  (T_out, RH_out) as exogenous inputs, taken from the real test data. This
  is a *conditional* forecast (assumes perfect weather knowledge), not a
  pure unconditional forecast — a real deployment would substitute an
  actual weather forecast, adding uncertainty not captured here. Tested
  directly with `run_part6b_ml_model_realistic_weather()`: a 24h-lagged
  weather proxy gives slightly worse RMSE (398.3) but better MAE/MAPE/sMAPE
  (243.8/38.8%/33.3%) than the conditional version. See report Part 9 Q5.
- **Data leakage control:** all lag/rolling features are shifted to only use
  information strictly before each timestamp; multi-step XGBoost forecasts
  are generated recursively, feeding earlier predictions into later lag
  features rather than peeking at true future values. See `src/all_in_one.py`
  (`build_lag_and_rolling_features`, `recursive_forecast`) for the exact logic.
- **rv1/rv2** (random noise columns documented by the original dataset
  authors) are dropped in `data_prep.py`.
