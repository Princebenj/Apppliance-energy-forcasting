# Appliance Energy Demand Forecasting — Time Series Case Study

Hourly forecasting of household appliance electricity demand using benchmark
models, SARIMAX, a feature-based gradient-boosting model (XGBoost), and a
time-series foundation model (Chronos), evaluated with a rolling-origin
backtest over 24-hour horizons.

## To Access The Notebooks

The Notebook files is inside the folder Energy Forecast, Notebooks, 

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
  are pooled for the final metrics. See `src/evaluation.py` for full detail
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
│   ├── data_prep.py          # Part 1: load, clean, resample to hourly
│   ├── eda.py                 # Part 1: plots, decomposition, ADF/KPSS, ACF/PACF
│   ├── evaluation.py          # Part 2: problem definition + shared metrics/backtest harness
│   ├── benchmarks.py          # Part 3: Mean, Naive, Seasonal Naive (daily/weekly), Drift
│   ├── sarimax.py              # Part 4: AIC grid search + SARIMAX rolling backtest
│   ├── features.py             # Part 5: sensor/weather/time/lag/rolling features
│   ├── ml_model.py             # Part 6: XGBoost recursive multi-step forecasting
│   ├── foundation_model.py     # Part 7: Chronos (RUN SEPARATELY - see file docstring)
│   ├── compare_models.py       # Part 8: aggregate all results, comparison table + plots
│   └── run_pipeline.py         # Runs everything above in order
├── outputs/
│   ├── figures/                # all generated plots (numbered by part/order produced)
│   └── metrics/                # CSV/parquet: grid search results, forecasts, comparison table
├── reports/
│   └── report.docx             # 8-page written report
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

This downloads/prepares the data (place `energydata_complete.csv` in
`data/raw/` first, or point `data_prep.py` at the UCI URL if your network
allows it), then runs Parts 1, 3, 4, 5, 6, and 8 end to end. **Part 4
(SARIMAX) is the slow step** (~30-40 minutes on 1 CPU core) because the
assignment requires an exhaustive AIC grid search over p∈[0,6], d∈[0,2],
q∈[0,6] (147 fits) plus 14 further refits for the rolling backtest; every
expensive step checkpoints to `outputs/metrics/` so an interrupted run can
be resumed by simply re-running the same command.

**Part 7 (Chronos foundation model) must be run separately**, in an
environment with internet access to Hugging Face (e.g. Google Colab — see
the docstring at the top of `src/foundation_model.py` for exact steps, ~2
minutes on Colab's free tier). Copy the resulting
`outputs/metrics/chronos_forecasts.parquet` into this project and re-run
`src/compare_models.py` to fold it into the final comparison table and plots.

## Key results (pooled RMSE/MAE over 14×24h test blocks)

| Model | RMSE | MAE | MAPE | sMAPE |
|---|---|---|---|---|
| **SARIMAX** | **381.6** | **215.6** | 33.3% | 30.0% |
| XGBoost | 395.9 | 254.6 | 44.5% | 35.1% |
| Mean (strongest RMSE benchmark) | 436.3 | 296.1 | 53.6% | 46.0% |
| SeasonalNaive_Weekly (strongest MAE benchmark) | 474.7 | 254.6 | 37.1% | 32.5% |
| SeasonalNaive_Daily | 510.7 | 288.6 | 43.8% | 35.6% |
| Naive | 660.3 | 512.0 | 113.4% | 65.4% |
| Drift | 662.1 | 513.6 | 113.8% | 65.5% |

SARIMAX(1,0,6)(0,1,1)[24] with exogenous outdoor temperature/humidity was
the best-performing model overall, improving RMSE by 12.6% over the
strongest benchmark. Full discussion, limitations, and the foundation-model
comparison are in `reports/report.docx`.

## Notes on methodology / caveats

- **Conditional forecasts:** SARIMAX and XGBoost use future outdoor weather
  (T_out, RH_out) as exogenous inputs, taken from the real test data. This
  is a *conditional* forecast (assumes perfect weather knowledge), not a
  pure unconditional forecast — a real deployment would substitute an
  actual weather forecast, adding uncertainty not captured here. See report
  Part 9 Q5.
- **Data leakage control:** all lag/rolling features are shifted to only use
  information strictly before each timestamp; multi-step XGBoost forecasts
  are generated recursively, feeding earlier predictions into later lag
  features rather than peeking at true future values. See `src/features.py`
  and `src/ml_model.py` docstrings.
- **rv1/rv2** (random noise columns documented by the original dataset
  authors) are dropped in `data_prep.py`.
