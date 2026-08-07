"""
evaluation.py
=============
Defines the forecasting problem (Part 2) and provides a shared,
rolling-origin backtesting harness + metrics used identically by every
model in this project (benchmarks, SARIMAX, ML, foundation model), so
that comparisons in Part 8 are apples-to-apples.

FORECASTING PROBLEM DEFINITION
-------------------------------
Target variable   : Appliances (Wh consumed per hour by appliances)
Frequency          : Hourly (resampled from 10-min raw data)
Forecast horizon   : 24 hours ahead (h = 1..24)
Test period        : Last 14 days of the dataset (336 hours)
Train period        : Everything before the test period (2,954 hours / ~123 days)
Evaluation design  : Rolling-origin ("walk-forward") backtesting.
                      The test period is split into 14 consecutive,
                      non-overlapping 24-hour blocks. For each block,
                      the model is given all data up to (not including)
                      that block's start and must forecast the next 24
                      hours. This yields 14 independent 24h-horizon
                      forecasts, giving a much more robust estimate of
                      real-world performance than a single forecast,
                      and mirrors how the model would actually be
                      deployed (re-forecast once per day).
Metrics             : RMSE, MAE, MAPE, sMAPE - computed (a) pooled over
                      all 14*24 = 336 predictions, and (b) by horizon
                      step (h=1..24) to see how error grows with lead time.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "energy_hourly.parquet"
TARGET = "Appliances"
HORIZON = 24
TEST_DAYS = 14


def load_hourly():
    return pd.read_parquet(PROCESSED)


def train_test_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    test_hours = test_days * 24
    train = df.iloc[:-test_hours].copy()
    test = df.iloc[-test_hours:].copy()
    return train, test


def get_origins(df: pd.DataFrame, test_days: int = TEST_DAYS, horizon: int = HORIZON):
    """Return the list of forecast-origin timestamps: the last timestamp of
    training data available for each of the 14 rolling 24h test blocks."""
    test_hours = test_days * 24
    test_start_idx = len(df) - test_hours
    origins = []
    for block in range(test_days):
        origin_idx = test_start_idx + block * horizon - 1  # last obs before block
        origins.append(df.index[origin_idx])
    return origins


# ---------------------------------------------------------------- metrics --
def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100)


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=1e-6)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def all_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


# --------------------------------------------------------- results store --
def forecasts_to_frame(model_name: str, origins, y_true_blocks, y_pred_blocks) -> pd.DataFrame:
    """Pack a model's rolling-origin forecasts into a tidy long dataframe:
    columns = [model, origin, horizon, timestamp, y_true, y_pred]"""
    rows = []
    for origin, yt, yp in zip(origins, y_true_blocks, y_pred_blocks):
        timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=len(yt), freq="h")
        for h, (ts, t, p) in enumerate(zip(timestamps, yt, yp), start=1):
            rows.append({"model": model_name, "origin": origin, "horizon": h,
                         "timestamp": ts, "y_true": t, "y_pred": p})
    return pd.DataFrame(rows)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per model, from the long-format results frame."""
    out = []
    for model, g in results_df.groupby("model"):
        m = all_metrics(g["y_true"], g["y_pred"])
        m["model"] = model
        out.append(m)
    return pd.DataFrame(out).set_index("model")[["RMSE", "MAE", "MAPE", "sMAPE"]].sort_values("RMSE")


if __name__ == "__main__":
    df = load_hourly()
    train, test = train_test_split(df)
    origins = get_origins(df)
    print(f"Full data: {len(df)} hours ({df.index.min()} to {df.index.max()})")
    print(f"Train: {len(train)} hours ({train.index.min()} to {train.index.max()})")
    print(f"Test:  {len(test)} hours ({test.index.min()} to {test.index.max()})")
    print(f"Rolling-origin test blocks: {len(origins)} blocks of {HORIZON}h each")
    print(f"First 3 origins: {origins[:3]}")
