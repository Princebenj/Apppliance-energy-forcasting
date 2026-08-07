"""
data_prep.py
============
Load the UCI "Appliances Energy Prediction" dataset, parse timestamps,
check missing values, and resample from 10-minute to hourly resolution.

Source data: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
Sampled every 10 minutes from 2016-01-11 to 2016-05-27 (Candanchu, Belgium house).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "energydata_complete.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw 10-minute resolution CSV and parse the timestamp index."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Drop the two random/noise variables (rv1, rv2) documented by the dataset
    # authors as randomly generated variables used only to test feature
    # selection robustness - they carry no real signal for forecasting.
    df = df.drop(columns=["rv1", "rv2"], errors="ignore")
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return per-column missing value counts, and check timestamp regularity."""
    missing = df.isna().sum()
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    missing_timestamps = full_range.difference(df.index)
    print(f"Expected 10-min timestamps: {len(full_range)}")
    print(f"Actual rows: {len(df)}")
    print(f"Missing timestamps (gaps in the 10-min grid): {len(missing_timestamps)}")
    print(f"NaN counts per column (top 5):\n{missing.sort_values(ascending=False).head()}")
    return missing


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample from 10-minute to hourly resolution.

    Appliances and lights are energy-use readings (Wh per 10-min interval),
    so we SUM them to get hourly Wh. Sensor/weather variables are
    instantaneous readings, so we take the MEAN over each hour.
    """
    sum_cols = ["Appliances", "lights"]
    mean_cols = [c for c in df.columns if c not in sum_cols]

    hourly_sum = df[sum_cols].resample("h").sum()
    hourly_mean = df[mean_cols].resample("h").mean()
    hourly = pd.concat([hourly_sum, hourly_mean], axis=1)[df.columns]
    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic calendar/time-of-day features (used later in Part 5)."""
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek  # Monday=0
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = out.index.month
    return out


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw()
    print("=== RAW (10-min) ===")
    print(df_raw.shape)
    check_missing(df_raw)

    df_hourly = resample_hourly(df_raw)
    df_hourly = add_time_features(df_hourly)
    print("\n=== HOURLY ===")
    print(df_hourly.shape)
    print(df_hourly.isna().sum().sum(), "total NaNs after resampling")

    out_path = PROCESSED_DIR / "energy_hourly.parquet"
    df_hourly.to_parquet(out_path)
    print(f"Saved hourly data to {out_path}")
    return df_hourly


if __name__ == "__main__":
    main()
