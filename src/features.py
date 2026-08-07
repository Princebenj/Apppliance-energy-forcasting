"""
features.py
============
Part 5: Feature engineering for the feature-based ML model (Part 6).

Feature groups
--------------
1. Sensor covariates   : indoor temperature/humidity sensors (T1-T9, RH_1-RH_9)
2. Weather covariates   : T_out, RH_out, Press_mm_hg, Windspeed, Visibility, Tdewpoint
3. Time-based features  : hour-of-day (cyclical sin/cos), day-of-week (cyclical
                          sin/cos), is_weekend, month
4. Lag features         : Appliances at t-1, t-2, t-3, t-24 (same hour yesterday),
                          t-168 (same hour last week)
5. Rolling-window feats : rolling mean/std of Appliances over past 3h, 24h, 168h
                          (computed using only past data - no leakage)

IMPORTANT - leakage control
----------------------------
All lag/rolling features are shifted so that, at any timestamp t, a feature
only uses information from times < t. When these features are used in a
forecasting loop, `Appliances` beyond the origin is UNKNOWN, so lag/rolling
features for multi-step-ahead horizons (h=2..24) are computed recursively
using the model's own PREVIOUS predictions, not the true future values. See
`build_forecast_features` and the recursive loop in ml_model.py. Sensor and
weather covariates for h=2..24, however, are assumed known (a "conditional
forecast" caveat discussed explicitly in the report, Part 9 Q5).
"""

import numpy as np
import pandas as pd

TARGET = "Appliances"
SENSOR_COLS = [c for c in
               ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
                "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9"]]
WEATHER_COLS = ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
LAGS = [1, 2, 3, 24, 168]
ROLL_WINDOWS = [3, 24, 168]


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    out["month"] = out.index.month
    return out


def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for w in ROLL_WINDOWS:
        # shift(1) first so the rolling window never includes the current value
        out[f"rollmean_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"rollstd_{w}"] = out[target_col].shift(1).rolling(w).std()
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_cyclical_time_features(df)
    out = add_lag_and_rolling_features(out)
    return out


def get_feature_columns():
    time_feats = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month"]
    lag_feats = [f"lag_{l}" for l in LAGS]
    roll_feats = [f"rollmean_{w}" for w in ROLL_WINDOWS] + [f"rollstd_{w}" for w in ROLL_WINDOWS]
    return SENSOR_COLS + WEATHER_COLS + time_feats + lag_feats + roll_feats


if __name__ == "__main__":
    from evaluation import load_hourly
    df = load_hourly()
    feat_df = build_feature_frame(df)
    cols = get_feature_columns()
    print(f"Built {len(cols)} features. NaNs before dropna: {feat_df[cols].isna().any(axis=1).sum()} rows "
          f"(expected ~168 from the longest lag/rolling window)")
    print(feat_df[cols].tail(3).T)
