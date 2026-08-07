"""
ml_model.py
===========
Part 6: Feature-based ML model (XGBoost) for 24h-ahead Appliance demand
forecasting, using the sensor/weather/time/lag features from Part 5.

Forecasting strategy: RECURSIVE multi-step.
At each origin, the model predicts h=1, then uses that prediction to update
the lag_1/lag_2/lag_3/rolling features before predicting h=2, and so on to
h=24. lag_24 and lag_168 features fall outside the 24h horizon for h=1 (they
reference real past data) but for h>1 they may start to reference the
model's own earlier predictions in the SAME block only once the horizon
exceeds the lag length - not the case here since horizon=24 <= lag_24=24
lag_168=168, so lag_24/lag_168 always reference genuine historical data
within a rolling-origin block. Only lag_1/2/3 and the rolling windows are
recursively updated with predictions. This is documented explicitly because
it is a common source of DATA LEAKAGE if not handled carefully (see Part 9
Q3 discussion).

Sensor/weather covariates for future hours are taken from the real test
data (a conditional-forecast assumption, same as SARIMAX - see Part 9 Q5).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from evaluation import (load_hourly, get_origins, forecasts_to_frame,
                         summarize, TARGET, HORIZON, TEST_DAYS)
from features import build_feature_frame, get_feature_columns, LAGS, ROLL_WINDOWS

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def prepare_training_data(df: pd.DataFrame, train_end):
    """Build the feature matrix for all rows up to (and including) train_end,
    dropping rows with NaN features (the first 168 hours, from the longest
    lag/rolling window)."""
    feat_df = build_feature_frame(df.loc[:train_end])
    cols = get_feature_columns()
    feat_df = feat_df.dropna(subset=cols + [TARGET])
    X = feat_df[cols]
    y = feat_df[TARGET]
    return X, y


def fit_xgb(X, y):
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=4,
    )
    model.fit(X, y)
    return model


def recursive_forecast(model, df: pd.DataFrame, origin, horizon: int, feature_cols):
    """Recursively forecast `horizon` steps ahead of `origin`, updating
    lag/rolling features with the model's own predictions as it goes.
    Sensor/weather covariates for future timestamps are taken from `df`
    (conditional-forecast assumption)."""
    history = df.loc[:origin, [TARGET] + [c for c in df.columns if c != TARGET]].copy()
    start_loc = df.index.get_loc(origin) + 1
    future_idx = df.index[start_loc: start_loc + horizon]

    preds = []
    working = df.copy()  # has real sensor/weather + real Appliances up to origin
    # From origin+1 onward, blank out Appliances so we can fill with predictions
    working.loc[future_idx, TARGET] = np.nan

    for ts in future_idx:
        feat_row = build_feature_frame(working.loc[:ts]).iloc[[-1]][feature_cols]
        pred = float(model.predict(feat_row)[0])
        pred = max(pred, 0)  # energy use can't be negative
        working.loc[ts, TARGET] = pred
        preds.append(pred)

    return np.array(preds)


def rolling_origin_ml(df: pd.DataFrame, checkpoint_path=None):
    origins = get_origins(df)
    feature_cols = get_feature_columns()

    done_origins = set()
    all_rows = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        prev = pd.read_parquet(checkpoint_path)
        all_rows = [prev]
        done_origins = set(pd.to_datetime(prev["origin"]).unique())
        print(f"  Resuming: {len(done_origins)} origins already checkpointed.")

    importances = []
    for i, origin in enumerate(origins):
        if pd.Timestamp(origin) in done_origins:
            print(f"  origin {i+1}/{len(origins)} ({origin}) already done, skipping")
            continue
        X_train, y_train = prepare_training_data(df, origin)
        model = fit_xgb(X_train, y_train)
        importances.append(model.feature_importances_)

        y_pred = recursive_forecast(model, df, origin, HORIZON, feature_cols)
        start_loc = df.index.get_loc(origin) + 1
        future_idx = df.index[start_loc: start_loc + HORIZON]
        y_true = df.loc[future_idx, TARGET].values

        block_df = forecasts_to_frame("XGBoost", [origin], [y_true], [y_pred])
        all_rows.append(block_df)
        if checkpoint_path is not None:
            pd.concat(all_rows, ignore_index=True).to_parquet(checkpoint_path)
        print(f"  origin {i+1}/{len(origins)} ({origin}) done", flush=True)

    results_df = pd.concat(all_rows, ignore_index=True)
    avg_importance = pd.Series(np.mean(importances, axis=0), index=feature_cols).sort_values(ascending=False) \
        if importances else None
    return results_df, avg_importance


def plot_feature_importance(importance: pd.Series, top_n=20, fname="10_xgb_feature_importance.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    top = importance.head(top_n).iloc[::-1]
    ax.barh(top.index, top.values, color="#2b6cb0")
    ax.set_title(f"XGBoost Feature Importance (avg over {len(importance)} features, top {top_n})")
    ax.set_xlabel("Importance (gain-based)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hourly()
    print("Running rolling-origin XGBoost backtest (14 origins, recursive 24h forecasts)...")
    results_df, importance = rolling_origin_ml(df, checkpoint_path=METRICS_DIR / "xgboost_forecasts.parquet")
    results_df.to_parquet(METRICS_DIR / "xgboost_forecasts.parquet")
    if importance is not None:
        importance.to_csv(METRICS_DIR / "xgboost_feature_importance.csv")

    if importance is not None:
        plot_feature_importance(importance)
    else:
        importance = pd.read_csv(METRICS_DIR / "xgboost_feature_importance.csv", index_col=0).iloc[:, 0]
        plot_feature_importance(importance)

    summary = summarize(results_df)
    print("\n=== XGBoost performance (pooled, 14 x 24h blocks) ===")
    print(summary.round(2))
    print("\nTop 10 features:")
    print(importance.head(10))


if __name__ == "__main__":
    main()
