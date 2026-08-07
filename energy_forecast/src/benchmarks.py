"""
benchmarks.py
=============
Part 3: Benchmark forecasters - Mean, Naive, Daily Seasonal Naive,
Weekly Seasonal Naive, and Drift - evaluated with the shared
rolling-origin harness from evaluation.py (24h horizon, 14 test blocks).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from evaluation import (load_hourly, train_test_split, get_origins,
                         forecasts_to_frame, summarize, TARGET, HORIZON, TEST_DAYS)

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def mean_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast = historical mean, repeated for the whole horizon."""
    return np.full(horizon, history.mean())


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast = last observed value, repeated for the whole horizon."""
    return np.full(horizon, history.iloc[-1])


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Forecast = value observed exactly one season ago, per step.
    season=24 -> daily seasonal naive; season=168 -> weekly seasonal naive."""
    last_season = history.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def drift_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Naive forecast + linear extrapolation of the average historical trend
    (slope between first and last observation)."""
    y1, yT, T = history.iloc[0], history.iloc[-1], len(history)
    slope = (yT - y1) / (T - 1)
    return yT + slope * np.arange(1, horizon + 1)


BENCHMARKS = {
    "Mean": mean_forecast,
    "Naive": naive_forecast,
    "SeasonalNaive_Daily": lambda h, H: seasonal_naive_forecast(h, H, season=24),
    "SeasonalNaive_Weekly": lambda h, H: seasonal_naive_forecast(h, H, season=168),
    "Drift": drift_forecast,
}


def run_benchmarks():
    df = load_hourly()
    origins = get_origins(df)
    all_results = []

    for name, fn in BENCHMARKS.items():
        y_true_blocks, y_pred_blocks = [], []
        for origin in origins:
            history = df.loc[:origin, TARGET]
            future_idx = df.index[df.index.get_loc(origin) + 1: df.index.get_loc(origin) + 1 + HORIZON]
            y_true = df.loc[future_idx, TARGET].values
            y_pred = fn(history, HORIZON)
            y_true_blocks.append(y_true)
            y_pred_blocks.append(y_pred)
        res = forecasts_to_frame(name, origins, y_true_blocks, y_pred_blocks)
        all_results.append(res)

    results_df = pd.concat(all_results, ignore_index=True)
    return results_df


def plot_example_block(results_df: pd.DataFrame, block_origin, fname="07_benchmark_example_block.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    sub = results_df[results_df["origin"] == block_origin]
    truth = sub[sub["model"] == sub["model"].iloc[0]][["timestamp", "y_true"]].drop_duplicates()
    ax.plot(truth["timestamp"], truth["y_true"], color="black", lw=2, label="Actual", zorder=5)
    for model, g in sub.groupby("model"):
        ax.plot(g["timestamp"], g["y_pred"], lw=1.2, ls="--", label=model)
    ax.set_title(f"Benchmark 24h Forecasts vs Actual — origin {block_origin}")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_metrics_by_horizon(results_df: pd.DataFrame, fname="08_benchmark_rmse_by_horizon.png"):
    from evaluation import rmse
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, g in results_df.groupby("model"):
        by_h = g.groupby("horizon").apply(lambda d: rmse(d["y_true"], d["y_pred"]))
        ax.plot(by_h.index, by_h.values, marker="o", ms=3, label=model)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("RMSE (Wh)")
    ax.set_title("Benchmark Models: RMSE by Forecast Horizon (pooled over 14 origins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = run_benchmarks()
    results_df.to_parquet(METRICS_DIR / "benchmark_forecasts.parquet")

    summary = summarize(results_df)
    print("=== Benchmark model comparison (pooled over 14 x 24h test blocks) ===")
    print(summary.round(2))
    summary.to_csv(METRICS_DIR / "benchmark_summary.csv")

    origins = sorted(results_df["origin"].unique())
    plot_example_block(results_df, origins[len(origins) // 2])
    plot_metrics_by_horizon(results_df)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
