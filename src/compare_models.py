"""
compare_models.py
==================
Part 8: Aggregate every model's rolling-origin forecasts into one
comparison table, forecast plots, error diagnostics, and comparison
against the strongest benchmark model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from evaluation import summarize, rmse, all_metrics

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def load_all_results():
    files = {
        "Mean": None, "Naive": None, "SeasonalNaive_Daily": None,
        "SeasonalNaive_Weekly": None, "Drift": None,  # from benchmark_forecasts.parquet
    }
    frames = []
    bench = pd.read_parquet(METRICS_DIR / "benchmark_forecasts.parquet")
    frames.append(bench)

    sarimax_path = METRICS_DIR / "sarimax_forecasts.parquet"
    if sarimax_path.exists():
        frames.append(pd.read_parquet(sarimax_path))

    xgb_path = METRICS_DIR / "xgboost_forecasts.parquet"
    if xgb_path.exists():
        frames.append(pd.read_parquet(xgb_path))

    chronos_path = METRICS_DIR / "chronos_forecasts.parquet"
    if chronos_path.exists():
        frames.append(pd.read_parquet(chronos_path))
        print("Chronos results found and included.")
    else:
        print("NOTE: chronos_forecasts.parquet not found - foundation model "
              "not yet run (must be executed in Colab, see foundation_model.py). "
              "Comparison will proceed without it.")

    return pd.concat(frames, ignore_index=True)


def comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    return summarize(results_df)


def plot_all_forecasts_example(results_df: pd.DataFrame, fname="11_all_models_example_block.png"):
    origins = sorted(results_df["origin"].unique())
    example_origin = origins[len(origins) // 2]
    sub = results_df[results_df["origin"] == example_origin]

    fig, ax = plt.subplots(figsize=(12, 6))
    truth = sub[sub["model"] == sub["model"].iloc[0]][["timestamp", "y_true"]].drop_duplicates()
    ax.plot(truth["timestamp"], truth["y_true"], color="black", lw=2.5, label="Actual", zorder=10)

    palette = plt.cm.tab10.colors
    for i, (model, g) in enumerate(sub.groupby("model")):
        g = g.sort_values("timestamp")
        ax.plot(g["timestamp"], g["y_pred"], lw=1.3, ls="--", label=model, color=palette[i % 10])

    ax.set_title(f"All Models: 24h Forecasts vs Actual — origin {example_origin}")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_rmse_by_horizon_all(results_df: pd.DataFrame, fname="12_rmse_by_horizon_all_models.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, g in results_df.groupby("model"):
        by_h = g.groupby("horizon").apply(lambda d: rmse(d["y_true"], d["y_pred"]))
        lw = 2.5 if model in ("SARIMAX", "XGBoost", "Chronos") else 1.2
        ls = "-" if model in ("SARIMAX", "XGBoost", "Chronos") else "--"
        ax.plot(by_h.index, by_h.values, marker="o", ms=3, label=model, lw=lw, ls=ls)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("RMSE (Wh)")
    ax.set_title("RMSE by Forecast Horizon — All Models")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_error_diagnostics(results_df: pd.DataFrame, model_name: str, fname_suffix=""):
    g = results_df[results_df["model"] == model_name].copy()
    g["error"] = g["y_pred"] - g["y_true"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(g["y_true"], g["y_pred"], alpha=0.4, s=15)
    lims = [0, max(g["y_true"].max(), g["y_pred"].max())]
    axes[0].plot(lims, lims, color="red", ls="--", lw=1)
    axes[0].set_xlabel("Actual (Wh)")
    axes[0].set_ylabel("Predicted (Wh)")
    axes[0].set_title(f"{model_name}: Predicted vs Actual")

    axes[1].hist(g["error"], bins=30, color="#c0392b")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title(f"{model_name}: Forecast Error Distribution")
    axes[1].set_xlabel("Error (Predicted - Actual, Wh)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"13_error_diagnostics_{model_name}{fname_suffix}.png", dpi=150)
    plt.close(fig)


def compare_to_strongest_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    benchmark_models = ["Mean", "Naive", "SeasonalNaive_Daily", "SeasonalNaive_Weekly", "Drift"]
    bench_summary = summary.loc[summary.index.intersection(benchmark_models)]
    strongest = bench_summary["RMSE"].idxmin()
    print(f"Strongest benchmark (by RMSE): {strongest}")

    out = summary.copy()
    out["RMSE_improvement_vs_strongest_benchmark_%"] = (
        (summary.loc[strongest, "RMSE"] - summary["RMSE"]) / summary.loc[strongest, "RMSE"] * 100
    ).round(1)
    return out, strongest


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results_df = load_all_results()
    results_df.to_parquet(METRICS_DIR / "all_model_forecasts.parquet")

    summary = comparison_table(results_df)
    print("\n=== FULL MODEL COMPARISON (pooled over 14 x 24h test blocks) ===")
    print(summary.round(2))

    summary_with_improvement, strongest = compare_to_strongest_benchmark(summary)
    summary_with_improvement.to_csv(METRICS_DIR / "final_comparison_table.csv")
    print(f"\n=== With improvement over strongest benchmark ({strongest}) ===")
    print(summary_with_improvement.round(2))

    plot_all_forecasts_example(results_df)
    plot_rmse_by_horizon_all(results_df)
    for model in results_df["model"].unique():
        plot_error_diagnostics(results_df, model)

    print(f"\nAll comparison outputs saved to {FIG_DIR} and {METRICS_DIR}")


if __name__ == "__main__":
    main()
