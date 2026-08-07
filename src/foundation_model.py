"""
foundation_model.py
====================
Part 7: Time-series foundation model (Amazon Chronos) for 24h-ahead
Appliance demand forecasting, using the SAME rolling-origin harness
(evaluation.py) as every other model in this project.

*** RUN THIS IN GOOGLE COLAB (or any machine with internet + ideally a GPU) ***
This sandbox's network is locked to a small package-registry allowlist and
cannot reach huggingface.co to download the pretrained Chronos weights.
Everything else in this project (Parts 1-6, 8, 9, 10) was run end-to-end in
the sandbox; only this one script needs to be run externally. Steps:

  1. Open a new Google Colab notebook (Runtime -> Change runtime type -> GPU
     is optional but speeds this up a lot; CPU also works, just slower).
  2. Upload `data/processed/energy_hourly.parquet` (or re-run
     src/data_prep.py there) and this file's sibling `evaluation.py`.
  3. !pip install chronos-forecasting torch pandas pyarrow
  4. Run this script: `python foundation_model.py`
  5. Download the resulting `outputs/metrics/chronos_forecasts.parquet` and
     drop it into this project's `outputs/metrics/` folder. The evaluation
     notebook (Part 8) will then pick it up automatically alongside the
     other models' saved forecasts.

Model used: amazon/chronos-t5-small (a good speed/accuracy trade-off for a
CPU-only Colab session; chronos-t5-base or -large can be substituted for
higher accuracy at higher compute cost - see Part 9 Q4 discussion of
whether the accuracy gain justifies the extra cost).

Chronos is a zero-shot univariate forecaster (pretrained on a large corpus
of diverse time series, no training on THIS data at all) - it only sees the
Appliances history at each origin, not the sensor/weather covariates. This
is an important, deliberate point of comparison for Part 9 Q4: can a
model with no access to our engineered features and no fine-tuning on this
domain still compete with SARIMAX/XGBoost?
"""

import numpy as np
import pandas as pd
from pathlib import Path

from evaluation import load_hourly, get_origins, forecasts_to_frame, summarize, TARGET, HORIZON

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
MODEL_ID = "amazon/chronos-t5-small"


def run_chronos_backtest():
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    df = load_hourly()
    origins = get_origins(df)
    y_true_blocks, y_pred_blocks = [], []

    for i, origin in enumerate(origins):
        history = df.loc[:origin, TARGET]
        context = torch.tensor(history.values, dtype=torch.float32)

        # Chronos returns num_samples sample paths; use the median as the
        # point forecast and keep the 10th/90th percentile for an interval.
        forecast = pipeline.predict(context=context, prediction_length=HORIZON, num_samples=100)
        samples = forecast[0].numpy()  # shape (num_samples, horizon)
        y_pred = np.median(samples, axis=0)
        y_pred = np.clip(y_pred, 0, None)

        start_loc = df.index.get_loc(origin) + 1
        future_idx = df.index[start_loc: start_loc + HORIZON]
        y_true = df.loc[future_idx, TARGET].values

        y_true_blocks.append(y_true)
        y_pred_blocks.append(y_pred)
        print(f"  origin {i+1}/{len(origins)} ({origin}) done")

    results_df = forecasts_to_frame("Chronos", origins, y_true_blocks, y_pred_blocks)
    return results_df


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = run_chronos_backtest()
    out_path = METRICS_DIR / "chronos_forecasts.parquet"
    results_df.to_parquet(out_path)

    summary = summarize(results_df)
    print("\n=== Chronos performance (pooled, 14 x 24h blocks) ===")
    print(summary.round(2))
    print(f"\nSaved forecasts to {out_path} - copy this file into the main "
          f"project's outputs/metrics/ folder before running Part 8 evaluation.")


if __name__ == "__main__":
    main()
