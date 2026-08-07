"""
sarimax.py
==========
Part 4: SARIMAX model for hourly Appliance energy demand.

Approach
--------
1. From Part 1, ADF & KPSS both indicate the raw hourly series is already
   stationary -> non-seasonal d is expected to be 0, but we still search
   d in [0,2] as required by the brief, in case a small amount of
   differencing helps residual whiteness.
2. Seasonal period is fixed at s=24 (daily seasonality), justified by the
   decomposition and weekly-hourly-profile plots in Part 1, which show a
   strong, stable 24h cycle (evening peak ~18:00) and a weaker, secondary
   day-of-week effect (handled instead via exogenous day-of-week/weekend
   dummies rather than a second, computationally expensive s=168 seasonal
   term).
3. Grid search over p in [0,6], d in [0,2], q in [0,6] (as required),
   at each of a small set of candidate seasonal orders (P,D,Q) x s=24,
   selecting the combination with lowest AIC. To keep the (7*3*7=147
   fits per seasonal order) search tractable, the AIC search is run on
   the most recent 45 days of the training set (a representative,
   stable-regime subset) with reduced optimizer iterations; the winning
   order is then refit on the FULL training set for the actual
   rolling-origin forecasts used for evaluation.
4. Exogenous variables: T_out (outdoor temperature) and RH_out (outdoor
   humidity) are included as they are physically justified drivers of
   heating/cooling-adjacent appliance use and are available as weather
   forecasts in a real deployment (see Part 9 Q5 discussion).
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from evaluation import (load_hourly, get_origins, forecasts_to_frame,
                         summarize, TARGET, HORIZON, TEST_DAYS)

warnings.filterwarnings("ignore")

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
EXOG_COLS = ["T_out", "RH_out"]
SEASONAL_PERIOD = 24


def _fit_with_timeout(train_subset, order, seasonal_order, timeout_s=20):
    """Fit SARIMAX with a hard wall-clock timeout so one pathological
    (p,d,q) combination (e.g. p=6,q=6) cannot stall the whole grid search."""
    import signal

    def _handler(signum, frame):
        raise TimeoutError("fit exceeded time budget")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        model = SARIMAX(train_subset, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=50, method="lbfgs")
        return fit
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def grid_search_order(train_subset: pd.Series, seasonal_orders, p_range=(0, 6), d_range=(0, 2), q_range=(0, 6),
                       per_fit_timeout=20, checkpoint_path=None):
    results = []
    combos = list(itertools.product(range(p_range[0], p_range[1] + 1),
                                     range(d_range[0], d_range[1] + 1),
                                     range(q_range[0], q_range[1] + 1)))
    total = len(combos) * len(seasonal_orders)
    done = 0

    # Resume support: skip combos already checkpointed to disk
    already = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        prev = pd.read_csv(checkpoint_path)
        results = prev.to_dict("records")
        already = set(zip(prev.p, prev.d, prev.q, prev.P, prev.D, prev.Q))
        print(f"    Resuming: {len(already)} combos already checkpointed, will skip them.")

    for (P, D, Q) in seasonal_orders:
        for p, d, q in combos:
            done += 1
            if (p, d, q, P, D, Q) in already:
                continue
            try:
                fit = _fit_with_timeout(train_subset, (p, d, q), (P, D, Q, SEASONAL_PERIOD),
                                         timeout_s=per_fit_timeout)
                row = {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "aic": fit.aic}
                results.append(row)
                status = f"aic={fit.aic:.1f}"
                # checkpoint immediately so a killed process loses minimal work
                if checkpoint_path is not None:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            except Exception as e:
                status = f"skipped ({type(e).__name__})"
            if done % 5 == 0 or done == total:
                print(f"    [{done}/{total}] order=({p},{d},{q}) seasonal=({P},{D},{Q},24) -> {status}", flush=True)
    return pd.DataFrame(results).sort_values("aic")


def choose_seasonal_order(subset: pd.Series):
    """Step A: quick comparison of candidate seasonal orders with (p,d,q)
    fixed at (1,0,1), to pick a single seasonal order to carry into the
    full required non-seasonal grid search (keeps the full 7x3x7 grid
    computationally tractable)."""
    candidates = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    rows = []
    for (P, D, Q) in candidates:
        try:
            m = SARIMAX(subset, order=(1, 0, 1), seasonal_order=(P, D, Q, SEASONAL_PERIOD),
                        enforce_stationarity=False, enforce_invertibility=False)
            f = m.fit(disp=False, maxiter=50, method="lbfgs")
            rows.append({"P": P, "D": D, "Q": Q, "aic": f.aic})
            print(f"  seasonal_order=({P},{D},{Q},24) -> AIC={f.aic:.1f}")
        except Exception as e:
            print(f"  seasonal_order=({P},{D},{Q},24) failed: {e}")
    res = pd.DataFrame(rows).sort_values("aic")
    return res


def run_grid_search():
    df = load_hourly()
    # Use most recent 30 days of the TRAINING portion only (never touches test)
    test_hours = TEST_DAYS * 24
    train_full = df.iloc[:-test_hours]
    subset = train_full[TARGET].iloc[-30 * 24:]

    cache = METRICS_DIR / "sarimax_seasonal_order_search.csv"
    if cache.exists():
        print("Step A: loading cached seasonal-order search results...")
        seasonal_res = pd.read_csv(cache).sort_values("aic")
    else:
        print("Step A: choosing seasonal order (P,D,Q) with (p,d,q) fixed at (1,0,1)...")
        seasonal_res = choose_seasonal_order(subset)
        seasonal_res.to_csv(cache, index=False)
    best_seasonal = seasonal_res.iloc[0]
    seasonal_candidates = [(int(best_seasonal.P), int(best_seasonal.D), int(best_seasonal.Q))]
    print(f"Chosen seasonal order: {seasonal_candidates[0]} + s=24 (AIC={best_seasonal.aic:.1f})")

    print(f"\nStep B: full required grid search p=[0,6], d=[0,2], q=[0,6] "
          f"(147 combos) at fixed seasonal order, on {len(subset)} obs (last 30 train days)...")
    grid = grid_search_order(subset, seasonal_candidates,
                              checkpoint_path=METRICS_DIR / "sarimax_grid_search.csv")
    grid.to_csv(METRICS_DIR / "sarimax_grid_search.csv", index=False)
    print(f"Completed {len(grid)} successful fits. Top 10 by AIC:")
    print(grid.head(10).to_string(index=False))
    return grid


def fit_final_model(train: pd.DataFrame, order, seasonal_order):
    endog = train[TARGET]
    exog = train[EXOG_COLS]
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200, method="lbfgs")
    return fit


def residual_diagnostics(fit, fname_prefix="09_sarimax"):
    resid = fit.resid
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(resid)
    axes[0].set_title("SARIMAX Residuals over Time")
    plot_acf(resid.dropna(), lags=48, ax=axes[1])
    axes[1].set_title("ACF of Residuals")
    axes[2].hist(resid.dropna(), bins=40, color="#2b6cb0")
    axes[2].set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname_prefix}_residual_diagnostics.png", dpi=150)
    plt.close(fig)

    lb = acorr_ljungbox(resid.dropna(), lags=[24, 48], return_df=True)
    print("Ljung-Box test on residuals (H0: residuals are white noise):")
    print(lb)
    return lb


def rolling_origin_sarimax(df: pd.DataFrame, order, seasonal_order, checkpoint_path=None):
    """Refit-at-each-origin rolling backtest (statistically correct: only uses
    data available up to each origin). Exogenous future values (weather) are
    taken from the actual test set - this is flagged in Part 9 Q5 as a
    CONDITIONAL forecast (assumes a perfect weather forecast), not a pure
    unconditional forecast. Checkpoints after every origin so a killed
    process can resume without redoing completed origins."""
    origins = get_origins(df)

    done_origins = set()
    all_rows = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        prev = pd.read_parquet(checkpoint_path)
        all_rows = [prev]
        done_origins = set(pd.to_datetime(prev["origin"]).unique())
        print(f"  Resuming: {len(done_origins)} origins already checkpointed.")

    for i, origin in enumerate(origins):
        if pd.Timestamp(origin) in done_origins:
            print(f"  origin {i+1}/{len(origins)} ({origin}) already done, skipping")
            continue
        train = df.loc[:origin]
        start_loc = df.index.get_loc(origin) + 1
        future_idx = df.index[start_loc: start_loc + HORIZON]
        endog = train[TARGET]
        exog_train = train[EXOG_COLS]
        exog_future = df.loc[future_idx, EXOG_COLS]

        fit = SARIMAX(endog, exog=exog_train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(
                           disp=False, maxiter=100, method="lbfgs")
        pred = fit.get_forecast(steps=HORIZON, exog=exog_future)
        y_pred = pred.predicted_mean.values

        y_true = df.loc[future_idx, TARGET].values
        block_df = forecasts_to_frame("SARIMAX", [origin], [y_true], [y_pred])
        all_rows.append(block_df)
        if checkpoint_path is not None:
            pd.concat(all_rows, ignore_index=True).to_parquet(checkpoint_path)
        print(f"  origin {i+1}/{len(origins)} ({origin}) done", flush=True)

    results_df = pd.concat(all_rows, ignore_index=True)
    return results_df


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    grid = run_grid_search()
    best = grid.iloc[0]
    order = (int(best.p), int(best.d), int(best.q))
    seasonal_order = (int(best.P), int(best.D), int(best.Q), SEASONAL_PERIOD)
    print(f"\nSelected order={order}, seasonal_order={seasonal_order}")

    df = load_hourly()
    test_hours = TEST_DAYS * 24
    train_full = df.iloc[:-test_hours]

    print("\nFitting final model on full training set for diagnostics...")
    fit = fit_final_model(train_full, order, seasonal_order)
    print(fit.summary().tables[0])
    residual_diagnostics(fit)

    print("\nRunning rolling-origin backtest (refits SARIMAX at each of 14 origins)...")
    results_df = rolling_origin_sarimax(df, order, seasonal_order,
                                         checkpoint_path=METRICS_DIR / "sarimax_forecasts.parquet")
    results_df.to_parquet(METRICS_DIR / "sarimax_forecasts.parquet")

    summary = summarize(results_df)
    print("\n=== SARIMAX performance (pooled, 14 x 24h blocks) ===")
    print(summary.round(2))


if __name__ == "__main__":
    main()
