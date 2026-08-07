"""
eda.py
======
Exploratory data analysis and stationarity testing for hourly Appliances
energy demand: initial plots, seasonal decomposition, ADF/KPSS tests,
ACF/PACF plots, and differencing if required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "energy_hourly.parquet"


def load_hourly():
    return pd.read_parquet(PROCESSED)


def plot_series(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(df.index, df["Appliances"], lw=0.7, color="#1f5fa6")
    axes[0].set_title("Hourly Appliance Energy Use (Wh) — Full Period")
    axes[0].set_ylabel("Wh / hour")
    axes[0].set_xlabel("Date")

    one_week = df.loc[df.index[500]:df.index[500] + pd.Timedelta(days=14)]
    axes[1].plot(one_week.index, one_week["Appliances"], color="#c0392b")
    axes[1].set_title("Hourly Appliance Energy Use — 14-Day Zoom (shows daily/weekly pattern)")
    axes[1].set_ylabel("Wh / hour")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_series_overview.png", dpi=150)
    plt.close(fig)


def plot_decomposition(df: pd.DataFrame):
    # Daily seasonality: period = 24 hours
    result = seasonal_decompose(df["Appliances"], model="additive", period=24, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Additive Seasonal Decomposition (period = 24h)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_decomposition_daily.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return result


def plot_weekly_pattern(df: pd.DataFrame):
    tmp = df.copy()
    tmp["hour"] = tmp.index.hour
    tmp["dayofweek"] = tmp.index.dayofweek
    pivot = tmp.pivot_table(index="hour", columns="dayofweek", values="Appliances", aggfunc="mean")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = day_names

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=col, marker="o", ms=3)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean Appliance Use (Wh)")
    ax.set_title("Average Hourly Profile by Day of Week")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_weekly_hourly_profile.png", dpi=150)
    plt.close(fig)


def adf_test(series: pd.Series, label: str):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"\n--- ADF test: {label} ---")
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4g}")
    print(f"# lags used:   {result[2]}")
    for k, v in result[4].items():
        print(f"  critical value ({k}): {v:.4f}")
    verdict = "STATIONARY (reject H0)" if result[1] < 0.05 else "NON-STATIONARY (fail to reject H0)"
    print(f"Verdict: {verdict}")
    return result


def kpss_test(series: pd.Series, label: str):
    stat, p, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"\n--- KPSS test: {label} ---")
    print(f"KPSS statistic: {stat:.4f}")
    print(f"p-value:        {p:.4g}")
    for k, v in crit.items():
        print(f"  critical value ({k}): {v:.4f}")
    verdict = "NON-STATIONARY (reject H0)" if p < 0.05 else "STATIONARY (fail to reject H0)"
    print(f"Verdict: {verdict}")
    return stat, p, lags, crit


def plot_acf_pacf(series: pd.Series, label: str, lags: int, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {label}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — {label}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_hourly()

    print("Summary statistics for Appliances (Wh/hour):")
    print(df["Appliances"].describe())

    plot_series(df)
    decomp = plot_decomposition(df)
    plot_weekly_pattern(df)

    plot_acf_pacf(df["Appliances"], "Appliances (raw, hourly)", lags=72, fname="04_acf_pacf_raw.png")

    # Stationarity tests on the raw series
    adf_test(df["Appliances"], "Appliances (raw)")
    kpss_test(df["Appliances"], "Appliances (raw)")

    # First difference (removes stochastic trend)
    diff1 = df["Appliances"].diff().dropna()
    adf_test(diff1, "Appliances (1st difference)")
    kpss_test(diff1, "Appliances (1st difference)")
    plot_acf_pacf(diff1, "Appliances (1st difference)", lags=72, fname="05_acf_pacf_diff1.png")

    # Seasonal (24h) difference
    seasonal_diff = df["Appliances"].diff(24).dropna()
    adf_test(seasonal_diff, "Appliances (24h seasonal difference)")
    kpss_test(seasonal_diff, "Appliances (24h seasonal difference)")
    plot_acf_pacf(seasonal_diff, "Appliances (24h seasonal diff)", lags=72, fname="06_acf_pacf_seasonal_diff.png")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
