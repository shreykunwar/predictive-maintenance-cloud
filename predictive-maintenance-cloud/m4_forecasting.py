"""
Milestone 4 - Point 1
Time-Series Forecasting for latency, CPU, and anomaly density
Inputs:
  - cloud_logs_features.csv         (from Milestone 2)
  - cloud_logs_anomalies.csv        (from Milestone 3)
Outputs (written into ./forecasts/):
  - <metric>__<service>__<env>__forecast_7d.csv
  - <metric>__<service>__<env>__forecast_30d.csv
  - PNG plots for each forecast horizon

Author: Shrey Kunwar
Date: Nov 2025
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Try Prophet first; fall back to ARIMA if unavailable
USE_PROPHET = True
try:
    from prophet import Prophet  # pip install prophet
except Exception:
    USE_PROPHET = False
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # pip install statsmodels
import matplotlib.pyplot as plt


# ------------------------------
# Helpers
# ------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def to_daily(df, ts_col, value_cols, group_cols):
    """
    Resample to daily by group. For continuous metrics take mean; for rates compute mean (already a rate).
    """
    out = []
    for keys, g in df.groupby(group_cols):
        g = g.set_index(ts_col).sort_index()
        agg = {}
        for c in value_cols:
            agg[c] = "mean"
        daily = g.resample("D").agg(agg).reset_index()
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                daily[k] = v
        else:
            daily[group_cols[0]] = keys
        out.append(daily)
    return pd.concat(out, ignore_index=True)

def fit_predict_prophet(series_df, periods, freq="D"):
    """
    Expects columns: ds, y
    Returns forecast dataframe with columns: ds, yhat, yhat_lower, yhat_upper
    """
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
    m.fit(series_df)
    future = m.make_future_dataframe(periods=periods, freq=freq, include_history=True)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst

def fit_predict_arima(series_df, periods):
    """
    Simple SARIMAX fallback. Returns dataframe with columns: ds, yhat
    """
    y = series_df.set_index("ds")["y"].asfreq("D").interpolate()
    # Basic seasonal guess (weekly)
    model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    pred = res.get_forecast(steps=periods)
    yhat = pred.predicted_mean
    conf = pred.conf_int()
    out_hist = pd.DataFrame({"ds": y.index, "yhat": y.values})
    out_fut = pd.DataFrame({
        "ds": future_index,
        "yhat": yhat.values,
        "yhat_lower": conf.iloc[:, 0].values,
        "yhat_upper": conf.iloc[:, 1].values
    })
    fcst = pd.concat([out_hist, out_fut], ignore_index=True)
    if "yhat_lower" not in fcst.columns:  # safety
        fcst["yhat_lower"] = np.nan
        fcst["yhat_upper"] = np.nan
    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def run_forecast(series_df, metric_name, svc, env, horizon_days, outdir):
    """
    series_df: columns ds, y
    """
    if len(series_df) < 14:
        print(f"Skipping {metric_name} {svc} {env}: not enough data points ({len(series_df)}).")
        return None

    try:
        if USE_PROPHET:
            fcst = fit_predict_prophet(series_df, periods=horizon_days, freq="D")
        else:
            fcst = fit_predict_arima(series_df, periods=horizon_days)
    except Exception as e:
        print(f"Model failed for {metric_name} {svc} {env} ({horizon_days}d): {e}")
        return None

    # Save CSV
    base = f"{metric_name}__{svc}__{env}__forecast_{horizon_days}d"
    csv_path = os.path.join(outdir, base + ".csv")
    fcst.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(9, 4.8))
    # history portion
    hist = fcst[fcst["ds"] <= series_df["ds"].max()]
    fut = fcst[fcst["ds"] > series_df["ds"].max()]
    plt.plot(hist["ds"], hist["yhat"], label="history")
    plt.plot(fut["ds"], fut["yhat"], label=f"forecast ({horizon_days}d)")
    if "yhat_lower" in fcst.columns and fcst["yhat_lower"].notna().any():
        plt.fill_between(fut["ds"], fut["yhat_lower"], fut["yhat_upper"], alpha=0.2, label="conf. interval")
    plt.title(f"{metric_name} forecast – {svc}/{env}")
    plt.xlabel("Date")
    plt.ylabel(metric_name)
    plt.legend()
    png_path = os.path.join(outdir, base + ".png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    return fcst


# ------------------------------
# Load inputs
# ------------------------------
features_path = "cloud_logs_features.csv"
anoms_path = "cloud_logs_anomalies.csv"

features = pd.read_csv(features_path, parse_dates=["timestamp"])
anoms = pd.read_csv(anoms_path, parse_dates=["timestamp"])

# Keep needed columns
features = features[[
    "timestamp", "service", "env", "latency_ms", "cpu_utilization"
]].copy()

# Build anomaly rate per day per service/env
anoms["is_anom"] = anoms["combined_anomaly"].astype(int)
anom_daily = (
    anoms
    .assign(date=anoms["timestamp"].dt.date)
    .groupby(["service", "env", "date"])
    .agg(anom_rate=("is_anom", "mean"))
    .reset_index()
)
anom_daily["ds"] = pd.to_datetime(anom_daily["date"])
anom_daily = anom_daily.drop(columns=["date"])


# ------------------------------
# Aggregate to daily for latency and CPU
# ------------------------------
features["ds"] = features["timestamp"].dt.floor("D")
lat_daily = (
    features.groupby(["service", "env", "ds"])["latency_ms"].mean().reset_index()
)
cpu_daily = (
    features.groupby(["service", "env", "ds"])["cpu_utilization"].mean().reset_index()
)

# ------------------------------
# Forecast configs
# ------------------------------
outdir = "forecasts"
ensure_dir(outdir)

services = sorted(features["service"].unique())
envs = sorted(features["env"].unique())
horizons = [7, 30]  # days

print(f"Using model: {'Prophet' if USE_PROPHET else 'ARIMA'}")
print(f"Services: {services}")
print(f"Envs    : {envs}")
print("----")

# ------------------------------
# Run forecasts for each metric / service / env
# ------------------------------
for svc in services:
    for env in envs:
        # 1) Latency
        df_lat = lat_daily[(lat_daily["service"] == svc) & (lat_daily["env"] == env)]
        if len(df_lat) > 0:
            series = df_lat.rename(columns={"ds": "ds", "latency_ms": "y"})[["ds", "y"]].dropna()
            for h in horizons:
                run_forecast(series, "latency_ms", svc, env, h, outdir)

        # 2) CPU utilization
        df_cpu = cpu_daily[(cpu_daily["service"] == svc) & (cpu_daily["env"] == env)]
        if len(df_cpu) > 0:
            series = df_cpu.rename(columns={"ds": "ds", "cpu_utilization": "y"})[["ds", "y"]].dropna()
            for h in horizons:
                run_forecast(series, "cpu_utilization", svc, env, h, outdir)

        # 3) Anomaly rate
        df_ar = anom_daily[(anom_daily["service"] == svc) & (anom_daily["env"] == env)]
        if len(df_ar) > 0:
            series = df_ar.rename(columns={"anom_rate": "y"})[["ds", "y"]].dropna()
            for h in horizons:
                run_forecast(series, "anomaly_rate", svc, env, h, outdir)

print("All forecasts generated.")
