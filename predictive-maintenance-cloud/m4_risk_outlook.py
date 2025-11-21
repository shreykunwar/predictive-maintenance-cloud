"""
Milestone 4 - Point 2
Service- and Environment-Level Risk Outlook

"""

import os
import re
import pandas as pd
import numpy as np

FORECAST_DIR = "forecasts"

# Weights for composite risk score (you can tweak these)
W_LATENCY = 0.4
W_CPU = 0.3
W_ANOM = 0.3


def list_forecast_files():
    if not os.path.isdir(FORECAST_DIR):
        raise FileNotFoundError(f"Forecast directory '{FORECAST_DIR}' not found. "
                                f"Run m4_forecasting.py first.")
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".csv")]
    return files


def parse_filename(fname):
    """
    Expected format:
      <metric>__<service>__<env>__forecast_<horizon>d.csv
    Example:
      latency_ms__EC2__prod__forecast_7d.csv
    """
    base = os.path.basename(fname)
    m = re.match(r"(.+)__([^_]+)__([^_]+)__forecast_(\d+)d\.csv", base)
    if not m:
        return None
    metric, svc, env, horizon = m.groups()
    return {
        "metric": metric,
        "service": svc,
        "env": env,
        "horizon": int(horizon),
        "file": fname,
    }


def normalize_series(s):
    """
    Min-max normalize a series to [0, 1].
    If constant or NaN, returns zeros.
    """
    s = s.astype(float)
    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    min_val, max_val = s.min(), s.max()
    if max_val == min_val:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_val) / (max_val - min_val)


def build_risk_for_horizon(horizon_days):
    """
    Build daily risk outlook for a given forecast horizon (e.g., 7 or 30 days).
    Returns:
      - daily_df: per-day risk scores
      - summary_df: per service/env summary
    """
    files = list_forecast_files()
    meta = [parse_filename(f) for f in files]
    meta = [m for m in meta if m is not None and m["horizon"] == horizon_days]

    if not meta:
        raise ValueError(f"No forecast files found for horizon {horizon_days} days.")

    # Group metadata by (service, env)
    groups = {}
    for m in meta:
        key = (m["service"], m["env"])
        groups.setdefault(key, []).append(m)

    daily_records = []

    for (svc, env), entries in groups.items():
        # Load metric forecasts if present
        df_lat = df_cpu = df_anom = None

        for e in entries:
            path = os.path.join(FORECAST_DIR, e["file"])
            df = pd.read_csv(path, parse_dates=["ds"])
            metric = e["metric"]

            # We only care about the forecast period; the file includes history + forecast.
            # Take the last `horizon_days` rows as the forecast horizon.
            df = df.sort_values("ds").tail(horizon_days).reset_index(drop=True)

            if metric == "latency_ms":
                df_lat = df.rename(columns={"yhat": "latency_hat"})
            elif metric == "cpu_utilization":
                df_cpu = df.rename(columns={"yhat": "cpu_hat"})
            elif metric == "anomaly_rate":
                df_anom = df.rename(columns={"yhat": "anom_hat"})

        # Build combined frame
        # Start with dates from whichever metric exists
        base = None
        for df_metric in [df_lat, df_cpu, df_anom]:
            if df_metric is not None:
                base = df_metric[["ds"]].copy()
                break

        if base is None:
            # No usable forecasts for this service/env
            continue

        base["service"] = svc
        base["env"] = env

        if df_lat is not None:
            base = base.merge(df_lat[["ds", "latency_hat"]], on="ds", how="left")
        else:
            base["latency_hat"] = np.nan

        if df_cpu is not None:
            base = base.merge(df_cpu[["ds", "cpu_hat"]], on="ds", how="left")
        else:
            base["cpu_hat"] = np.nan

        if df_anom is not None:
            base = base.merge(df_anom[["ds", "anom_hat"]], on="ds", how="left")
        else:
            base["anom_hat"] = np.nan

        # Normalize metrics for risk calculation
        base["latency_norm"] = normalize_series(base["latency_hat"])
        base["cpu_norm"] = normalize_series(base["cpu_hat"])
        base["anom_norm"] = normalize_series(base["anom_hat"])

        # Composite risk score
        base["risk_score"] = (
            W_LATENCY * base["latency_norm"] +
            W_CPU * base["cpu_norm"] +
            W_ANOM * base["anom_norm"]
        ).round(3)

        daily_records.append(base)

    if not daily_records:
        raise ValueError(f"No valid forecasts could be combined for horizon {horizon_days} days.")

    daily_df = pd.concat(daily_records, ignore_index=True)

    # Build summary per service/env
    summary = (
        daily_df
        .groupby(["service", "env"])
        .agg(
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            median_risk=("risk_score", "median"),
        )
        .reset_index()
    )

    # Identify the worst day for each service/env
    idx = daily_df.groupby(["service", "env"])["risk_score"].idxmax()
    worst_days = (
        daily_df.loc[idx, ["service", "env", "ds", "risk_score"]]
        .rename(columns={"ds": "worst_day", "risk_score": "worst_day_risk"})
    )

    summary = summary.merge(worst_days, on=["service", "env"], how="left")

    # Round for readability
    for col in ["avg_risk", "max_risk", "median_risk", "worst_day_risk"]:
        summary[col] = summary[col].round(3)

    return daily_df, summary


def main():
    # 7-day outlook
    daily_7d, summary_7d = build_risk_for_horizon(7)
    daily_7d = daily_7d.sort_values(["service", "env", "ds"])
    summary_7d = summary_7d.sort_values(["avg_risk"], ascending=False)

    daily_7d.to_csv("risk_outlook_daily_7d.csv", index=False)
    summary_7d.to_csv("risk_outlook_summary_7d.csv", index=False)
    print("Saved: risk_outlook_daily_7d.csv")
    print("Saved: risk_outlook_summary_7d.csv")

    # 30-day outlook
    daily_30d, summary_30d = build_risk_for_horizon(30)
    daily_30d = daily_30d.sort_values(["service", "env", "ds"])
    summary_30d = summary_30d.sort_values(["avg_risk"], ascending=False)

    daily_30d.to_csv("risk_outlook_daily_30d.csv", index=False)
    summary_30d.to_csv("risk_outlook_summary_30d.csv", index=False)
    print("Saved: risk_outlook_daily_30d.csv")
    print("Saved: risk_outlook_summary_30d.csv")


if __name__ == "__main__":
    main()
