"""
Milestone 3 
Cross-service/environment comparison
Anomaly export and integration for next milestone
Author: Shrey Kunwar
Date: November 2025
"""

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
anoms_path = "cloud_logs_anomalies.csv"        # output from anomaly_detection_model.py
features_path = "cloud_logs_features.csv"      # used for hour extraction if needed

df = pd.read_csv(anoms_path, parse_dates=["timestamp"])

# Ensure required columns exist
required_cols = {
    "timestamp", "service", "env", "latency_ms", "cpu_utilization", "status",
    "anomaly_iforest", "anomaly_zscore_latency", "anomaly_zscore_cpu", "combined_anomaly"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {anoms_path}: {sorted(missing)}. "
                     f"Run anomaly_detection_model.py before this step.")

# Derive time features if not present
if "hour" not in df.columns:
    df["hour"] = df["timestamp"].dt.hour

# --------------------------------------------------------------------------------------
# 3) Cross-service/environment comparison
# --------------------------------------------------------------------------------------

# Group-level summary stats
grp = df.groupby(["service", "env"])

summary = grp.agg(
    total_logs=("combined_anomaly", "size"),
    anomaly_rate=("combined_anomaly", "mean"),
    iforest_rate=("anomaly_iforest", "mean"),
    z_latency_rate=("anomaly_zscore_latency", "mean"),
    z_cpu_rate=("anomaly_zscore_cpu", "mean"),
    latency_p50=("latency_ms", "median"),
    latency_p95=("latency_ms", lambda x: np.percentile(x, 95)),
    cpu_p50=("cpu_utilization", "median"),
    cpu_p95=("cpu_utilization", lambda x: np.percentile(x, 95)),
).reset_index()

# Convert rates to percentages for readability
for col in ["anomaly_rate", "iforest_rate", "z_latency_rate", "z_cpu_rate"]:
    summary[col] = (summary[col] * 100).round(2)

summary["latency_p50"] = summary["latency_p50"].round(2)
summary["latency_p95"] = summary["latency_p95"].round(2)
summary["cpu_p50"] = summary["cpu_p50"].round(3)
summary["cpu_p95"] = summary["cpu_p95"].round(3)

summary_out = "anomaly_group_summary.csv"
summary.to_csv(summary_out, index=False)
print(f"Saved group comparison: {summary_out}")

# Hour-of-day peaks per service/env (which hours show the highest anomaly rate)
hourly = (
    df.groupby(["service", "env", "hour"])["combined_anomaly"]
    .mean()
    .reset_index()
    .rename(columns={"combined_anomaly": "anomaly_rate"})
)
hourly["anomaly_rate"] = (hourly["anomaly_rate"] * 100).round(2)

# For convenience, keep top 3 hours per service/env by anomaly rate
hourly["rank"] = hourly.groupby(["service", "env"])["anomaly_rate"].rank(ascending=False, method="first")
hourly_peaks = hourly[hourly["rank"] <= 3].sort_values(["service", "env", "rank"])
hourly_out = "anomaly_hourly_peaks.csv"
hourly_peaks.to_csv(hourly_out, index=False)
print(f"Saved hourly anomaly peaks: {hourly_out}")

# --------------------------------------------------------------------------------------
# 4) Consolidated anomaly export (with a severity score)
# --------------------------------------------------------------------------------------

# Simple severity score using z-scores of latency and CPU
def zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma

df["z_latency"] = zscore(df["latency_ms"]).abs()
df["z_cpu"] = zscore(df["cpu_utilization"]).abs()
df["severity"] = ((df["z_latency"] + df["z_cpu"]) / 2).round(3)

# Keep only anomalies for the consolidated export
anomaly_cols = [
    "timestamp", "service", "env", "latency_ms", "cpu_utilization", "status",
    "anomaly_iforest", "anomaly_zscore_latency", "anomaly_zscore_cpu", "combined_anomaly",
    "severity", "hour"
]
consolidated = df[df["combined_anomaly"] == 1][anomaly_cols].sort_values(["severity", "timestamp"], ascending=[False, True])

consolidated_out = "cloud_anomalies_consolidated.csv"
consolidated.to_csv(consolidated_out, index=False)
print(f"Saved consolidated anomalies: {consolidated_out}")

# Top 50 highest-severity anomalies (handy for review/next milestone)
top50 = consolidated.head(50)
top50_out = "top50_high_severity_anomalies.csv"
top50.to_csv(top50_out, index=False)
print(f"Saved top 50 anomalies: {top50_out}")

print("Done.")
