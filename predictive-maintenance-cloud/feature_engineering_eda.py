"""
Predictive Maintenance Project - Milestone 2
Feature Engineering & EDA
Author: Shrey Kunwar
Date: October 2025
"""

import pandas as pd
import numpy as np

# Load cleaned structured logs
df = pd.read_csv("cloud_logs_structured.csv", parse_dates=["timestamp"])

# Error rate per service/environment
error_map = {"ERROR": 1, "WARN": 0.5, "OK": 0}
df["error_flag"] = df["status"].map(error_map)
error_summary = (
    df.groupby(["service", "env"])["error_flag"]
    .mean()
    .reset_index()
    .rename(columns={"error_flag": "error_rate"})
)

# Rolling metrics (3h & 6h windows)
df = df.sort_values("timestamp")
df["rolling_latency_3h"] = df["latency_ms"].rolling(window=12, min_periods=1).mean()
df["rolling_latency_6h"] = df["latency_ms"].rolling(window=24, min_periods=1).mean()
df["rolling_cpu_3h"] = df["cpu_utilization"].rolling(window=12, min_periods=1).mean()
df["rolling_cpu_6h"] = df["cpu_utilization"].rolling(window=24, min_periods=1).mean()

# Change rates
df["latency_change"] = df["latency_ms"].pct_change().round(3)
df["cpu_change"] = df["cpu_utilization"].pct_change().round(3)

# Merge error rates
df = df.merge(error_summary, on=["service", "env"], how="left")

# Save feature dataset
df.to_csv("cloud_logs_features.csv", index=False)
print("✅ Feature dataset saved as 'cloud_logs_features.csv'")

# Correlation summary
corr = df[["latency_ms", "cpu_utilization", "error_flag"]].corr()
print("\nCorrelation Matrix:\n", corr)
corr.to_csv("feature_correlation_summary.csv", index=True)
print("✅ Correlation summary saved as 'feature_correlation_summary.csv'")
