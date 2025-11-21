"""
Predictive Maintenance Project - Milestone 3
Anomaly Detection & Model Evaluation
Author: Shrey Kunwar
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
print("Loading feature dataset...")
df = pd.read_csv("cloud_logs_features.csv", parse_dates=["timestamp"])

# Focus on production environment for anomaly detection
df = df[df["env"] == "prod"].copy()

# Isolation Forest Model
features = ["latency_ms", "cpu_utilization", "error_flag", "rolling_latency_3h", "rolling_cpu_3h"]
df = df.dropna(subset=features)
X = df[features]

print("Training Isolation Forest...")
model = IsolationForest(contamination=0.02, random_state=42)
df["anomaly_iforest"] = model.fit_predict(X)

# Map predictions (-1 = anomaly, 1 = normal)
df["anomaly_iforest"] = df["anomaly_iforest"].map({1: 0, -1: 1})

# Z-score Based Detection
def detect_zscore(series, threshold=3.0):
    z = (series - series.mean()) / series.std()
    return (abs(z) > threshold).astype(int)

df["anomaly_zscore_latency"] = detect_zscore(df["latency_ms"])
df["anomaly_zscore_cpu"] = detect_zscore(df["cpu_utilization"])

# Combine Anomalies
df["combined_anomaly"] = (
    (df["anomaly_iforest"] + df["anomaly_zscore_latency"] + df["anomaly_zscore_cpu"]) > 0
).astype(int)

# Model Evaluation
# For evaluation, assume ERROR logs are actual anomalies
df["true_label"] = (df["status"] == "ERROR").astype(int)

precision = precision_score(df["true_label"], df["combined_anomaly"])
recall = recall_score(df["true_label"], df["combined_anomaly"])
f1 = f1_score(df["true_label"], df["combined_anomaly"])

print("\nModel Evaluation Summary:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# Save Results
df.to_csv("cloud_logs_anomalies.csv", index=False)
print("\nAnomaly detection results saved as 'cloud_logs_anomalies.csv'")
