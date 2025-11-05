
import pandas as pd
import numpy as np

# LOAD RAW DATA
print("🔹 Loading raw cloud logs...")
df = pd.read_csv("cloud_logs_raw.csv")

print(f"Initial shape: {df.shape}")
print("Columns:", list(df.columns))

# STEP 2: REMOVE DUPLICATES
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# STEP 3: HANDLE MISSING VALUES
print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

# Drop rows missing critical fields
df = df.dropna(subset=["timestamp", "service", "region"])
df["error_code"].fillna("", inplace=True)

# CONVERT & VALIDATE TIMESTAMPS
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)

# STANDARDIZE CATEGORICAL FIELDS
df["service"] = df["service"].str.upper().str.strip()
df["region"] = df["region"].str.lower().str.strip()
df["env"] = df["env"].str.lower().str.strip()
df["status"] = df["status"].str.upper().str.strip()

# Keep only known values
valid_services = ["EC2", "RDS", "LAMBDA"]
valid_regions = ["us-east-1", "us-west-2"]
df = df[df["service"].isin(valid_services)]
df = df[df["region"].isin(valid_regions)]


# FILTER MALFORMED RECORDS
df = df[df["latency_ms"].between(0, 5000)]
df = df[df["cpu_utilization"].between(0, 1)]

# SAVE CLEANED DATA (Intermediate)
df.to_csv("cloud_logs_clean.csv", index=False)
print("✅ Intermediate cleaned file saved as 'cloud_logs_clean.csv'")

# FINAL STRUCTURING
# Add derived fields for later analysis
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

# Reorder columns
df = df[
    [
        "timestamp",
        "date",
        "hour",
        "service",
        "region",
        "env",
        "instance_id",
        "latency_ms",
        "cpu_utilization",
        "status",
        "error_code",
        "message",
    ]
]

# SUMMARY SNAPSHOT
print("\n✅ Cleaning complete!")
print(f"Final shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print("\nService breakdown:")
print(df["service"].value_counts())
print("\nStatus distribution:")
print((df["status"].value_counts(normalize=True) * 100).round(2))

# SAVE FINAL STRUCTURED DATA
df.to_csv("cloud_logs_structured.csv", index=False)
print("✅ Final structured dataset saved as 'cloud_logs_structured.csv'")
