"""
Predictive Maintenance Project - Milestone 1
QC Summary Report Generator
Author: Shrey Kunwar
Date: October 2025
"""

import pandas as pd

# Load files
raw = pd.read_csv("cloud_logs_raw.csv")
clean = pd.read_csv("cloud_logs_clean.csv")
structured = pd.read_csv("cloud_logs_structured.csv")

summary = []

summary.append("========== QC SUMMARY REPORT (MILESTONE 1) ==========\n")

summary.append(f"Raw file records: {len(raw):,}")
summary.append(f"Clean file records: {len(clean):,}")
summary.append(f"Structured file records: {len(structured):,}")
summary.append(f"Records removed during cleaning: {len(raw) - len(clean):,}\n")

# Missing values check
summary.append("Missing Values in Structured Data:")
missing = structured.isnull().sum()
has_missing = False
for col, val in missing.items():
    if val > 0:
        pct = round((val / len(structured)) * 100, 2)
        summary.append(f"  {col}: {val} missing ({pct}%)")
        has_missing = True
if not has_missing:
    summary.append("  ✅ No missing values found.\n")

# Date range
structured["timestamp"] = pd.to_datetime(structured["timestamp"], errors="coerce")
date_min = structured["timestamp"].min()
date_max = structured["timestamp"].max()
summary.append(f"Date Range: {date_min} → {date_max}\n")

# Service and status distribution
service_counts = structured["service"].value_counts()
status_counts = structured["status"].value_counts(normalize=True) * 100

summary.append("Service Distribution:")
for svc, cnt in service_counts.items():
    summary.append(f"  {svc}: {cnt:,} logs")

summary.append("\nStatus Distribution:")
for st, pct in status_counts.items():
    summary.append(f"  {st}: {pct:.2f}%")

# Save report (UTF-8 to fix encoding issue)
report_path = "QC_Milestone1_Report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary))
print(f"\n✅ QC summary exported as '{report_path}'")
