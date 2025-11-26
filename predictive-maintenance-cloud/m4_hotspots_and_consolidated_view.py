"""
Milestone 4 - Remaining Points
Point 3: Hotspot Identification
Point 4: Consolidated Risk View

Definitions:
  - Hotspot: days with risk_score above a dynamic threshold per horizon
             (here: risk_score >= 0.7, or top N by service/env)
  - Risk Tier: High / Medium / Low based on avg_risk and max_risk

Author: Shrey Kunwar
Date: November 2025
"""

import pandas as pd
import numpy as np


# -----------------------------
# Helper functions
# -----------------------------

def classify_risk_tier(avg_risk, max_risk):
    """
    Classify a service/env into a qualitative risk tier.
    You can tune the thresholds based on your findings.
    """
    if max_risk >= 0.75 or avg_risk >= 0.6:
        return "High"
    elif max_risk >= 0.4 or avg_risk >= 0.3:
        return "Medium"
    else:
        return "Low"


def identify_hotspots(daily_df, threshold=0.7, top_n_per_group=3):
    """
    Identify hotspots as days where:
      - risk_score >= threshold OR
      - among the top N risk days per (service, env)

    Returns a filtered dataframe of hotspots.
    """
    df = daily_df.copy()
    df["is_above_threshold"] = df["risk_score"] >= threshold

    # Rank days by risk per service/env
    df["risk_rank"] = df.groupby(["service", "env"])["risk_score"].rank(
        ascending=False, method="first"
    )

    df["is_top_n"] = df["risk_rank"] <= top_n_per_group

    hotspots = df[(df["is_above_threshold"]) | (df["is_top_n"])].copy()

    # Clean up columns for output
    hotspots = hotspots[
        ["ds", "service", "env", "risk_score", "latency_hat", "cpu_hat", "anom_hat",
         "latency_norm", "cpu_norm", "anom_norm", "is_above_threshold", "risk_rank"]
    ].sort_values(["service", "env", "ds"])

    return hotspots


# -----------------------------
# Load risk outlook data
# -----------------------------

daily_7d = pd.read_csv("risk_outlook_daily_7d.csv", parse_dates=["ds"])
summary_7d = pd.read_csv("risk_outlook_summary_7d.csv", parse_dates=["worst_day"])

daily_30d = pd.read_csv("risk_outlook_daily_30d.csv", parse_dates=["ds"])
summary_30d = pd.read_csv("risk_outlook_summary_30d.csv", parse_dates=["worst_day"])


# -----------------------------
# Point 3: Hotspot Identification
# -----------------------------

# Identify hotspots in 7-day horizon
hotspots_7d = identify_hotspots(daily_7d, threshold=0.7, top_n_per_group=3)
hotspots_7d.to_csv("hotspots_7d.csv", index=False)
print("Saved: hotspots_7d.csv")

# Identify hotspots in 30-day horizon
hotspots_30d = identify_hotspots(daily_30d, threshold=0.7, top_n_per_group=5)
hotspots_30d.to_csv("hotspots_30d.csv", index=False)
print("Saved: hotspots_30d.csv")


# -----------------------------
# Point 4: Consolidated Risk View
# -----------------------------

# Add risk tiers to summaries
summary_7d["risk_tier_7d"] = summary_7d.apply(
    lambda r: classify_risk_tier(r["avg_risk"], r["max_risk"]), axis=1
)
summary_30d["risk_tier_30d"] = summary_30d.apply(
    lambda r: classify_risk_tier(r["avg_risk"], r["max_risk"]), axis=1
)

# Rename columns to reflect horizon
summary_7d_renamed = summary_7d.rename(
    columns={
        "avg_risk": "avg_risk_7d",
        "max_risk": "max_risk_7d",
        "median_risk": "median_risk_7d",
        "worst_day": "worst_day_7d",
        "worst_day_risk": "worst_day_risk_7d",
    }
)

summary_30d_renamed = summary_30d.rename(
    columns={
        "avg_risk": "avg_risk_30d",
        "max_risk": "max_risk_30d",
        "median_risk": "median_risk_30d",
        "worst_day": "worst_day_30d",
        "worst_day_risk": "worst_day_risk_30d",
    }
)

# Merge 7d and 30d summaries into one consolidated view
consolidated = summary_7d_renamed.merge(
    summary_30d_renamed,
    on=["service", "env"],
    how="outer",
    suffixes=("_7d", "_30d")
)

# Optional: compute an overall tier (e.g., higher of the two horizons)
def combined_tier(row):
    tiers = { "Low": 1, "Medium": 2, "High": 3 }
    t7 = row.get("risk_tier_7d", "Low")
    t30 = row.get("risk_tier_30d", "Low")
    # pick the more severe tier
    if tiers.get(t7, 1) >= tiers.get(t30, 1):
        return t7
    else:
        return t30

consolidated["overall_risk_tier"] = consolidated.apply(combined_tier, axis=1)

# Sort for readability: highest risk first
consolidated = consolidated.sort_values(
    ["overall_risk_tier", "avg_risk_7d", "avg_risk_30d"],
    ascending=[False, False, False]
)

consolidated.to_csv("consolidated_risk_summary.csv", index=False)
print("Saved: consolidated_risk_summary.csv")

