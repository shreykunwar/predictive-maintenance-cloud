#Milestone 5

import pandas as pd
import numpy as np


# ----------------------------
# Config (tune if needed)
# ----------------------------

# Dynamic thresholds per (service, env) using quantiles of risk_score
# Example: <= q60 = Low, q60..q85 = Medium, >= q85 = High
Q_LOW = 0.60
Q_HIGH = 0.85

# Severity scoring weights (applied on top of risk_score)
# We assume risk_score already blends latency/cpu/anom, and we add a severity uplift
W_SEV_LAT = 0.30
W_SEV_CPU = 0.20
W_SEV_ANOM = 0.50

# Alert rules
ALERT_TIER = "High"          # High tiers always alert
TOP_N_ALERTS = 25            # keep the alert list manageable
MIN_SEVERITY_FOR_ALERT = 0.70  # optional absolute gate on severity score (0..1-ish)


# ----------------------------
# Helpers
# ----------------------------

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def compute_dynamic_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute q60 and q85 thresholds per (service, env) from df['risk_score'].
    """
    grp = df.groupby(["service", "env"])["risk_score"]
    out = grp.quantile([Q_LOW, Q_HIGH]).unstack(level=-1).reset_index()
    out = out.rename(columns={Q_LOW: "thr_low", Q_HIGH: "thr_high"})
    return out

def assign_tier(risk_score, thr_low, thr_high) -> str:
    if pd.isna(risk_score) or pd.isna(thr_low) or pd.isna(thr_high):
        return "Unknown"
    if risk_score >= thr_high:
        return "High"
    if risk_score >= thr_low:
        return "Medium"
    return "Low"

def compute_severity(row) -> float:
    """
    Severity-weighted score using risk_score and normalized components.
    Assumes latency_norm/cpu_norm/anom_norm are already in 0..1.
    """
    base = float(row["risk_score"])
    lat = float(row.get("latency_norm", 0.0) or 0.0)
    cpu = float(row.get("cpu_norm", 0.0) or 0.0)
    anm = float(row.get("anom_norm", 0.0) or 0.0)

    uplift = (W_SEV_LAT * lat) + (W_SEV_CPU * cpu) + (W_SEV_ANOM * anm)
    sev = base * (1.0 + uplift)

    # clamp to keep readable
    return float(np.clip(sev, 0.0, 2.0))

def build_outputs(daily_path: str, horizon_label: str):
    """
    Loads risk_outlook_daily_*.csv, assigns tiers, computes severity, returns dataframe.
    """
    df = pd.read_csv(daily_path, parse_dates=["ds"])

    required = [
        "ds", "service", "env",
        "risk_score",
        "latency_norm", "cpu_norm", "anom_norm",
        "latency_hat", "cpu_hat", "anom_hat"
    ]
    require_cols(df, required)

    # Compute per-group thresholds based on the same horizon dataset
    thr = compute_dynamic_thresholds(df)
    df = df.merge(thr, on=["service", "env"], how="left")

    # Assign tiers
    df["risk_tier"] = df.apply(lambda r: assign_tier(r["risk_score"], r["thr_low"], r["thr_high"]), axis=1)

    # Compute severity score
    df["severity_score"] = df.apply(compute_severity, axis=1).round(3)

    # Add horizon label for consolidation later
    df["horizon"] = horizon_label

    return df

def make_alert_list(df7: pd.DataFrame, df30: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prioritized alert list using:
      - risk_tier == High OR severity_score >= MIN_SEVERITY_FOR_ALERT
    Prioritize 7d first, then 30d, by severity_score desc and earliest date.
    """
    df = pd.concat([df7, df30], ignore_index=True)

    # Alert conditions
    alert_mask = (df["risk_tier"] == ALERT_TIER) | (df["severity_score"] >= MIN_SEVERITY_FOR_ALERT)
    alerts = df[alert_mask].copy()

    # Rank within each service/env by highest severity
    alerts["alert_rank_in_group"] = alerts.groupby(["service", "env", "horizon"])["severity_score"] \
                                          .rank(ascending=False, method="first")

    # Keep top few per group to avoid flooding
    alerts = alerts[alerts["alert_rank_in_group"] <= 3].copy()

    # Sort overall priority
    alerts = alerts.sort_values(
        ["horizon", "severity_score", "ds"],
        ascending=[True, False, True]
    )

    # Keep only essential columns for supervisor-friendly list
    out_cols = [
        "horizon", "ds", "service", "env",
        "risk_tier", "risk_score", "severity_score",
        "latency_hat", "cpu_hat", "anom_hat"
    ]
    alerts = alerts[out_cols].copy()

    # Enforce global cap
    alerts = alerts.head(TOP_N_ALERTS).reset_index(drop=True)

    return alerts

def write_summary_txt(df7: pd.DataFrame, df30: pd.DataFrame, alerts: pd.DataFrame, path: str):
    """
    Create a short documentation-style summary.
    """
    def tier_counts(df):
        return df["risk_tier"].value_counts().to_dict()

    counts_7 = tier_counts(df7)
    counts_30 = tier_counts(df30)

    # Top risky service/env by avg severity in 7d
    top7 = (
        df7.groupby(["service", "env"])["severity_score"].mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    lines = []
    lines.append("MILESTONE 5 SUMMARY")
    lines.append("")
    lines.append("What was done:")
    lines.append("1) Applied service/env-specific dynamic thresholds to classify risk tiers (Low/Medium/High).")
    lines.append("2) Computed severity-weighted scores using risk_score with additional uplift from normalized latency/CPU/anomaly components.")
    lines.append("3) Generated an initial alert list prioritizing High-tier and high-severity windows for review.")
    lines.append("")
    lines.append("Threshold method:")
    lines.append(f"- Per (service, env), thresholds are computed from risk_score quantiles: Low<=q{int(Q_LOW*100)}, High>=q{int(Q_HIGH*100)}.")
    lines.append("")
    lines.append("Risk-tier distribution:")
    lines.append(f"- 7-day horizon counts: {counts_7}")
    lines.append(f"- 30-day horizon counts: {counts_30}")
    lines.append("")
    lines.append("Top service/environment by average 7-day severity:")
    for _, r in top7.iterrows():
        lines.append(f"- {r['service']} / {r['env']}: avg severity {r['severity_score']:.3f}")
    lines.append("")
    lines.append("Alerting rule:")
    lines.append(f"- Alert if risk_tier == '{ALERT_TIER}' OR severity_score >= {MIN_SEVERITY_FOR_ALERT}")
    lines.append(f"- Output capped to top {TOP_N_ALERTS} alerts overall, max 3 per service/env per horizon.")
    lines.append("")
    lines.append("Files produced:")
    lines.append("- m5_risk_classification_7d.csv")
    lines.append("- m5_risk_classification_30d.csv")
    lines.append("- m5_alert_list_initial.csv")
    lines.append("- Milestone5_Summary.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# Main
# ----------------------------

def main():
    df7 = build_outputs("risk_outlook_daily_7d.csv", "7d")
    df30 = build_outputs("risk_outlook_daily_30d.csv", "30d")

    # Save classification outputs
    df7.to_csv("m5_risk_classification_7d.csv", index=False)
    df30.to_csv("m5_risk_classification_30d.csv", index=False)

    # Create alert list
    alerts = make_alert_list(df7, df30)
    alerts.to_csv("m5_alert_list_initial.csv", index=False)

    # Write documentation summary
    write_summary_txt(df7, df30, alerts, "Milestone5_Summary.txt")

    print("Saved: m5_risk_classification_7d.csv")
    print("Saved: m5_risk_classification_30d.csv")
    print("Saved: m5_alert_list_initial.csv")
    print("Saved: Milestone5_Summary.txt")
    print("Milestone 5 remaining steps completed.")

if __name__ == "__main__":
    main()
