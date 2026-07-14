import pandas as pd
import numpy as np

CANONICAL_PATH = "aerotwin/data/processed/canonical.parquet"


def check_variable(df_var: pd.DataFrame, variable: str, reactor_id: str) -> dict:
    """Run quality checks on a single variable's time series."""
    issues = {}

    total = len(df_var)
    bad_quality = (df_var["quality"] == "bad").sum()
    issues["bad_quality_flag_count"] = int(bad_quality)
    issues["bad_quality_pct"] = round(100 * bad_quality / total, 2) if total else 0

    dupes = df_var["timestamp"].duplicated().sum()
    issues["duplicate_timestamps"] = int(dupes)

    ts_sorted = df_var["timestamp"].sort_values()
    gaps = ts_sorted.diff().dropna()
    expected = gaps.median()
    long_gaps = gaps[gaps > expected * 5]
    issues["long_outage_count"] = int(len(long_gaps))
    issues["longest_outage_minutes"] = round(gaps.max().total_seconds() / 60, 1) if len(gaps) else 0

    good = df_var[df_var["quality"] == "good"].sort_values("timestamp")
    frozen_streak = (good["value"].diff() == 0).astype(int)
    frozen_run = frozen_streak.groupby((frozen_streak != frozen_streak.shift()).cumsum()).cumsum()
    issues["longest_frozen_run"] = int(frozen_run.max()) if len(frozen_run) else 0

    if variable in ("airflow", "blower_airflow", "suspended_solids", "dissolved_oxygen",
                    "ammonium", "nitrate", "nitrous_oxide", "phosphate"):
        negative = (good["value"] < 0).sum()
        issues["negative_value_count"] = int(negative)

    if variable == "valve_position":
        out_of_range = ((good["value"] < 0) | (good["value"] > 100)).sum()
        issues["out_of_range_count"] = int(out_of_range)

    vals = good["value"].dropna()
    if len(vals) > 100:
        mean, std = vals.mean(), vals.std()
        jumps = good["value"].diff().abs()
        sudden_jumps = (jumps > 6 * std).sum() if std > 0 else 0
        issues["sudden_jump_count"] = int(sudden_jumps)

    return issues


def main():
    print("Loading canonical data...")
    df = pd.read_parquet(CANONICAL_PATH)

    report_rows = []
    for (reactor_id, variable), group in df.groupby(["reactor_id", "variable"]):
        issues = check_variable(group, variable, reactor_id)
        row = {"reactor_id": reactor_id, "variable": variable, "row_count": len(group)}
        row.update(issues)
        report_rows.append(row)

    report = pd.DataFrame(report_rows).sort_values(["reactor_id", "variable"])
    out_path = "aerotwin/data/processed/quality_report.csv"
    report.to_csv(out_path, index=False)

    print("\n=== DATA QUALITY REPORT ===\n")
    print(report.to_string(index=False))
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()