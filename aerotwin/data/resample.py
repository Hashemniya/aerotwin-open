import pandas as pd

CANONICAL_PATH = "aerotwin/data/processed/canonical.parquet"
OUT_PATH = "aerotwin/data/processed/canonical_10min.parquet"

RESAMPLE_FREQ = "10min"
MAX_GAP_TO_FILL = 3  # interpolate gaps up to 3 x 10min = 30min; longer gaps stay NaN


def resample_series(df_var: pd.DataFrame) -> pd.DataFrame:
    good = df_var[df_var["quality"] == "good"].copy()
    good = good.dropna(subset=["value"])
    good = good.set_index("timestamp").sort_index()

    if len(good) < 2:
        return pd.DataFrame()

    resampled = good["value"].resample(RESAMPLE_FREQ).mean()
    resampled = resampled.interpolate(method="linear", limit=MAX_GAP_TO_FILL)

    out = resampled.reset_index()
    out.columns = ["timestamp", "value"]
    out["reactor_id"] = df_var["reactor_id"].iloc[0]
    out["variable"] = df_var["variable"].iloc[0]
    out["unit"] = df_var["unit"].iloc[0]
    out["plant_id"] = df_var["plant_id"].iloc[0]
    out["value_type"] = "measured_resampled"
    return out


def main():
    print("Loading canonical data...")
    df = pd.read_parquet(CANONICAL_PATH)

    frames = []
    for (reactor_id, variable), group in df.groupby(["reactor_id", "variable"]):
        result = resample_series(group)
        if not result.empty:
            frames.append(result)
            n_before = len(group)
            n_after = result["value"].notna().sum()
            print(f"  {reactor_id:6s} {variable:20s} {n_before:>8,} -> {n_after:>8,} rows (10min)")

    out_df = pd.concat(frames, ignore_index=True).sort_values(["reactor_id", "variable", "timestamp"])
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(out_df):,} resampled records to {OUT_PATH}")


if __name__ == "__main__":
    main()