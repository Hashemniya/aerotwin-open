import numpy as np
import pandas as pd
from driver_forecast import build_seasonal_profile, forecast_drivers_seasonal

WIDE_PATH = "aerotwin/data/processed/tank1_wide.parquet"
HORIZON_STEPS = 144  # 24h


def main():
    df = pd.read_parquet(WIDE_PATH)
    n = len(df)
    train_end = int(n * 0.65)
    train = df.iloc[:train_end]
    profile, overall_mean = build_seasonal_profile(train)

    test_start = int(n * 0.90)
    sample_idx = np.random.choice(range(test_start, n - HORIZON_STEPS), size=200, replace=False)

    seasonal_errs, persistence_errs = [], []
    for idx in sample_idx:
        start_time = df.index[idx]
        true_future = df[["airflow", "temperature"]].iloc[idx + 1: idx + 1 + HORIZON_STEPS]

        seasonal_fc = forecast_drivers_seasonal(df, profile, overall_mean, start_time, HORIZON_STEPS)
        recent_avg = df[["airflow", "temperature"]].iloc[max(0, idx - 144):idx].mean()
        persistence_fc = pd.DataFrame([recent_avg] * HORIZON_STEPS, index=true_future.index)

        seasonal_errs.append((seasonal_fc.values - true_future.values))
        persistence_errs.append((persistence_fc.values - true_future.values))

    seasonal_mae = np.nanmean(np.abs(np.concatenate(seasonal_errs)), axis=0)
    persistence_mae = np.nanmean(np.abs(np.concatenate(persistence_errs)), axis=0)

    for i, col in enumerate(["airflow", "temperature"]):
        print(f"{col:12s} seasonal_mae={seasonal_mae[i]:.3f}  persistence_mae={persistence_mae[i]:.3f}  "
              f"seasonal_better={seasonal_mae[i] < persistence_mae[i]}")


if __name__ == "__main__":
    main()