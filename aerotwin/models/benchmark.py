import numpy as np
import pandas as pd
from xgboost import XGBRegressor

WIDE_PATH = "aerotwin/data/processed/tank1_wide.parquet"

TARGETS = ["dissolved_oxygen", "ammonium", "nitrate", "nitrous_oxide"]
HORIZONS_MIN = [30, 120, 360, 1440]  # 30min, 2h, 6h, 24h
STEP_MIN = 10  # resampled frequency


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["hour"] = feat.index.hour
    feat["dayofweek"] = feat.index.dayofweek
    for col in TARGETS + ["airflow", "temperature", "oxygen_setpoint", "valve_position"]:
        if col in feat.columns:
            feat[f"{col}_lag1"] = feat[col].shift(1)
            feat[f"{col}_lag6"] = feat[col].shift(6)   # 1h ago
            feat[f"{col}_lag144"] = feat[col].shift(144)  # 24h ago
    return feat


def time_split(df: pd.DataFrame):
    n = len(df)
    i_train = int(n * 0.65)
    i_tune = int(n * 0.80)
    i_calib = int(n * 0.90)
    return {
        "train": df.iloc[:i_train],
        "tune": df.iloc[i_train:i_tune],
        "calib": df.iloc[i_tune:i_calib],
        "test": df.iloc[i_calib:],
    }


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse


def main():
    df = pd.read_parquet(WIDE_PATH)
    feat = build_features(df)

    results = []

    for target in TARGETS:
        for horizon_min in HORIZONS_MIN:
            steps = horizon_min // STEP_MIN
            data = feat.copy()
            data["target"] = data[target].shift(-steps)
            data = data.dropna(subset=["target", target] + [c for c in data.columns if "_lag" in c])

            splits = time_split(data)
            train, test = splits["train"], splits["test"]

            feature_cols = [c for c in data.columns if c not in TARGETS + ["target", "process_phase"]]

            # Persistence baseline: predict current value stays the same
            persist_pred = test[target].values
            persist_mae, persist_rmse = evaluate(test["target"].values, persist_pred)

            # XGBoost
            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, n_jobs=2)
            model.fit(train[feature_cols], train["target"])
            xgb_pred = model.predict(test[feature_cols])
            xgb_mae, xgb_rmse = evaluate(test["target"].values, xgb_pred)

            results.append({
                "target": target, "horizon_min": horizon_min,
                "persist_mae": round(persist_mae, 4), "persist_rmse": round(persist_rmse, 4),
                "xgb_mae": round(xgb_mae, 4), "xgb_rmse": round(xgb_rmse, 4),
                "xgb_improvement_pct": round(100 * (persist_mae - xgb_mae) / persist_mae, 1),
            })
            print(f"{target:20s} {horizon_min:>5d}min  persist_mae={persist_mae:.3f}  xgb_mae={xgb_mae:.3f}  "
                  f"improvement={results[-1]['xgb_improvement_pct']}%")

    report = pd.DataFrame(results)
    report.to_csv("aerotwin/data/processed/benchmark_report.csv", index=False)
    print("\nSaved benchmark report.")


if __name__ == "__main__":
    main()