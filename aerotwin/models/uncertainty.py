import numpy as np
import pandas as pd
from xgboost import XGBRegressor

WIDE_PATH = "aerotwin/data/processed/tank1_wide.parquet"
PARAMS_PATH = "aerotwin/models/greybox_params.npy"
DT_HOURS = 10 / 60

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]
DRIVER_COLS = ["airflow", "temperature", "oxygen_setpoint"]
TARGETS = ["ammonium", "nitrate", "dissolved_oxygen"]
HORIZONS_MIN = [30, 120, 360, 1440]
STEP_MIN = 10
CONFIDENCE = 0.90


INFLUENT_NH4 = 25.0
INFLUENT_NO3 = 0.2


def step_vec(state_arr, driver_arr, params):
    nh4, no3, o2 = state_arr[:, 0], state_arr[:, 1], state_arr[:, 2]
    airflow, temp, o2_setpoint = driver_arr[:, 0], driver_arr[:, 1], driver_arr[:, 2]
    k_nit, K_NH4, K_O2, k_aer, k_resp, theta, k_dil = params

    temp_factor = theta ** (temp - 15.0)
    nit_rate = k_nit * temp_factor * (nh4 / (nh4 + K_NH4 + 1e-6)) * (o2 / (o2 + K_O2 + 1e-6))

    dil_nh4 = k_dil * (INFLUENT_NH4 - nh4)
    dil_no3 = k_dil * (INFLUENT_NO3 - no3)
    dil_o2 = k_dil * (0.0 - o2)

    d_nh4 = -nit_rate + dil_nh4
    d_no3 = 0.6 * nit_rate + dil_no3
    aeration_input = k_aer * np.clip(airflow, 0, None) * np.clip(o2_setpoint - o2, 0, None)
    d_o2 = aeration_input - 4.3 * nit_rate - k_resp + dil_o2

    nh4_next = np.clip(nh4 + d_nh4 * DT_HOURS, 0, None)
    no3_next = np.clip(no3 + d_no3 * DT_HOURS, 0, None)
    o2_next = np.clip(o2 + d_o2 * DT_HOURS, 0, None)
    return np.stack([nh4_next, no3_next, o2_next], axis=1)


def compute_physics_forecasts(df, params, horizon_steps):
    state = df[STATE_COLS].values.astype(float)
    drivers = df[DRIVER_COLS].values.astype(float)
    max_step = max(horizon_steps)
    N = len(df)
    L = N - max_step
    cur = state[0:L].copy()
    forecasts = {}
    for t in range(max_step):
        drv_t = drivers[t:t + L]
        cur = step_vec(cur, drv_t, params)
        step_num = t + 1
        if step_num in horizon_steps:
            forecasts[step_num] = cur.copy()
    return forecasts, L


def build_features(df):
    feat = df.copy()
    feat["hour"] = feat.index.hour
    feat["dayofweek"] = feat.index.dayofweek
    for col in ["ammonium", "nitrate", "dissolved_oxygen", "nitrous_oxide",
                "airflow", "temperature", "oxygen_setpoint", "valve_position"]:
        if col in feat.columns:
            feat[f"{col}_lag1"] = feat[col].shift(1)
            feat[f"{col}_lag6"] = feat[col].shift(6)
            feat[f"{col}_lag144"] = feat[col].shift(144)
    return feat


def time_split_idx(n):
    return int(n * 0.65), int(n * 0.80), int(n * 0.90)


def main():
    df = pd.read_parquet(WIDE_PATH)
    params = np.load(PARAMS_PATH)
    horizon_steps = [h // STEP_MIN for h in HORIZONS_MIN]

    print("Computing physics rollout forecasts...")
    forecasts, L = compute_physics_forecasts(df, params, horizon_steps)
    feat = build_features(df)
    feat_L = feat.iloc[:L]

    results = []
    alpha = 1 - CONFIDENCE

    for target in TARGETS:
        idx = STATE_COLS.index(target)
        for h_min, h_step in zip(HORIZONS_MIN, horizon_steps):
            true_target = df[target].values[h_step:h_step + L]
            physics_pred = forecasts[h_step][:, idx]

            data = feat_L.copy()
            data["true_target"] = true_target
            data["physics_pred"] = physics_pred
            data["residual"] = data["true_target"] - data["physics_pred"]

            feature_cols = [c for c in feat_L.columns if "_lag" in c or c in ("hour", "dayofweek")]
            data = data.dropna(subset=feature_cols + ["true_target", "physics_pred"])

            n = len(data)
            i_train, i_tune, i_calib = time_split_idx(n)
            train = data.iloc[:i_train]
            calib = data.iloc[i_tune:i_calib]
            test = data.iloc[i_calib:]

            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=2)
            model.fit(train[feature_cols], train["residual"])

           # Get calibration-set signed errors: true - pred
            # positive error => model UNDER-predicted (true was higher) => risk for upper-limit checks (NH4, NO3)
            # negative error => model OVER-predicted (true was lower) => risk for lower-limit checks (O2)
            calib_correction = model.predict(calib[feature_cols])
            calib_hybrid_pred = calib["physics_pred"].values + calib_correction
            calib_err = calib["true_target"].values - calib_hybrid_pred  # true - pred

            upper_margin = np.quantile(calib_err[calib_err > 0], CONFIDENCE) if (calib_err > 0).any() else 0.0
            lower_margin = np.quantile(-calib_err[calib_err < 0], CONFIDENCE) if (calib_err < 0).any() else 0.0

            # Apply asymmetric bounds to test set, check coverage
            test_correction = model.predict(test[feature_cols])
            test_hybrid_pred = test["physics_pred"].values + test_correction
            lower = test_hybrid_pred - lower_margin
            upper = test_hybrid_pred + upper_margin
            covered = (test["true_target"].values >= lower) & (test["true_target"].values <= upper)
            coverage = covered.mean()

            results.append({
                "target": target, "horizon_min": h_min,
                "upper_margin": round(upper_margin, 4),
                "lower_margin": round(lower_margin, 4),
                "target_coverage": CONFIDENCE,
                "actual_coverage": round(coverage, 3),
            })
            print(f"{target:20s} {h_min:>5d}min  upper_margin={upper_margin:.3f}  lower_margin={lower_margin:.3f}  "
                  f"target_cov={CONFIDENCE}  actual_cov={coverage:.3f}")

    report = pd.DataFrame(results)
    report.to_csv("aerotwin/data/processed/uncertainty_report.csv", index=False)

    # Save per-target/horizon half-widths for later use by the optimizer/dashboard
    widths_dict = {
        f"{r['target']}_{r['horizon_min']}": {"upper": r["upper_margin"], "lower": r["lower_margin"]}
        for r in results
    }
    np.save("aerotwin/models/uncertainty_widths.npy", widths_dict)
    print("\nSaved uncertainty report and widths.")


if __name__ == "__main__":
    main()