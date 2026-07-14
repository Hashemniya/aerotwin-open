import numpy as np
import pandas as pd
from scipy.optimize import minimize

WIDE_PATH = "aerotwin/data/processed/tank1_wide.parquet"
DT_HOURS = 10 / 60  # 10-minute steps in hours

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]
DRIVER_COLS = ["airflow", "temperature", "oxygen_setpoint"]


def step(state, drivers, params):
    """One Euler integration step of the simplified biology model."""
    nh4, no3, o2 = state
    airflow, temp, o2_setpoint = drivers
    k_nit, K_NH4, K_O2, k_aer, k_resp, theta = params

    temp_factor = theta ** (temp - 15.0)  # Arrhenius-style correction, ref 15C

    nit_rate = k_nit * temp_factor * (nh4 / (nh4 + K_NH4 + 1e-6)) * (o2 / (o2 + K_O2 + 1e-6))

    d_nh4 = -nit_rate
    d_no3 = 0.6 * nit_rate  # not all nitrified N shows as NO3 (some lost to gas/uptake)
    aeration_input = k_aer * max(airflow, 0.0) * max(o2_setpoint - o2, 0.0)
    d_o2 = aeration_input - 4.3 * nit_rate - k_resp

    nh4_next = max(nh4 + d_nh4 * DT_HOURS, 0.0)
    no3_next = max(no3 + d_no3 * DT_HOURS, 0.0)
    o2_next = max(o2 + d_o2 * DT_HOURS, 0.0)
    return np.array([nh4_next, no3_next, o2_next])


def rollout(df, params, start_idx, n_steps):
    """Simulate n_steps forward from start_idx using measured drivers."""
    state = df[STATE_COLS].iloc[start_idx].values.astype(float)
    preds = [state]
    for i in range(n_steps):
        drivers = df[DRIVER_COLS].iloc[start_idx + i].values
        if np.any(pd.isna(drivers)):
            break
        state = step(state, drivers, params)
        preds.append(state)
    return np.array(preds)


def loss_fn(params, state_arr, driver_arr, next_arr, scales):
    """Vectorized one-step-ahead prediction error, normalized per variable."""
    nh4, no3, o2 = state_arr[:, 0], state_arr[:, 1], state_arr[:, 2]
    airflow, temp, o2_setpoint = driver_arr[:, 0], driver_arr[:, 1], driver_arr[:, 2]
    k_nit, K_NH4, K_O2, k_aer, k_resp, theta = params

    temp_factor = theta ** (temp - 15.0)
    nit_rate = k_nit * temp_factor * (nh4 / (nh4 + K_NH4 + 1e-6)) * (o2 / (o2 + K_O2 + 1e-6))

    d_nh4 = -nit_rate
    d_no3 = 0.6 * nit_rate
    aeration_input = k_aer * np.clip(airflow, 0, None) * np.clip(o2_setpoint - o2, 0, None)
    d_o2 = aeration_input - 4.3 * nit_rate - k_resp

    nh4_next = np.clip(nh4 + d_nh4 * DT_HOURS, 0, None)
    no3_next = np.clip(no3 + d_no3 * DT_HOURS, 0, None)
    o2_next = np.clip(o2 + d_o2 * DT_HOURS, 0, None)

    pred = np.stack([nh4_next, no3_next, o2_next], axis=1)
    err = (pred - next_arr) / scales
    return np.mean(err ** 2)


def main():
    df = pd.read_parquet(WIDE_PATH)
    n = len(df)
    train_end = int(n * 0.65)
    tune_end = int(n * 0.80)

    train_slice = df.iloc[0:train_end]
    valid_mask = train_slice[STATE_COLS + DRIVER_COLS].notna().all(axis=1)
    valid_mask &= train_slice[STATE_COLS].shift(-1).notna().all(axis=1)
    valid_idx = np.where(valid_mask.values)[0]
    valid_idx = valid_idx[valid_idx < len(train_slice) - 1]

    sample_idx = np.random.choice(valid_idx, size=min(20000, len(valid_idx)), replace=False)
    state_arr = train_slice[STATE_COLS].values[sample_idx].astype(float)
    driver_arr = train_slice[DRIVER_COLS].values[sample_idx].astype(float)
    next_arr = train_slice[STATE_COLS].values[sample_idx + 1].astype(float)
    scales = train_slice[STATE_COLS].std().values

    x0 = [0.3, 1.0, 0.5, 0.02, 0.05, 1.03]  # k_nit, K_NH4, K_O2, k_aer, k_resp, theta
    bounds = [(0.01, 2.0), (0.1, 5.0), (0.05, 3.0), (0.001, 0.5), (0.0, 0.5), (1.0, 1.1)]

    print("Calibrating grey-box parameters (vectorized, should take a few seconds)...")
    result = minimize(loss_fn, x0, args=(state_arr, driver_arr, next_arr, scales), bounds=bounds,
                       method="L-BFGS-B", options={"maxiter": 100})
    print(f"Calibration finished. Loss: {result.fun:.5f}")
    print(f"Params: k_nit={result.x[0]:.3f}, K_NH4={result.x[1]:.3f}, K_O2={result.x[2]:.3f}, "
          f"k_aer={result.x[3]:.4f}, k_resp={result.x[4]:.4f}, theta={result.x[5]:.4f}")

    np.save("aerotwin/models/greybox_params.npy", result.x)
    print("Saved params to aerotwin/models/greybox_params.npy")

    # Quick multi-step evaluation at a few horizons using test split
    test_start = tune_end + 1000
    horizons = [3, 12, 36, 144]  # steps = 30min,2h,6h,24h at 10min resolution
    for h in horizons:
        preds = rollout(df, result.x, test_start, h)
        if len(preds) <= h:
            continue
        true_end = df[STATE_COLS].iloc[test_start + h].values.astype(float)
        pred_end = preds[-1]
        mae = np.abs(pred_end - true_end)
        print(f"  horizon={h*10}min  MAE nh4={mae[0]:.3f} no3={mae[1]:.3f} o2={mae[2]:.3f}")


if __name__ == "__main__":
    main()