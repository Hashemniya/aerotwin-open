import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "aerotwin/models")

WIDE_PATH = "aerotwin/data/processed/tank1_wide.parquet"
PARAMS_PATH = "aerotwin/models/greybox_params.npy"
WIDTHS_PATH = "aerotwin/models/uncertainty_widths.npy"
DT_HOURS = 10 / 60
STEPS_24H = 12  # 2h at 10-min resolution

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]

# Hard safety limits (typical municipal WWTP effluent targets; adjust per real plant permit)
LIMITS = {
    "ammonium_max": 6.0,      # mg/L (plant's 75th percentile is ~3.1; allow headroom above typical peaks)
    "nitrate_max": 10.0,      # mg/L
    "o2_min": 0.05,            # mg/L (plant cycles into low-O2 anoxic phases by design; this catches true sensor-zero/failure, not normal cycling)
}

SCENARIOS = {
    "conservative": 1.15,  # raise setpoint 15% -> safer, more energy
    "balanced": 1.00,      # keep current setpoint pattern
    "green": 0.85,         # lower setpoint 15% -> less energy, more risk
}


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


def simulate_scenario(start_state, airflow_forecast, temp_forecast, o2_setpoint_forecast, params):
    """Roll the physics model forward one scenario, return full trajectory."""
    state = start_state.reshape(1, 3).astype(float)
    trajectory = [state[0].copy()]
    for t in range(len(airflow_forecast)):
        drivers = np.array([[airflow_forecast[t], temp_forecast[t], o2_setpoint_forecast[t]]])
        state = step_vec(state, drivers, params)
        trajectory.append(state[0].copy())
    return np.array(trajectory)


def get_uncertainty_margin(widths_dict, target, horizon_min, side):
    key = f"{target}_{horizon_min}"
    entry = widths_dict.get(key, {"upper": 0.5, "lower": 0.5})
    return entry[side]


def check_safety(trajectory, widths_dict):
    """Check end-of-horizon prediction +/- uncertainty against hard limits."""
    final_nh4, final_no3, final_o2 = trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2]

    nh4_margin = get_uncertainty_margin(widths_dict, "ammonium", 120, "upper")
    no3_margin = get_uncertainty_margin(widths_dict, "nitrate", 120, "upper")
    o2_margin = get_uncertainty_margin(widths_dict, "dissolved_oxygen", 120, "lower")

    worst_nh4 = final_nh4 + nh4_margin
    worst_no3 = final_no3 + no3_margin
    worst_o2 = final_o2 - o2_margin

    violations = []
    if worst_nh4 > LIMITS["ammonium_max"]:
        violations.append(f"ammonium upper bound {worst_nh4:.2f} exceeds limit {LIMITS['ammonium_max']}")
    if worst_no3 > LIMITS["nitrate_max"]:
        violations.append(f"nitrate upper bound {worst_no3:.2f} exceeds limit {LIMITS['nitrate_max']}")
    if worst_o2 < LIMITS["o2_min"]:
        violations.append(f"oxygen lower bound {worst_o2:.2f} below minimum {LIMITS['o2_min']}")

    return len(violations) == 0, violations, {"worst_nh4": worst_nh4, "worst_no3": worst_no3, "worst_o2": worst_o2}


from driver_forecast import build_seasonal_profile, forecast_drivers_seasonal


def run_optimizer(df, params, widths_dict, start_idx=None, seasonal_profile=None, overall_mean=None):
    if start_idx is None:
        start_idx = len(df) - STEPS_24H - 1  # use last full available 24h window

    start_state = df[STATE_COLS].iloc[start_idx].values.astype(float)

    recent_airflow = df["airflow"].iloc[max(0, start_idx - 144):start_idx].mean()
    recent_o2setpoint = df["oxygen_setpoint"].iloc[max(0, start_idx - 144):start_idx].mean()
    temp_forecast = np.full(STEPS_24H, df["temperature"].iloc[start_idx])
    airflow_forecast_base = np.full(STEPS_24H, recent_airflow)

    results = []
    for name, multiplier in SCENARIOS.items():
        o2_setpoint_forecast = np.full(STEPS_24H, recent_o2setpoint * multiplier)
        airflow_forecast = airflow_forecast_base

        trajectory = simulate_scenario(start_state, airflow_forecast, temp_forecast,
                                        o2_setpoint_forecast, params)
        safe, violations, worst_case = check_safety(trajectory, widths_dict)
        aeration_index = float(np.sum(o2_setpoint_forecast) * DT_HOURS)  # energy proxy

        results.append({
            "plan": name,
            "setpoint_multiplier": multiplier,
            "safe": safe,
            "violations": violations,
            "aeration_index": round(aeration_index, 2),
            "final_ammonium": round(trajectory[-1, 0], 3),
            "final_nitrate": round(trajectory[-1, 1], 3),
            "final_o2": round(trajectory[-1, 2], 3),
            **{k: round(v, 3) for k, v in worst_case.items()},
        })
    return results


def main():
    df = pd.read_parquet(WIDE_PATH)
    df = df.dropna(subset=STATE_COLS + ["airflow", "temperature", "oxygen_setpoint"])
    params = np.load(PARAMS_PATH)
    widths_dict = np.load(WIDTHS_PATH, allow_pickle=True).item()

    results = run_optimizer(df, params, widths_dict)

    print("\n=== AERATION PLAN OPTIONS (24h ahead) ===\n")
    for r in results:
        status = "SAFE" if r["safe"] else "REJECTED"
        print(f"[{status}] {r['plan'].upper():13s} setpoint x{r['setpoint_multiplier']}  "
              f"aeration_index={r['aeration_index']}")
        print(f"    final: NH4={r['final_ammonium']} NO3={r['final_nitrate']} O2={r['final_o2']}")
        print(f"    worst-case (with uncertainty): NH4<={r['worst_nh4']} NO3<={r['worst_no3']} O2>={r['worst_o2']}")
        if r["violations"]:
            for v in r["violations"]:
                print(f"    VIOLATION: {v}")
        print()

    safe_plans = [r for r in results if r["safe"]]
    if safe_plans:
        best = min(safe_plans, key=lambda r: r["aeration_index"])
        print(f"Recommended (lowest energy among safe plans): {best['plan'].upper()}")
    else:
        print("No recommendation: all candidate plans violate safety limits under current conditions.")


def scan_history(df, params, widths_dict, n_samples=200, seed=42):
    """Run the optimizer at many random historical points to see overall behavior."""
    rng = np.random.default_rng(seed)
    valid_start = 200
    valid_end = len(df) - STEPS_24H - 1
    sample_points = rng.choice(range(valid_start, valid_end), size=n_samples, replace=False)

    safe_count = 0
    rejected_count = 0
    example_safe = None
    for idx in sample_points:
        results = run_optimizer(df, params, widths_dict, start_idx=idx)
        safe_plans = [r for r in results if r["safe"]]
        if safe_plans:
            safe_count += 1
            if example_safe is None:
                example_safe = (idx, results)
        else:
            rejected_count += 1

    print(f"\n=== SCAN OF {n_samples} HISTORICAL POINTS ===")
    print(f"At least one safe plan found: {safe_count} ({100*safe_count/n_samples:.1f}%)")
    print(f"All plans rejected: {rejected_count} ({100*rejected_count/n_samples:.1f}%)")

    if example_safe:
        idx, results = example_safe
        print(f"\nExample SAFE case at index {idx} (timestamp {df.index[idx]}):")
        for r in results:
            status = "SAFE" if r["safe"] else "REJECTED"
            print(f"  [{status}] {r['plan'].upper():13s} aeration_index={r['aeration_index']}")


if __name__ == "__main__":
    df = pd.read_parquet(WIDE_PATH)
    df = df.dropna(subset=STATE_COLS + ["airflow", "temperature", "oxygen_setpoint"])
    params = np.load(PARAMS_PATH)
    widths_dict = np.load(WIDTHS_PATH, allow_pickle=True).item()

    main()
    scan_history(df, params, widths_dict, n_samples=200)