import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WIDE_PATH = DATA_DIR / "tank1_wide.parquet"
PARAMS_PATH = Path(__file__).parent.parent / "models" / "greybox_params.npy"
WIDTHS_PATH = Path(__file__).parent.parent / "models" / "uncertainty_widths.npy"

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]
DRIVER_COLS = ["airflow", "temperature", "oxygen_setpoint"]
DT_HOURS = 10 / 60
LIMITS = {"ammonium_max": 6.0, "nitrate_max": 10.0, "o2_min": 0.05}
SCENARIOS = {"conservative": 1.15, "balanced": 1.00, "green": 0.85}
INFLUENT_NH4 = 25.0
INFLUENT_NO3 = 0.2

# --- Stub auth / tier store. Replace with real billing/auth in production. ---
API_KEYS = {
    "demo-key-monitoring": "monitoring",
    "demo-key-advisory": "advisory",
}
TIER_ENDPOINTS = {
    "monitoring": {"forecast", "health", "latest_state"},
    "advisory": {"forecast", "health", "latest_state", "optimize"},
}


def check_access(api_key: Optional[str], endpoint: str):
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    tier = API_KEYS[api_key]
    if endpoint not in TIER_ENDPOINTS[tier]:
        raise HTTPException(
            status_code=403,
            detail=f"Your plan ('{tier}') does not include access to '{endpoint}'. Upgrade to 'advisory' for optimizer recommendations.",
        )
    return tier


# --- Physics + hybrid model (same logic validated throughout this project) ---

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
    return np.stack([
        np.clip(nh4 + d_nh4 * DT_HOURS, 0, None),
        np.clip(no3 + d_no3 * DT_HOURS, 0, None),
        np.clip(o2 + d_o2 * DT_HOURS, 0, None),
    ], axis=1)


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


def get_margin(widths_dict, target, horizon_min, side):
    key = f"{target}_{horizon_min}"
    return widths_dict.get(key, {"upper": 0.5, "lower": 0.5})[side]


# --- App state, loaded once at startup ---

class ModelStore:
    df: pd.DataFrame = None
    params: np.ndarray = None
    widths: dict = None
    models: dict = None
    feature_cols: list = None


store = ModelStore()

app = FastAPI(
    title="BioTwin API",
    description="Physics-informed hybrid digital twin for wastewater aeration. "
                 "Currently calibrated only on the Avedøre WWTP reference dataset.",
    version="0.1.0",
)


@app.on_event("startup")
def load_models():
    store.df = pd.read_parquet(WIDE_PATH)
    store.params = np.load(PARAMS_PATH)
    store.widths = np.load(WIDTHS_PATH, allow_pickle=True).item()

    horizon_steps = 12
    feat = build_features(store.df)
    state = store.df[STATE_COLS].values.astype(float)
    drivers = store.df[DRIVER_COLS].values.astype(float)
    N = len(store.df)
    L = N - horizon_steps
    cur = state[0:L].copy()
    for t in range(horizon_steps):
        cur = step_vec(cur, drivers[t:t + L], store.params)
    physics_pred_all = cur

    feat_L = feat.iloc[:L]
    feature_cols = [c for c in feat_L.columns if "_lag" in c or c in ("hour", "dayofweek")]
    store.feature_cols = feature_cols

    models = {}
    for i, target in enumerate(STATE_COLS):
        true_target = store.df[target].values[horizon_steps:horizon_steps + L]
        data = feat_L.copy()
        data["physics_pred"] = physics_pred_all[:, i]
        data["true_target"] = true_target
        data["residual"] = data["true_target"] - data["physics_pred"]
        data = data.dropna(subset=feature_cols + ["true_target", "physics_pred"])
        n = len(data)
        train = data.iloc[:int(n * 0.8)]
        model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, n_jobs=2)
        model.fit(train[feature_cols], train["residual"])
        models[target] = model
    store.models = models


# --- Request/response schemas ---

class ForecastRequest(BaseModel):
    horizon_minutes: int = 120  # only 120 (2h) is validated; other values run but are unvalidated


class ForecastResponse(BaseModel):
    timestamp_used: str
    horizon_minutes: int
    forecast: dict
    uncertainty_90pct: dict
    disclaimer: str


class OptimizeResponse(BaseModel):
    timestamp_used: str
    plans: list
    recommended_plan: Optional[str]
    disclaimer: str


# --- Endpoints ---

@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None)):
    check_access(x_api_key, "health")
    return {"status": "ok", "data_loaded": store.df is not None}


@app.get("/state/latest")
def latest_state(x_api_key: Optional[str] = Header(default=None)):
    check_access(x_api_key, "latest_state")
    latest = store.df.iloc[-1]
    return {
        "timestamp": str(store.df.index[-1]),
        "ammonium": float(latest["ammonium"]),
        "nitrate": float(latest["nitrate"]),
        "dissolved_oxygen": float(latest["dissolved_oxygen"]),
        "temperature": float(latest["temperature"]),
        "airflow": float(latest["airflow"]),
        "oxygen_setpoint": float(latest["oxygen_setpoint"]),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, x_api_key: Optional[str] = Header(default=None)):
    check_access(x_api_key, "forecast")

    horizon_steps = req.horizon_minutes // 10
    feat = build_features(store.df)
    last_row = feat.iloc[[-1]]
    if last_row[store.feature_cols].isna().any(axis=None):
        last_row = feat.dropna(subset=store.feature_cols).iloc[[-1]]

    start_state = store.df.loc[last_row.index[0], STATE_COLS].values.astype(float).reshape(1, 3)
    recent_airflow = store.df["airflow"].tail(144).mean()
    recent_temp = store.df["temperature"].tail(144).mean()
    recent_o2setpoint = store.df["oxygen_setpoint"].tail(144).mean()

    cur = start_state.copy()
    for _ in range(horizon_steps):
        drivers = np.array([[recent_airflow, recent_temp, recent_o2setpoint]])
        cur = step_vec(cur, drivers, store.params)
    final_physics = cur[0]

    corrections = np.array([store.models[t].predict(last_row[store.feature_cols])[0] for t in STATE_COLS])
    final_hybrid = final_physics + corrections

    margins = {
        t: get_margin(store.widths, t, min(req.horizon_minutes, 1440), "upper" if t != "dissolved_oxygen" else "lower")
        for t in STATE_COLS
    }

    return ForecastResponse(
        timestamp_used=str(store.df.index[-1]),
        horizon_minutes=req.horizon_minutes,
        forecast={t: round(float(v), 3) for t, v in zip(STATE_COLS, final_hybrid)},
        uncertainty_90pct={t: round(float(margins[t]), 3) for t in STATE_COLS},
        disclaimer="Model calibrated on Avedøre WWTP historical data only. Forecast accuracy is "
                   "strongest at 30min-2h horizons; longer horizons are less validated.",
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(x_api_key: Optional[str] = Header(default=None)):
    check_access(x_api_key, "optimize")

    start_state = store.df[STATE_COLS].iloc[-1].values.astype(float)
    recent_airflow = store.df["airflow"].tail(144).mean()
    recent_temp = store.df["temperature"].tail(144).mean()
    recent_o2setpoint = store.df["oxygen_setpoint"].tail(144).mean()
    horizon_steps = 12

    nh4_margin = get_margin(store.widths, "ammonium", 120, "upper")
    no3_margin = get_margin(store.widths, "nitrate", 120, "upper")
    o2_margin = get_margin(store.widths, "dissolved_oxygen", 120, "lower")

    plans = []
    for name, mult in SCENARIOS.items():
        o2_setpoint = recent_o2setpoint * mult
        cur = start_state.reshape(1, 3).copy()
        for _ in range(horizon_steps):
            drivers = np.array([[recent_airflow, recent_temp, o2_setpoint]])
            cur = step_vec(cur, drivers, store.params)
        final_nh4, final_no3, final_o2 = cur[0]

        worst_nh4 = final_nh4 + nh4_margin
        worst_no3 = final_no3 + no3_margin
        worst_o2 = final_o2 - o2_margin
        violations = []
        if worst_nh4 > LIMITS["ammonium_max"]:
            violations.append(f"Ammonium worst-case {worst_nh4:.2f} exceeds limit {LIMITS['ammonium_max']}")
        if worst_no3 > LIMITS["nitrate_max"]:
            violations.append(f"Nitrate worst-case {worst_no3:.2f} exceeds limit {LIMITS['nitrate_max']}")
        if worst_o2 < LIMITS["o2_min"]:
            violations.append(f"Oxygen worst-case {worst_o2:.2f} below minimum {LIMITS['o2_min']}")

        safe = len(violations) == 0
        aeration_index = round(o2_setpoint * horizon_steps * DT_HOURS, 3)
        plans.append({
            "plan": name, "safe": safe, "violations": violations,
            "setpoint": round(float(o2_setpoint), 3), "aeration_index": aeration_index,
        })

    safe_plans = [p for p in plans if p["safe"]]
    recommended = min(safe_plans, key=lambda p: p["aeration_index"])["plan"] if safe_plans else None

    return OptimizeResponse(
        timestamp_used=str(store.df.index[-1]),
        plans=plans,
        recommended_plan=recommended,
        disclaimer="Safety limits are demo defaults, not a validated discharge permit. "
                   "Decision support only -- not for automated equipment control. "
                   "See BioTwin's Trust & Safety documentation for validated safe-recommendation rates.",
    )