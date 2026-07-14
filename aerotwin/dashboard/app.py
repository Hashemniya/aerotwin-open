import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from xgboost import XGBRegressor

st.set_page_config(page_title="AeroTwin Open", layout="wide", page_icon="💧")

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WIDE_PATH = DATA_DIR / "tank1_wide.parquet"

PARAMS_PATH = Path(__file__).parent.parent / "models" / "greybox_params.npy"

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]
DRIVER_COLS = ["airflow", "temperature", "oxygen_setpoint"]
DT_HOURS = 10 / 60


def step_vec(state_arr, driver_arr, params):
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
    return np.stack([nh4_next, no3_next, o2_next], axis=1)


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


@st.cache_resource
def train_hybrid_models(_df):
    """Train one XGBoost residual-correction model per target at a 2h horizon. Cached for the session."""
    params = np.load(PARAMS_PATH)
    horizon_steps = 12  # 2h at 10min resolution
    feat = build_features(_df)

    state = _df[STATE_COLS].values.astype(float)
    drivers = _df[DRIVER_COLS].values.astype(float)
    N = len(_df)
    L = N - horizon_steps
    cur = state[0:L].copy()
    for t in range(horizon_steps):
        cur = step_vec(cur, drivers[t:t + L], params)
    physics_pred_all = cur  # shape (L, 3)

    feat_L = feat.iloc[:L]
    feature_cols = [c for c in feat_L.columns if "_lag" in c or c in ("hour", "dayofweek")]

    models = {}
    for i, target in enumerate(STATE_COLS):
        true_target = _df[target].values[horizon_steps:horizon_steps + L]
        data = feat_L.copy()
        data["physics_pred"] = physics_pred_all[:, i]
        data["true_target"] = true_target
        data["residual"] = data["true_target"] - data["physics_pred"]
        data = data.dropna(subset=feature_cols + ["true_target", "physics_pred"])

        n = len(data)
        train = data.iloc[:int(n * 0.8)]  # use most of it, dashboard doesn't need held-out test here
        model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, n_jobs=2)
        model.fit(train[feature_cols], train["residual"])
        models[target] = model

    return models, params, feature_cols


def forecast_scenario(df, models, params, feature_cols, horizon_steps, o2_setpoint_override=None):
    """Roll the hybrid model forward from the latest available state."""
    feat = build_features(df)
    last_row = feat.iloc[[-1]]
    if last_row[feature_cols].isna().any(axis=None):
        last_row = feat.dropna(subset=feature_cols).iloc[[-1]]

    start_state = df.loc[last_row.index[0], STATE_COLS].values.astype(float).reshape(1, 3)
    recent_airflow = df["airflow"].tail(144).mean()
    recent_temp = df["temperature"].tail(144).mean()
    recent_o2setpoint = df["oxygen_setpoint"].tail(144).mean()
    o2_setpoint = o2_setpoint_override if o2_setpoint_override is not None else recent_o2setpoint

    cur = start_state.copy()
    trajectory = [cur[0].copy()]
    for t in range(horizon_steps):
        drivers = np.array([[recent_airflow, recent_temp, o2_setpoint]])
        cur = step_vec(cur, drivers, params)
        trajectory.append(cur[0].copy())
    trajectory = np.array(trajectory)

    final_physics = trajectory[-1]
    corrections = np.zeros(3)
    for i, target in enumerate(STATE_COLS):
        pred = models[target].predict(last_row[feature_cols])[0]
        corrections[i] = pred
    final_hybrid = final_physics + corrections

    return trajectory, final_hybrid


@st.cache_data
def load_wide_data():
    df = pd.read_parquet(WIDE_PATH)
    return df


def latest_status(df):
    latest = df.iloc[-1]
    latest_time = df.index[-1]
    return latest, latest_time


def main():
    st.title("💧 AeroTwin Open — Aeration Digital Twin")
    st.caption("Decision support for wastewater aeration. Not for automated equipment control.")

    if not WIDE_PATH.exists():
        st.error(f"Data not found at {WIDE_PATH}. Run the data pipeline scripts first (ingest → resample → prepare_wide).")
        return

    df = load_wide_data()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Plant Condition", "📈 Historical Replay", "🔮 Forecast", "🧪 Scenario Lab"])

    with tab1:
        st.subheader("Latest Plant Condition — Tank 1")
        latest, latest_time = latest_status(df)
        st.caption(f"As of {latest_time} (most recent record in dataset)")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Ammonium (NH4)", f"{latest['ammonium']:.2f} mg/L")
        col2.metric("Nitrate (NO3)", f"{latest['nitrate']:.2f} mg/L")
        col3.metric("Dissolved O2", f"{latest['dissolved_oxygen']:.2f} mg/L")
        col4.metric("N2O", f"{latest['nitrous_oxide']:.3f} mg/L")
        col5.metric("Airflow", f"{latest['airflow']:.0f}")

        col6, col7, col8 = st.columns(3)
        col6.metric("Temperature", f"{latest['temperature']:.1f} °C")
        col7.metric("O2 Setpoint", f"{latest['oxygen_setpoint']:.2f} mg/L")
        col8.metric("Valve Position", f"{latest['valve_position']:.0f}%")

        st.divider()
        st.subheader("Data Quality Snapshot")
        recent = df.tail(144)  # last 24h
        missing_pct = recent.isna().mean() * 100
        quality_df = pd.DataFrame({
            "Variable": missing_pct.index,
            "Missing % (last 24h)": missing_pct.values.round(1),
        })
        quality_df["Status"] = quality_df["Missing % (last 24h)"].apply(
            lambda x: "🟢 Good" if x < 5 else ("🟡 Watch" if x < 20 else "🔴 Poor")
        )
        st.dataframe(quality_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Historical Replay")
        n_days = st.slider("Days to show", min_value=1, max_value=90, value=7)
        subset = df.tail(n_days * 144)

        variables = st.multiselect(
            "Select variables to plot",
            options=["ammonium", "nitrate", "dissolved_oxygen", "nitrous_oxide",
                     "airflow", "temperature", "oxygen_setpoint", "valve_position"],
            default=["ammonium", "nitrate", "dissolved_oxygen"],
        )
        if variables:
            st.line_chart(subset[variables])
        else:
            st.info("Select at least one variable above.")


    with tab3:
        st.subheader("Forecast (Hybrid Physics + AI Model)")
        st.caption("2-hour ahead forecast using recent driver averages (airflow, temperature, setpoint).")

        with st.spinner("Training forecast models (first load only, cached after)..."):
            models, gb_params, feature_cols = train_hybrid_models(df)

        trajectory, final_hybrid = forecast_scenario(df, models, gb_params, feature_cols, horizon_steps=12)

        col1, col2, col3 = st.columns(3)
        col1.metric("Ammonium (2h)", f"{final_hybrid[0]:.2f} mg/L", f"{final_hybrid[0]-df['ammonium'].iloc[-1]:.2f}")
        col2.metric("Nitrate (2h)", f"{final_hybrid[1]:.2f} mg/L", f"{final_hybrid[1]-df['nitrate'].iloc[-1]:.2f}")
        col3.metric("Dissolved O2 (2h)", f"{final_hybrid[2]:.2f} mg/L", f"{final_hybrid[2]-df['dissolved_oxygen'].iloc[-1]:.2f}")

        traj_df = pd.DataFrame(trajectory, columns=STATE_COLS)
        traj_df.index = pd.timedelta_range(0, periods=len(traj_df), freq="10min").total_seconds() / 60
        traj_df.index.name = "minutes ahead"
        st.line_chart(traj_df)
        st.caption("Physics-only rollout shown above; the 2h endpoint metrics include the AI correction.")

    with tab4:
        st.subheader("Scenario Laboratory")
        st.caption("Change the oxygen setpoint and see the expected effect on the twin's forecast.")

        with st.spinner("Training scenario models (cached after first load)..."):
            models, gb_params, feature_cols = train_hybrid_models(df)

        current_setpoint = df["oxygen_setpoint"].tail(144).mean()
        new_setpoint = st.slider("Oxygen setpoint (mg/L)", min_value=0.1, max_value=4.0,
                                  value=float(round(current_setpoint, 2)), step=0.1)

        traj, final_hybrid = forecast_scenario(df, models, gb_params, feature_cols, horizon_steps=12,
                                                o2_setpoint_override=new_setpoint)

        col1, col2, col3 = st.columns(3)
        col1.metric("Ammonium (2h)", f"{final_hybrid[0]:.2f} mg/L")
        col2.metric("Nitrate (2h)", f"{final_hybrid[1]:.2f} mg/L")
        col3.metric("Dissolved O2 (2h)", f"{final_hybrid[2]:.2f} mg/L")

        traj_df = pd.DataFrame(traj, columns=STATE_COLS)
        traj_df.index = pd.timedelta_range(0, periods=len(traj_df), freq="10min").total_seconds() / 60
        traj_df.index.name = "minutes ahead"
        st.line_chart(traj_df)

if __name__ == "__main__":
    main()