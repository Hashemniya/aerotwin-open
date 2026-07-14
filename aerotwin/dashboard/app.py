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

WIDTHS_PATH = Path(__file__).parent.parent / "models" / "uncertainty_widths.npy"

LIMITS = {
    "ammonium_max": 6.0,
    "nitrate_max": 10.0,
    "o2_min": 0.05,
}

SCENARIOS = {
    "conservative": 1.15,
    "balanced": 1.00,
    "green": 0.85,
}


@st.cache_data
def load_uncertainty_widths():
    return np.load(WIDTHS_PATH, allow_pickle=True).item()


def get_margin(widths_dict, target, horizon_min, side):
    key = f"{target}_{horizon_min}"
    entry = widths_dict.get(key, {"upper": 0.5, "lower": 0.5})
    return entry[side]


def run_optimizer_dashboard(df, gb_params, widths_dict, horizon_steps=12, horizon_min=120):
    start_state = df[STATE_COLS].iloc[-1].values.astype(float)
    recent_airflow = df["airflow"].tail(144).mean()
    recent_temp = df["temperature"].tail(144).mean()
    recent_o2setpoint = df["oxygen_setpoint"].tail(144).mean()

    nh4_margin = get_margin(widths_dict, "ammonium", horizon_min, "upper")
    no3_margin = get_margin(widths_dict, "nitrate", horizon_min, "upper")
    o2_margin = get_margin(widths_dict, "dissolved_oxygen", horizon_min, "lower")

    results = []
    for name, multiplier in SCENARIOS.items():
        o2_setpoint = recent_o2setpoint * multiplier
        cur = start_state.reshape(1, 3).copy()
        trajectory = [cur[0].copy()]
        for t in range(horizon_steps):
            drivers = np.array([[recent_airflow, recent_temp, o2_setpoint]])
            cur = step_vec(cur, drivers, gb_params)
            trajectory.append(cur[0].copy())
        trajectory = np.array(trajectory)
        final_nh4, final_no3, final_o2 = trajectory[-1]

        worst_nh4 = final_nh4 + nh4_margin
        worst_no3 = final_no3 + no3_margin
        worst_o2 = final_o2 - o2_margin

        violations = []
        if worst_nh4 > LIMITS["ammonium_max"]:
            violations.append(f"Ammonium worst-case {worst_nh4:.2f} mg/L exceeds limit {LIMITS['ammonium_max']} mg/L")
        if worst_no3 > LIMITS["nitrate_max"]:
            violations.append(f"Nitrate worst-case {worst_no3:.2f} mg/L exceeds limit {LIMITS['nitrate_max']} mg/L")
        if worst_o2 < LIMITS["o2_min"]:
            violations.append(f"Oxygen worst-case {worst_o2:.2f} mg/L below minimum {LIMITS['o2_min']} mg/L")

        aeration_index = round(o2_setpoint * horizon_steps * DT_HOURS, 2)

        results.append({
            "plan": name, "safe": len(violations) == 0, "violations": violations,
            "aeration_index": aeration_index, "setpoint": round(o2_setpoint, 2),
            "final_nh4": round(final_nh4, 2), "final_no3": round(final_no3, 2), "final_o2": round(final_o2, 2),
            "worst_nh4": round(worst_nh4, 2), "worst_no3": round(worst_no3, 2), "worst_o2": round(worst_o2, 2),
        })
    return results

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Plant Condition", "📈 Historical Replay", "🔮 Forecast",
        "🧪 Scenario Lab", "🎯 Optimized Plan", "🛡️ Trust & Safety",
    ])

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

    with tab5:
        st.subheader("Optimized Aeration Plan (2h ahead)")
        st.caption("Three candidate plans, safety-checked against uncertainty-adjusted limits. "
                   "A plan is only recommended if it stays safe even in the worst plausible case.")

        gb_params = np.load(PARAMS_PATH)
        widths_dict = load_uncertainty_widths()
        results = run_optimizer_dashboard(df, gb_params, widths_dict)

        cols = st.columns(3)
        for col, r in zip(cols, results):
            with col:
                if r["safe"]:
                    st.success(f"✅ {r['plan'].upper()}")
                else:
                    st.error(f"❌ {r['plan'].upper()} — REJECTED")
                st.metric("Setpoint", f"{r['setpoint']} mg/L")
                st.metric("Aeration index (energy proxy)", r["aeration_index"])
                st.write(f"NH4: {r['final_nh4']} (worst-case ≤{r['worst_nh4']})")
                st.write(f"NO3: {r['final_no3']} (worst-case ≤{r['worst_no3']})")
                st.write(f"O2: {r['final_o2']} (worst-case ≥{r['worst_o2']})")
                if r["violations"]:
                    for v in r["violations"]:
                        st.caption(f"⚠️ {v}")

        st.divider()
        safe_plans = [r for r in results if r["safe"]]
        if safe_plans:
            best = min(safe_plans, key=lambda r: r["aeration_index"])
            st.success(f"**Recommended: {best['plan'].upper()}** (lowest energy among safe plans)")
        else:
            st.warning("**No recommendation.** All candidate plans exceed safety limits under current "
                       "conditions and forecast uncertainty. This is the system correctly refusing "
                       "rather than guessing.")

    with tab6:
        st.subheader("Trust & Safety")
        st.caption("Model confidence, known limitations, and why recommendations get rejected.")

        st.markdown("**Known limitations of this demo (stated plainly, not hidden):**")
        st.markdown("""
        - Forecasts use recent-average driver values (airflow, temperature) rather than real driver
          forecasts, which widens uncertainty compared to a production deployment.
        - The hybrid model's accuracy is strongest at short horizons (30min–2h) and weaker at 24h,
          especially for ammonium — this is why the optimizer here uses a 2h decision horizon.
        - Safety limits (ammonium/nitrate/O2) are demo defaults based on typical municipal WWTP
          ranges, not this specific plant's actual discharge permit — a real deployment must use the
          plant's real limits.
        - Across a scan of 200 random historical points, only ~14% produced at least one safe plan
          at these settings — meaning this system is deliberately cautious and will often correctly
          decline to recommend rather than force an answer.
        """)

        st.divider()
        st.markdown("**Current data quality (last 24h):**")
        recent = df.tail(144)
        missing_pct = recent.isna().mean() * 100
        for var, pct in missing_pct.items():
            if pct > 5:
                st.warning(f"{var}: {pct:.1f}% missing in last 24h")
        if (missing_pct <= 5).all():
            st.success("All variables within normal data availability (< 5% missing in last 24h).")

if __name__ == "__main__":
    main()