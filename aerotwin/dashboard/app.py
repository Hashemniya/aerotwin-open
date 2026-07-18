import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))


AVEDORE_LAT, AVEDORE_LON = 55.615, 12.457  # Avedøre WWTP, Copenhagen, Denmark


def fetch_live_temperature_forecast(latitude, longitude, hours_ahead=24):
    """Fetch a real temperature forecast from Open-Meteo (no API key required).
    Only meaningful when forecasting from the actual current date/time."""
    import requests
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m&forecast_days={max(1, hours_ahead // 24 + 1)}"
    )
    resp = requests.get(url, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]
    return pd.Series(temps, index=times).iloc[:hours_ahead]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BioTwin", layout="wide", page_icon="💧",
                    initial_sidebar_state="expanded")

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WIDE_PATH = DATA_DIR / "tank1_wide.parquet"
PARAMS_PATH = Path(__file__).parent.parent / "models" / "greybox_params.npy"
WIDTHS_PATH = Path(__file__).parent.parent / "models" / "uncertainty_widths.npy"

STATE_COLS = ["ammonium", "nitrate", "dissolved_oxygen"]
DRIVER_COLS = ["airflow", "temperature", "oxygen_setpoint"]
DT_HOURS = 10 / 60

LIMITS = {"ammonium_max": 6.0, "nitrate_max": 10.0, "o2_min": 0.05}
SCENARIOS = {"conservative": 1.15, "balanced": 1.00, "green": 0.85}

REQUIRED_UPLOAD_COLS = ["timestamp", "ammonium", "nitrate", "dissolved_oxygen",
                         "airflow", "temperature", "oxygen_setpoint"]
OPTIONAL_UPLOAD_COLS = ["nitrous_oxide", "phosphate", "process_phase",
                         "suspended_solids", "valve_position"]

# Vibrant per-metric colors (icon badges, charts)
METRIC_COLORS = {
    "ammonium": "#f97316",         # orange
    "nitrate": "#3b82f6",          # blue
    "dissolved_oxygen": "#10b981", # emerald
    "nitrous_oxide": "#eab308",    # yellow
    "airflow": "#06b6d4",          # cyan
    "temperature": "#ef4444",      # red
    "oxygen_setpoint": "#8b5cf6",  # violet
    "valve_position": "#f59e0b",   # amber
}
ICONS = {
    "ammonium": "🟠", "nitrate": "🔵", "dissolved_oxygen": "🟢",
    "nitrous_oxide": "🟡", "airflow": "💨", "temperature": "🌡️",
    "oxygen_setpoint": "🎯", "valve_position": "🔧",
}

PRIMARY = "#1e3a5f"
PRIMARY_DARK = "#0f2540"
ACCENT = "#10b981"
DANGER = "#ef4444"
WARN = "#f59e0b"
BG_PAGE = "#f1f5f9"
BG_CARD = "#ffffff"
BORDER = "#e2e8f0"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
    .stApp {{ background: {BG_PAGE}; }}
    header[data-testid="stHeader"] {{ height: 0; min-height: 0; visibility: hidden; }}
    div[data-testid="stAppViewContainer"] > .main {{ padding-top: 0 !important; }}
    .main .block-container {{ padding-top: 1rem !important; padding-bottom: 2rem; max-width: 1400px; }}

    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        border-radius: 16px; padding: 20px 28px; margin-bottom: 14px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 6px 20px rgba(15, 37, 64, 0.2);
    }}
    .hero-left {{ display: flex; align-items: center; gap: 14px; }}
    .hero-icon {{
        font-size: 2rem; background: rgba(255,255,255,0.14); border-radius: 12px;
        width: 52px; height: 52px; display: flex; align-items: center; justify-content: center;
    }}
    .hero h1 {{ color: white; font-size: 1.4rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }}
    .hero p {{ color: #bcd3e8; margin: 2px 0 0 0; font-size: 0.86rem; }}
    .hero-badge {{
        background: rgba(255,255,255,0.14); color: #dceaf5; padding: 6px 14px;
        border-radius: 24px; font-size: 0.76rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.16); white-space: nowrap;
    }}

    .section-label {{ display: flex; align-items: center; gap: 8px; margin: 4px 0 10px 0; }}
    .section-label .bar {{ width: 4px; height: 18px; background: {ACCENT}; border-radius: 3px; }}
    .section-label h3 {{ margin: 0; font-size: 1.02rem; font-weight: 700; color: {PRIMARY_DARK}; }}
    .section-sub {{ color: #7d8ea1; font-size: 0.8rem; margin: -8px 0 12px 12px; }}

    /* KPI cards with colored icon badge + sparkline slot */
    .kpi-card {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 14px 16px 10px 16px; box-shadow: 0 1px 4px rgba(15,37,64,0.05);
        height: 100%; min-height: 88px;
    }}
    .kpi-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .kpi-badge {{
        width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center;
        justify-content: center; font-size: 0.95rem; flex-shrink: 0;
    }}
    .kpi-label {{ color: #64748b; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
    .kpi-value {{ color: {PRIMARY_DARK}; font-size: 1.2rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.25; white-space: nowrap; }}
    .kpi-unit {{ font-size: 0.72rem; font-weight: 500; color: #94a3b8; }}
    .kpi-delta-up {{ color: {DANGER}; font-size: 0.72rem; font-weight: 700; }}
    .kpi-delta-down {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700; }}

    .chart-card {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 14px 16px 4px 16px; box-shadow: 0 1px 6px rgba(15,37,64,0.05); margin-bottom: 12px;
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; padding: 9px 16px; font-weight: 600; font-size: 0.86rem; color: #64748b;
    }}
    .stTabs [aria-selected="true"] {{
        background: {BG_CARD} !important; color: {PRIMARY_DARK} !important; box-shadow: 0 -2px 0 {ACCENT} inset;
    }}

    /* Plan cards */
    .plan-card {{
        border-radius: 14px; padding: 16px; background: {BG_CARD}; border: 1.5px solid {BORDER};
        box-shadow: 0 1px 6px rgba(15,37,64,0.05); height: 100%; position: relative; overflow: hidden;
    }}
    .plan-card.safe {{ border-color: #bbf7d0; }}
    .plan-card.rejected {{ border-color: #fecaca; opacity: 0.94; }}
    .plan-card::before {{ content: ""; position: absolute; top:0; left:0; right:0; height:4px; }}
    .plan-card.safe::before {{ background: {ACCENT}; }}
    .plan-card.rejected::before {{ background: {DANGER}; }}
    .plan-title {{ font-size: 1rem; font-weight: 700; color: {PRIMARY_DARK}; margin: 4px 0 6px 0; text-transform: capitalize; }}
    .badge-safe {{ background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; display: inline-block; }}
    .badge-rejected {{ background: #fee2e2; color: {DANGER}; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; display: inline-block; }}
    .plan-metric-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dashed {BORDER}; font-size: 0.8rem; color: #475569; }}
    .plan-metric-row:last-child {{ border-bottom: none; }}
    .plan-metric-val {{ font-weight: 700; color: {PRIMARY_DARK}; }}
    .violation-line {{ color: {DANGER}; font-size: 0.72rem; margin-top: 6px; background: #fef2f2; padding: 5px 8px; border-radius: 6px; }}
    .energy-bar-track {{ background: #f1f5f9; border-radius: 6px; height: 8px; margin-top: 4px; overflow: hidden; }}
    .energy-bar-fill {{ height: 100%; border-radius: 6px; }}

    .rec-banner-safe {{ background: linear-gradient(90deg, #dcfce7, #f0fdf4); border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 18px; color: #15803d; font-weight: 600; font-size: 0.92rem; }}
    .rec-banner-warn {{ background: linear-gradient(90deg, #fef3c7, #fffbeb); border: 1px solid #fde68a; border-radius: 12px; padding: 14px 18px; color: #92400e; font-weight: 600; font-size: 0.9rem; }}

    section[data-testid="stSidebar"] {{ background: {PRIMARY_DARK}; }}
    section[data-testid="stSidebar"] * {{ color: #cbd8e6 !important; }}
    .sb-logo {{ display: flex; align-items: center; gap: 10px; padding: 4px 0 12px 0; }}
    .sb-logo-icon {{ font-size: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; }}
    .sb-logo-text {{ font-weight: 800; font-size: 1.05rem; color: white !important; }}
    .sb-step {{ display: flex; align-items: center; gap: 8px; padding: 5px 0; font-size: 0.74rem; color: #94a8bd !important; }}
    .sb-step .dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
    .sb-card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }}
    .sb-card-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; color: #7590a8 !important; font-weight: 700; margin-bottom: 3px; }}
    .sb-card-value {{ font-size: 0.85rem; font-weight: 600; color: white !important; }}

    .dq-pill-good {{ background:#dcfce7; color:#15803d; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }}
    .dq-pill-watch {{ background:#fef3c7; color:#92400e; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }}
    .dq-pill-poor {{ background:#fee2e2; color:{DANGER}; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }}

    .health-score-wrap {{ display: flex; align-items: center; gap: 20px; }}
    .health-score-num {{ font-size: 2.2rem; font-weight: 900; color: {PRIMARY_DARK}; line-height: 1; }}
    .health-score-label {{ font-size: 0.78rem; color: #64748b; font-weight: 600; }}
    .health-bar-track {{ flex: 1; background: #f1f5f9; border-radius: 8px; height: 12px; overflow: hidden; }}
    .health-bar-fill {{ height: 100%; border-radius: 8px; background: linear-gradient(90deg, {ACCENT}, #34d399); }}

    .source-card {{
        background: {BG_CARD}; border: 2px solid {BORDER}; border-radius: 16px;
        padding: 28px 24px; text-align: center; box-shadow: 0 1px 8px rgba(15,37,64,0.05); height: 100%;
    }}
    .source-card h4 {{ color: {PRIMARY_DARK}; font-size: 1.1rem; margin: 8px 0 4px 0; }}
    .source-card p {{ color: #64748b; font-size: 0.84rem; }}
    .source-icon {{ font-size: 2.4rem; }}
    [data-testid="stMetric"] {{ display: none; }}

    .price-card {{
        background: {BG_CARD}; border: 2px solid {BORDER}; border-radius: 16px;
        padding: 26px 22px; text-align: center; height: 100%;
        box-shadow: 0 1px 8px rgba(15,37,64,0.05);
    }}
    .price-card.featured {{ border-color: {ACCENT}; box-shadow: 0 4px 16px rgba(16,185,129,0.15); }}
    .price-tier-name {{ font-size: 1.1rem; font-weight: 800; color: {PRIMARY_DARK}; margin-bottom: 4px; }}
    .price-amount {{ font-size: 2rem; font-weight: 900; color: {PRIMARY_DARK}; margin: 10px 0 2px 0; }}
    .price-period {{ font-size: 0.8rem; color: #94a3b8; margin-bottom: 16px; }}
    .price-feature {{ font-size: 0.85rem; color: #475569; padding: 6px 0; border-bottom: 1px dashed {BORDER}; text-align: left; }}
    .price-feature:last-child {{ border-bottom: none; }}
    .price-badge-featured {{
        background: {ACCENT}; color: white; padding: 3px 12px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 700; display: inline-block; margin-bottom: 10px;
    }}
</style>
"""

def section_label(title, subtitle=None):
    st.markdown(f"""<div class="section-label"><div class="bar"></div><h3>{title}</h3></div>""",
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"""<div class="section-sub">{subtitle}</div>""", unsafe_allow_html=True)


def kpi_card(metric_key, label, value, unit="", delta=None, delta_good_when="down"):
    color = METRIC_COLORS.get(metric_key, "#64748b")
    icon = ICONS.get(metric_key, "•")
    delta_html = ""
    if delta is not None and not pd.isna(delta):
        is_up = delta > 0
        arrow = "▲" if is_up else "▼"
        good = (is_up and delta_good_when == "up") or (not is_up and delta_good_when == "down")
        cls = "kpi-delta-down" if good else "kpi-delta-up"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.2f}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <div class="kpi-badge" style="background:{color}22; color:{color};">{icon}</div>
            <span class="kpi-label">{label}</span>
        </div>
        <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def html_legend(y_cols):
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:0.82rem;color:#475569;font-weight:600;">'
        f'<span style="width:10px;height:10px;border-radius:3px;background:{METRIC_COLORS.get(c, ACCENT)};display:inline-block;"></span>'
        f'{c.replace("_", " ").title()}</span>'
        for c in y_cols
    )
    st.markdown(f'<div style="margin-bottom:8px;">{chips}</div>', unsafe_allow_html=True)


def area_line_chart(df, y_cols, height=320, y_title=None):
    x_col = df.index.name or "index"
    plot_df = df[y_cols].reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: x_col})
    plot_df = plot_df.melt(id_vars=x_col, var_name="variable", value_name="value")

    is_time = pd.api.types.is_datetime64_any_dtype(df.index)
    x_enc = alt.X(f"{x_col}:T", title=None) if is_time else alt.X(f"{x_col}:Q", title=None)

    color_scale = alt.Scale(domain=y_cols, range=[METRIC_COLORS.get(c, ACCENT) for c in y_cols])

    area = alt.Chart(plot_df).mark_area(
        opacity=0.12, interpolate="monotone", line=False
    ).encode(x=x_enc, y=alt.Y("value:Q", title=y_title), color=alt.Color("variable:N", scale=color_scale, legend=None))

    line = alt.Chart(plot_df).mark_line(strokeWidth=3, interpolate="monotone").encode(
        x=x_enc, y=alt.Y("value:Q", title=y_title),
        color=alt.Color("variable:N", scale=color_scale, legend=None),
        tooltip=["variable", "value"],
    )

    chart = (area + line).properties(height=height).configure_axis(
        gridColor="#f1f5f9", domainColor=BORDER, labelColor="#94a3b8", labelFontSize=11, titleColor="#64748b",
    ).configure_view(strokeWidth=0)

    return chart


def mini_sparkline(series, color, height=36):
    sdf = series.reset_index()
    sdf.columns = ["x", "y"]
    chart = alt.Chart(sdf).mark_area(
        opacity=0.18, line={"color": color, "strokeWidth": 1.8}, color=color, interpolate="monotone"
    ).encode(
        x=alt.X("x:T", axis=None), y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
    ).properties(height=height).configure_view(strokeWidth=0)
    return chart


# ---------------------------------------------------------------------------
# Data + model functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_avedore_data():
    return pd.read_parquet(WIDE_PATH)


def validate_and_load_upload(uploaded_file):
    try:
        raw = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read file as CSV: {e}"

    missing = [c for c in REQUIRED_UPLOAD_COLS if c not in raw.columns]
    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}"

    try:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    except Exception as e:
        return None, f"Could not parse 'timestamp' column as datetime: {e}"

    raw = raw.set_index("timestamp").sort_index()
    for col in OPTIONAL_UPLOAD_COLS:
        if col not in raw.columns:
            raw[col] = np.nan

    numeric_cols = REQUIRED_UPLOAD_COLS[1:] + [c for c in OPTIONAL_UPLOAD_COLS if c != "process_phase"]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if len(raw) < 20:
        return None, "File has too few rows (need at least 20) to display meaningfully."

    return raw, None


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
    params = np.load(PARAMS_PATH)
    horizon_steps = 12
    feat = build_features(_df)

    state = _df[STATE_COLS].values.astype(float)
    drivers = _df[DRIVER_COLS].values.astype(float)
    N = len(_df)
    L = N - horizon_steps
    cur = state[0:L].copy()
    for t in range(horizon_steps):
        cur = step_vec(cur, drivers[t:t + L], params)
    physics_pred_all = cur

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
        train = data.iloc[:int(n * 0.8)]
        model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, n_jobs=2)
        model.fit(train[feature_cols], train["residual"])
        models[target] = model

    return models, params, feature_cols


def forecast_scenario(df, models, params, feature_cols, horizon_steps, o2_setpoint_override=None,
                       temp_forecast_override=None):
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
        step_temp = temp_forecast_override[t] if temp_forecast_override is not None else recent_temp
        drivers = np.array([[recent_airflow, step_temp, o2_setpoint]])
        cur = step_vec(cur, drivers, params)
        trajectory.append(cur[0].copy())
    trajectory = np.array(trajectory)

    final_physics = trajectory[-1]
    corrections = np.array([models[t].predict(last_row[feature_cols])[0] for t in STATE_COLS])
    final_hybrid = final_physics + corrections
    return trajectory, final_hybrid


@st.cache_data
def load_uncertainty_widths():
    return np.load(WIDTHS_PATH, allow_pickle=True).item()


def get_margin(widths_dict, target, horizon_min, side):
    key = f"{target}_{horizon_min}"
    entry = widths_dict.get(key, {"upper": 0.5, "lower": 0.5})
    return entry[side]


def run_optimizer_dashboard(df, gb_params, widths_dict, horizon_steps=12, horizon_min=120,
                             temp_forecast_override=None):
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
            step_temp = temp_forecast_override[t] if temp_forecast_override is not None else recent_temp
            drivers = np.array([[recent_airflow, step_temp, o2_setpoint]])
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
            "trajectory": trajectory,
        })
    return results


# ---------------------------------------------------------------------------
# Landing screen
# ---------------------------------------------------------------------------

def render_landing():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-left">
            <div class="hero-icon">💧</div>
            <div>
                <h1>BioTwin</h1>
                <p>A digital twin for wastewater aeration. Choose a data source to begin.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="source-card">
            <div class="source-icon">📤</div>
            <h4>Upload Your Data</h4>
            <p>Test BioTwin on your own plant's SCADA export. Requires a CSV with a timestamp
            column and core variables (ammonium, nitrate, dissolved oxygen, airflow, temperature,
            oxygen setpoint).</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:
            df, error = validate_and_load_upload(uploaded_file)
            if error:
                st.error(error)
            else:
                st.session_state.data_source = "uploaded"
                st.session_state.uploaded_df = df
                st.rerun()
        with st.expander("Required CSV format"):
            st.code(
                "timestamp,ammonium,nitrate,dissolved_oxygen,airflow,temperature,oxygen_setpoint,"
                "nitrous_oxide,phosphate,suspended_solids,valve_position\n"
                "2024-01-01 00:00:00,2.1,5.3,0.8,120,14.2,1.5,0.02,1.1,180,45",
                language="csv",
            )
            st.caption("Only the first 7 columns are required; the rest are optional and will show as "
                      "unavailable if omitted.")

    with col2:
        st.markdown("""
        <div class="source-card">
            <div class="source-icon">🌊</div>
            <h4>Load Real Avedøre Data</h4>
            <p>Explore BioTwin using two years of real SCADA data from Avedøre WWTP, Copenhagen.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Load Avedøre Data", use_container_width=True, type="primary"):
            if not WIDE_PATH.exists():
                st.error(f"Avedøre data not found at {WIDE_PATH}. Run the data pipeline scripts first.")
            else:
                st.session_state.data_source = "avedore"
                st.rerun()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.session_state.setdefault("data_source", None)

    if st.session_state.data_source is None:
        render_landing()
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if st.session_state.data_source == "avedore":
        df = load_avedore_data()
        source_label = "Avedøre WWTP (real SCADA data)"
        models_available = True
    else:
        df = st.session_state.uploaded_df
        source_label = "Your uploaded data"
        models_available = False

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    latest_time = df.index[-1]
    spark_window = df.tail(min(144, len(df)))

    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-icon">💧</div>
            <div class="sb-logo-text">BioTwin</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-card">
            <div class="sb-card-label">Data Source</div>
            <div class="sb-card-value">{source_label}</div>
        </div>
        <div class="sb-card">
            <div class="sb-card-label">Mode</div>
            <div class="sb-card-value">Decision support only</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Change data source", use_container_width=True):
            st.session_state.data_source = None
            st.session_state.pop("uploaded_df", None)
            st.rerun()

        st.markdown('<div class="sb-card-label" style="margin:12px 0 6px 4px;">Pipeline</div>',
                    unsafe_allow_html=True)
        steps = [
            ("Raw SCADA ingestion", "#06b6d4"), ("Canonical Data Model", "#3b82f6"),
            ("Quality checks", "#8b5cf6"), ("10-min resample", "#f59e0b"),
            ("Grey-box physics", "#10b981"), ("Hybrid AI correction", "#f97316"),
            ("Uncertainty calibration", "#eab308"), ("Safe optimizer", "#ef4444"),
        ]
        steps_html = "".join(
            f'<div class="sb-step"><div class="dot" style="background:{c};"></div><div>{s}</div></div>'
            for s, c in steps
        )
        st.markdown(f'<div class="sb-card">{steps_html}</div>', unsafe_allow_html=True)
        st.caption("Built with Streamlit, XGBoost, and a physics-informed grey-box model.")

    st.markdown(f"""
    <div class="hero">
        <div class="hero-left">
            <div class="hero-icon">💧</div>
            <div>
                <h1>BioTwin</h1>
                <p>What aeration plan should the plant use next — safely, and with energy in mind?</p>
            </div>
        </div>
        <div class="hero-badge">🕒 Data as of {latest_time.strftime('%Y-%m-%d %H:%M UTC')}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Plant Condition", "📈 Historical Replay", "🔮 Forecast",
        "🧪 Scenario Lab", "🎯 Optimized Plan", "🛡️ Trust & Safety", "💰 Pricing",
    ])

    # --- Tab 1: Plant Condition -------------------------------------------------
    with tab1:
        section_label("Latest Plant Condition", f"Most recent record — {source_label}")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            kpi_card("ammonium", "Ammonium", f"{latest['ammonium']:.2f}", "mg/L",
                     delta=latest['ammonium'] - prev['ammonium'], delta_good_when="down")
        with r1c2:
            kpi_card("nitrate", "Nitrate", f"{latest['nitrate']:.2f}", "mg/L",
                     delta=latest['nitrate'] - prev['nitrate'], delta_good_when="down")
        with r1c3:
            kpi_card("dissolved_oxygen", "Dissolved O2", f"{latest['dissolved_oxygen']:.2f}", "mg/L",
                     delta=latest['dissolved_oxygen'] - prev['dissolved_oxygen'], delta_good_when="up")
        with r1c4:
            n2o_val = latest['nitrous_oxide']
            kpi_card("nitrous_oxide", "N2O", f"{n2o_val:.3f}" if pd.notna(n2o_val) else "n/a", "mg/L")

        st.write("")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            kpi_card("airflow", "Airflow", f"{latest['airflow']:.0f}", "m³/h")
        with r2c2:
            kpi_card("temperature", "Temperature", f"{latest['temperature']:.1f}", "°C")
        with r2c3:
            kpi_card("oxygen_setpoint", "O2 Setpoint", f"{latest['oxygen_setpoint']:.2f}", "mg/L")
        with r2c4:
            vp = latest['valve_position']
            kpi_card("valve_position", "Valve Position", f"{vp:.0f}" if pd.notna(vp) else "n/a", "%")

        st.write("")
        col_left, col_right = st.columns([3, 2])

        with col_left:
            section_label("Recent Trend (24h)", None)
            trend_vars = [v for v in ["ammonium", "nitrate", "dissolved_oxygen"] if spark_window[v].notna().any()]
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            if trend_vars:
                html_legend(trend_vars)
                st.altair_chart(area_line_chart(spark_window, trend_vars, height=260, y_title="mg/L"),
                            use_container_width=True)
            else:
                st.info("Not enough recent data to plot.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            section_label("Data Health", "Aggregate quality over the last 24h")
            recent = df.tail(min(144, len(df)))
            missing_pct = (recent.isna().mean() * 100).round(1)
            health_score = round(100 - missing_pct.mean(), 1)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            bar_color = ACCENT if health_score >= 90 else (WARN if health_score >= 70 else DANGER)
            st.markdown(f"""
            <div class="health-score-wrap">
                <div>
                    <div class="health-score-num" style="color:{bar_color};">{health_score}%</div>
                    <div class="health-score-label">Healthy</div>
                </div>
                <div class="health-bar-track"><div class="health-bar-fill" style="width:{health_score}%; background:{bar_color};"></div></div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            rows_html = ""
            for var, pct in missing_pct.items():
                if pct < 5:
                    pill = f'<span class="dq-pill-good">🟢 {pct}%</span>'
                elif pct < 20:
                    pill = f'<span class="dq-pill-watch">🟡 {pct}%</span>'
                else:
                    pill = f'<span class="dq-pill-poor">🔴 {pct}%</span>'
                icon = ICONS.get(var, "•")
                rows_html += (f'<div class="plan-metric-row"><span>{icon} '
                              f'{var.replace("_", " ").title()}</span><span>{pill}</span></div>')
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: Historical Replay ------------------------------------------------
    with tab2:
        section_label("Historical Replay", "Explore how plant variables have moved over time")
        max_days = max(1, len(df) // 144)
        col_a, col_b = st.columns([1, 2])
        with col_a:
            n_days = st.slider("Days to show", min_value=1, max_value=min(90, max_days),
                                value=min(7, max_days))
        with col_b:
            available_vars = [c for c in ["ammonium", "nitrate", "dissolved_oxygen", "nitrous_oxide",
                                           "airflow", "temperature", "oxygen_setpoint", "valve_position"]
                              if c in df.columns and df[c].notna().any()]
            variables = st.multiselect("Variables", options=available_vars,
                                        default=available_vars[:3])
        subset = df.tail(n_days * 144)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if variables:
            html_legend(variables)
            st.altair_chart(area_line_chart(subset, variables, height=400), use_container_width=True)
        else:
            st.info("Select at least one variable above.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tabs 3-6: require calibrated models (Avedøre only, for now) ------------
    if not models_available:
        for tab, name in [(tab3, "Forecast"), (tab4, "Scenario Lab"),
                           (tab5, "Optimized Plan"), (tab6, "Trust & Safety")]:
            with tab:
                st.info(
                    f"**{name} is not yet available for uploaded data.** BioTwin's physics and AI "
                    "models are currently calibrated on Avedøre WWTP only. Applying them to a new "
                    "plant's biology without recalibration would be misleading. A plant-onboarding "
                    "and recalibration workflow is planned — for now, switch to 'Load Real Avedøre "
                    "Data' to see these features in action."
                )
        return

    # --- Tab 3: Forecast -----------------------------------------------------
    with tab3:
        section_label("2-Hour Forecast", "Hybrid physics + AI model")
        with st.spinner("Training forecast models (first load only, cached after)..."):
            models, gb_params, feature_cols = train_hybrid_models(df)

        temp_override = None
        is_live = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() < 48 * 3600
        if is_live:
            try:
                live_temp = fetch_live_temperature_forecast(AVEDORE_LAT, AVEDORE_LON, hours_ahead=2)
                temp_override = live_temp.values
                st.caption("🌦️ Using a live Open-Meteo temperature forecast (data is current).")
            except Exception:
                st.caption("🌦️ Live weather forecast unavailable right now — using recent-average temperature instead.")
        else:
            st.caption("🌦️ Using recent-average temperature — live forecasts only apply to current/live data, "
                      "not this historical replay.")

        trajectory, final_hybrid = forecast_scenario(df, models, gb_params, feature_cols, horizon_steps=12,
                                                      temp_forecast_override=temp_override)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("ammonium", "Ammonium (2h)", f"{final_hybrid[0]:.2f}", "mg/L",
                     delta=final_hybrid[0]-latest['ammonium'], delta_good_when="down")
        with c2:
            kpi_card("nitrate", "Nitrate (2h)", f"{final_hybrid[1]:.2f}", "mg/L",
                     delta=final_hybrid[1]-latest['nitrate'], delta_good_when="down")
        with c3:
            kpi_card("dissolved_oxygen", "Diss. O2 (2h)", f"{final_hybrid[2]:.2f}", "mg/L",
                     delta=final_hybrid[2]-latest['dissolved_oxygen'], delta_good_when="up")

        st.write("")
        traj_df = pd.DataFrame(trajectory, columns=STATE_COLS)
        traj_df.index = pd.timedelta_range(0, periods=len(traj_df), freq="10min").total_seconds() / 60
        traj_df.index.name = "minutes_ahead"
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        html_legend(STATE_COLS)
        st.altair_chart(area_line_chart(traj_df, STATE_COLS, height=340, y_title="mg/L"),
                        use_container_width=True)
        st.caption("Physics-only rollout shown above; the 2h endpoint metrics include the AI correction.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 4: Scenario Lab --------------------------------------------------
    with tab4:
        section_label("Scenario Laboratory", "Change the oxygen setpoint and see the expected effect")
        with st.spinner("Training scenario models (cached after first load)..."):
            models, gb_params, feature_cols = train_hybrid_models(df)

        current_setpoint = df["oxygen_setpoint"].tail(144).mean()
        new_setpoint = st.slider("Oxygen setpoint (mg/L)", min_value=0.1, max_value=4.0,
                                  value=float(round(current_setpoint, 2)), step=0.1)

        traj, final_hybrid = forecast_scenario(df, models, gb_params, feature_cols, horizon_steps=12,
                                                o2_setpoint_override=new_setpoint)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("ammonium", "Ammonium (2h)", f"{final_hybrid[0]:.2f}", "mg/L")
        with c2:
            kpi_card("nitrate", "Nitrate (2h)", f"{final_hybrid[1]:.2f}", "mg/L")
        with c3:
            kpi_card("dissolved_oxygen", "Diss. O2 (2h)", f"{final_hybrid[2]:.2f}", "mg/L")

        st.write("")
        traj_df = pd.DataFrame(traj, columns=STATE_COLS)
        traj_df.index = pd.timedelta_range(0, periods=len(traj_df), freq="10min").total_seconds() / 60
        traj_df.index.name = "minutes_ahead"
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        html_legend(STATE_COLS)
        st.altair_chart(area_line_chart(traj_df, STATE_COLS, height=340, y_title="mg/L"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # --- Tab 5: Optimized Plan ------------------------------------------------
    with tab5:
        section_label("Optimized Aeration Plan", "Three candidate plans, safety-checked against uncertainty-adjusted limits")

        gb_params = np.load(PARAMS_PATH)
        widths_dict = load_uncertainty_widths()

        temp_override = None
        is_live = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() < 48 * 3600
        if is_live:
            try:
                live_temp = fetch_live_temperature_forecast(AVEDORE_LAT, AVEDORE_LON, hours_ahead=2)
                temp_override = live_temp.values
            except Exception:
                temp_override = None

        results = run_optimizer_dashboard(df, gb_params, widths_dict, temp_forecast_override=temp_override)
        max_energy = max(r["aeration_index"] for r in results) or 1

        plan_colors = {"conservative": "#10b981", "balanced": "#3b82f6", "green": "#f59e0b"}

        cols = st.columns(3)
        for col, r in zip(cols, results):
            with col:
                status_class = "safe" if r["safe"] else "rejected"
                badge = '<span class="badge-safe">✅ SAFE</span>' if r["safe"] else '<span class="badge-rejected">❌ REJECTED</span>'
                violations_html = "".join(f'<div class="violation-line">⚠️ {v}</div>' for v in r["violations"])
                bar_pct = round(100 * r["aeration_index"] / max_energy)
                bar_color = plan_colors.get(r["plan"], ACCENT)
                st.markdown(f"""
                <div class="plan-card {status_class}">
                    <div class="plan-title">{r['plan']}</div>
                    {badge}
                    <div style="margin-top:12px;">
                        <div class="plan-metric-row"><span>Setpoint</span><span class="plan-metric-val">{r['setpoint']} mg/L</span></div>
                        <div class="plan-metric-row"><span>Aeration index</span><span class="plan-metric-val">{r['aeration_index']}</span></div>
                    </div>
                    <div class="energy-bar-track"><div class="energy-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div></div>
                    <div style="margin-top:10px;">
                        <div class="plan-metric-row"><span>NH4 (worst-case)</span><span class="plan-metric-val">{r['final_nh4']} (≤{r['worst_nh4']})</span></div>
                        <div class="plan-metric-row"><span>NO3 (worst-case)</span><span class="plan-metric-val">{r['final_no3']} (≤{r['worst_no3']})</span></div>
                        <div class="plan-metric-row"><span>O2 (worst-case)</span><span class="plan-metric-val">{r['final_o2']} (≥{r['worst_o2']})</span></div>
                    </div>
                    {violations_html}
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        traj_rows = []
        for r in results:
            t = np.arange(len(r["trajectory"])) * 10
            for minute, val in zip(t, r["trajectory"][:, 0]):
                traj_rows.append({"minutes_ahead": minute, "ammonium": val,
                                  "plan": f"{r['plan']} ({'safe' if r['safe'] else 'rejected'})"})
        traj_df2 = pd.DataFrame(traj_rows)
        traj_chart = alt.Chart(traj_df2).mark_line(strokeWidth=2.6, interpolate="monotone").encode(
            x=alt.X("minutes_ahead:Q", title="minutes ahead"),
            y=alt.Y("ammonium:Q", title="Ammonium (mg/L)"),
            color=alt.Color("plan:N", legend=alt.Legend(title=None, orient="top")),
            strokeDash=alt.condition("indexof(datum.plan, 'rejected') >= 0", alt.value([4, 4]), alt.value([1, 0])),
        ).properties(height=280).configure_axis(
            gridColor="#f1f5f9", domainColor=BORDER, labelColor="#94a3b8"
        ).configure_view(strokeWidth=0)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.altair_chart(traj_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        safe_plans = [r for r in results if r["safe"]]
        if safe_plans:
            best = min(safe_plans, key=lambda r: r["aeration_index"])
            st.markdown(f'<div class="rec-banner-safe">✅ Recommended: <b>{best["plan"].upper()}</b> '
                       f'— lowest energy among safe plans.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rec-banner-warn">⚠️ No recommendation. All candidate plans exceed '
                       'safety limits under current conditions and forecast uncertainty — the system is '
                       'correctly refusing rather than guessing.</div>', unsafe_allow_html=True)

    # --- Tab 6: Trust & Safety --------------------------------------------------
    with tab6:
        section_label("Trust & Safety", "Model confidence, known limitations, and why recommendations get rejected")

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Known limitations of this demo (stated plainly, not hidden):**")
        st.markdown("""
        - Forecasts use recent-average driver values (airflow, temperature) rather than real driver
          forecasts. A seasonal-climatology alternative was tested and found worse than persistence
          (see the validation script in the repo) — this remains an open item for future work.
        - The hybrid model's accuracy is strongest at short horizons (30min–2h) and weaker at 24h,
          especially for ammonium — this is why the optimizer uses a 2h decision horizon.
        - Safety limits are demo defaults based on typical municipal WWTP ranges, not this plant's
          actual discharge permit — a real deployment must use the plant's real limits.
        - Across a reproducible 200-point historical scan (seed=42), only 10.5% of moments produced
          at least one safe plan — this system is deliberately cautious.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Current data quality (last 24h):**")
        recent = df.tail(min(144, len(df)))
        missing_pct = recent.isna().mean() * 100
        any_warn = False
        for var, pct in missing_pct.items():
            if pct > 5:
                st.warning(f"{var}: {pct:.1f}% missing in last 24h")
                any_warn = True
        if not any_warn:
            st.success("All variables within normal data availability (< 5% missing in last 24h).")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        section_label("Pricing", "Choose the plan that fits your plant's needs")

        pricing = [
            {
                "tier": "Monitoring", "price": "€490", "period": "per plant / month",
                "featured": False,
                "features": [
                    "Live SCADA dashboard (Plant Condition, Historical Replay)",
                    "2-hour hybrid physics + AI forecasts",
                    "Data quality monitoring & alerts",
                    "Canonical Data Model integration",
                    "Email support",
                ],
            },
            {
                "tier": "Advisory", "price": "€1,490", "period": "per plant / month",
                "featured": True,
                "features": [
                    "Everything in Monitoring, plus:",
                    "Safety-gated aeration plan recommendations",
                    "Scenario Lab (what-if simulation)",
                    "Uncertainty-calibrated safety checks",
                    "One-time plant calibration/onboarding included",
                    "Priority support",
                ],
            },
            {
                "tier": "Enterprise / API", "price": "Custom", "period": "annual license",
                "featured": False,
                "features": [
                    "Everything in Advisory, plus:",
                    "Direct API access for your own systems",
                    "Multi-plant / multi-reactor deployment",
                    "Custom safety limits per discharge permit",
                    "Dedicated onboarding & SLA",
                    "White-label / on-prem deployment options",
                ],
            },
        ]

        cols = st.columns(3)
        for col, tier in zip(cols, pricing):
            with col:
                cls = "price-card featured" if tier["featured"] else "price-card"
                badge = '<div class="price-badge-featured">MOST POPULAR</div>' if tier["featured"] else ""
                features_html = "".join(f'<div class="price-feature">✓ {f}</div>' for f in tier["features"])
                card_html = (
                    f'<div class="{cls}">'
                    f'{badge}'
                    f'<div class="price-tier-name">{tier["tier"]}</div>'
                    f'<div class="price-amount">{tier["price"]}</div>'
                    f'<div class="price-period">{tier["period"]}</div>'
                    f'<div style="text-align:left;">{features_html}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.write("")
        


if __name__ == "__main__":
    main()