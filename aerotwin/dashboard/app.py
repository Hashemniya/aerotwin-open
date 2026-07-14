import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AeroTwin Open", layout="wide", page_icon="💧")

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WIDE_PATH = DATA_DIR / "tank1_wide.parquet"


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

    tab1, tab2 = st.tabs(["📊 Plant Condition", "📈 Historical Replay"])

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


if __name__ == "__main__":
    main()