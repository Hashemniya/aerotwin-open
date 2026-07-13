import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AeroTwin Open", layout="wide")
st.title("AeroTwin Open — Aeration Digital Twin")
st.write("Skeleton is running. Next: data ingestion.")

# placeholder chart to confirm plotting works
df = pd.DataFrame({
    "time": pd.date_range("2026-01-01", periods=50, freq="10min"),
    "dissolved_oxygen": np.random.normal(2, 0.3, 50)
})
st.line_chart(df.set_index("time"))