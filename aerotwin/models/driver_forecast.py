import numpy as np
import pandas as pd


def build_seasonal_profile(df, driver_cols=("airflow", "temperature")):
    """Average driver value by (hour, dayofweek), computed from historical data."""
    profile_df = df[list(driver_cols)].copy()
    profile_df["hour"] = profile_df.index.hour
    profile_df["dayofweek"] = profile_df.index.dayofweek
    profile = profile_df.groupby(["dayofweek", "hour"])[list(driver_cols)].mean()
    overall_mean = df[list(driver_cols)].mean()
    return profile, overall_mean


def forecast_drivers_seasonal(df, profile, overall_mean, start_time, horizon_steps, step_minutes=10):
    """Forecast driver values for each future step using hour/day-of-week seasonal averages.
    Falls back to the overall historical mean if a specific (dayofweek, hour) bin has no data."""
    timestamps = pd.date_range(start_time, periods=horizon_steps + 1, freq=f"{step_minutes}min")[1:]
    rows = []
    for ts in timestamps:
        key = (ts.dayofweek, ts.hour)
        if key in profile.index:
            rows.append(profile.loc[key])
        else:
            rows.append(overall_mean)
    return pd.DataFrame(rows, index=timestamps)


# --- Production hook (not used on historical demo data) ---------------------
def fetch_live_temperature_forecast(latitude, longitude, hours_ahead=24):
    """Fetch a real temperature forecast from Open-Meteo (no API key required).
    Only valid when forecasting from the actual current date/time — do not use
    this against historical replay data, since it would return real weather for
    today, not for the historical timestamp being simulated."""
    import requests
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m&forecast_days={max(1, hours_ahead // 24 + 1)}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]
    return pd.Series(temps, index=times).iloc[:hours_ahead]