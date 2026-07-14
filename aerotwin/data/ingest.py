import os
import pandas as pd

RAW_PATH = "aerotwin/data/raw/aved_raw.csv"
OUT_PATH = "aerotwin/data/processed/canonical.parquet"
PLANT_ID = "avedore"

# (raw_value_col, raw_quality_col, reactor_id, variable, unit, value_type)
MAPPING = [
    ("BIOLOGY.LINE 3 TANK 1.NH4 value", "BIOLOGY.LINE 3 TANK 1.NH4 quality", "tank_1", "ammonium", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.NO3 value", "BIOLOGY.LINE 3 TANK 1.NO3 quality", "tank_1", "nitrate", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.N2O value", "BIOLOGY.LINE 3 TANK 1.N2O quality", "tank_1", "nitrous_oxide", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.O2 value", "BIOLOGY.LINE 3 TANK 1.O2 quality", "tank_1", "dissolved_oxygen", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.O2.SETPOINT value", "BIOLOGY.LINE 3 TANK 1.O2.SETPOINT quality", "tank_1", "oxygen_setpoint", "mg/L", "operator_setpoint"),
    ("BIOLOGY.LINE 3 TANK 1.PROCESSPHASE value", "BIOLOGY.LINE 3 TANK 1.PROCESSPHASE quality", "tank_1", "process_phase", "category", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.Q.AIRFLOW value", "BIOLOGY.LINE 3 TANK 1.Q.AIRFLOW quality", "tank_1", "airflow", "m3/h", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.SS value", "BIOLOGY.LINE 3 TANK 1.SS quality", "tank_1", "suspended_solids", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.TEMPERATURE value", "BIOLOGY.LINE 3 TANK 1.TEMPERATURE quality", "tank_1", "temperature", "degC", "measured"),
    ("BIOLOGY.LINE 3 TANK 1.PO4 value", "BIOLOGY.LINE 3 TANK 1.PO4 quality", "tank_1", "phosphate", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 1 VALVE 1.PCT value", "BIOLOGY.LINE 3 TANK 1 VALVE 1.PCT quality", "tank_1", "valve_position", "pct", "measured"),

    ("BIOLOGY.LINE 3 TANK 2.O2 value", "BIOLOGY.LINE 3 TANK 2.O2 quality", "tank_2", "dissolved_oxygen", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 2.O2.SETPOINT value", "BIOLOGY.LINE 3 TANK 2.O2.SETPOINT quality", "tank_2", "oxygen_setpoint", "mg/L", "operator_setpoint"),
    ("BIOLOGY.LINE 3 TANK 2.PROCESSPHASE value", "BIOLOGY.LINE 3 TANK 2.PROCESSPHASE quality", "tank_2", "process_phase", "category", "measured"),
    ("BIOLOGY.LINE 3 TANK 2.Q.AIRFLOW value", "BIOLOGY.LINE 3 TANK 2.Q.AIRFLOW quality", "tank_2", "airflow", "m3/h", "measured"),
    ("BIOLOGY.LINE 3 TANK 2.SS value", "BIOLOGY.LINE 3 TANK 2.SS quality", "tank_2", "suspended_solids", "mg/L", "measured"),
    ("BIOLOGY.LINE 3 TANK 2.TEMPERATURE value", "BIOLOGY.LINE 3 TANK 2.TEMPERATURE quality", "tank_2", "temperature", "degC", "measured"),
    ("BIOLOGY.LINE 3 TANK 2 VALVE 1.PCT value", "BIOLOGY.LINE 3 TANK 2 VALVE 1.PCT quality", "tank_2", "valve_position", "pct", "measured"),

    ("BIOLOGY.BLOWERSTATION 1.Q.AIRFLOW value", "BIOLOGY.BLOWERSTATION 1.Q.AIRFLOW quality", "plant", "blower_airflow", "m3/h", "measured"),
    ("INLET.Q value", "INLET.Q quality", "plant", "inlet_flow", "m3/h", "measured"),
]


def main():
    print("Loading raw CSV...")
    usecols = ["time"] + [c for pair in MAPPING for c in pair[:2]]
    df = pd.read_csv(RAW_PATH, usecols=usecols)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    frames = []
    for value_col, quality_col, reactor_id, variable, unit, value_type in MAPPING:
        sub = df[["time", value_col, quality_col]].copy()
        sub.columns = ["timestamp", "value", "quality_raw"]
        sub = sub.dropna(subset=["value"])
        sub["plant_id"] = PLANT_ID
        sub["reactor_id"] = reactor_id
        sub["variable"] = variable
        sub["unit"] = unit
        sub["value_type"] = value_type
        sub["source"] = "SCADA"
        sub["quality"] = sub["quality_raw"].apply(lambda q: "good" if q == 1 else "bad")
        sub = sub.drop(columns=["quality_raw"])
        frames.append(sub)
        print(f"  {reactor_id:6s} {variable:20s} {len(sub):>8,} rows")

    long_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")

    os.makedirs("aerotwin/data/processed", exist_ok=True)
    long_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(long_df):,} canonical records to {OUT_PATH}")


if __name__ == "__main__":
    main()