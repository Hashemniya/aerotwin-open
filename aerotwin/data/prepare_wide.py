import pandas as pd

IN_PATH = "aerotwin/data/processed/canonical_10min.parquet"
OUT_PATH = "aerotwin/data/processed/tank1_wide.parquet"

REACTOR_ID = "tank_1"


def main():
    print("Loading resampled canonical data...")
    df = pd.read_parquet(IN_PATH)
    df = df[df["reactor_id"] == REACTOR_ID]

    wide = df.pivot_table(index="timestamp", columns="variable", values="value", aggfunc="mean")
    wide = wide.sort_index()

    print(f"Wide table shape: {wide.shape}")
    print(f"Date range: {wide.index.min()} to {wide.index.max()}")
    print(f"Columns: {list(wide.columns)}")

    wide.to_parquet(OUT_PATH)
    print(f"\nSaved wide table to {OUT_PATH}")


if __name__ == "__main__":
    main()