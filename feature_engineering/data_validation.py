"""Pull compressor_normal_operation from LNN_data, validate, scale, and window
it into fixed-length sequences ready for LNN autoencoder training.

No train/val/test split here — that's a modeling concern, handled later
(e.g. inside the training script). This produces one scaled, windowed
dataset covering the full table.

Usage:
    .venv/Scripts/python.exe feature_engineering/build_features.py
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.errors as pa_errors
from sklearn.preprocessing import StandardScaler

from db_connection import get_sqlalchemy_engine
from schema import CompressorSchema, SENSOR_TAGS

TABLE_NAME = "compressor_normal_operation"
OUT_DIR = Path("data/features")


def load_data(engine) -> pd.DataFrame:
    query = f"SELECT * FROM dbo.{TABLE_NAME} ORDER BY equipment_id, [timestamp]"
    return pd.read_sql(query, engine)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return CompressorSchema.validate(df, lazy=True)
    except pa_errors.SchemaErrors as err:
        print(f"Schema validation failed: {len(err.failure_cases)} failure case(s)")
        print(err.failure_cases.head(20).to_string())
        raise


def check_continuity(df: pd.DataFrame) -> None:
    for equip_id, group in df.groupby("equipment_id"):
        deltas = group["timestamp"].diff().dropna().unique()
        if len(deltas) > 1:
            print(f"WARNING: {equip_id} has non-uniform sample spacing: {deltas[:5]}")


def make_windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(arr)
    if n < window:
        return np.empty((0, window, arr.shape[1]), dtype=np.float32)
    starts = range(0, n - window + 1, stride)
    return np.stack([arr[s : s + window] for s in starts]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=60, help="Window length in minutes")
    parser.add_argument("--stride", type=int, default=15, help="Stride in minutes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = get_sqlalchemy_engine()
    print("Pulling data from LNN_data...")
    df = load_data(engine)
    print(f"Pulled {len(df):,} rows")

    print("Validating schema...")
    df = validate_data(df)
    check_continuity(df)
    print("Schema OK")

    print("Grade mix:", df["process_grade"].value_counts(normalize=True).round(3).to_dict())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[SENSOR_TAGS].to_numpy()).astype(np.float32)
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")

    windows = make_windows(scaled, args.window, args.stride)
    np.savez_compressed(OUT_DIR / "windows.npz", X=windows)
    print(f"windows shape = {windows.shape}")

    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(
            {"sensor_tags": SENSOR_TAGS, "window_minutes": args.window, "stride_minutes": args.stride},
            f,
            indent=2,
        )

    print(f"\nSaved scaler, windows.npz, and feature_columns.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
