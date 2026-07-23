"""Pull compressor_normal_operation from the configured database (see .env),
validate, scale, and window it into fixed-length sequences ready for LNN
autoencoder training.

process_grade runs in multi-week campaigns (see
data_generation/generate_compressor_normal_data.py) rather than being uniform
over time, so a single global chronological cutoff can starve val/test of
whole grades. Splitting is therefore campaign-stratified: each grade's own
rows are cut chronologically at train_frac/val_frac/test_frac independently,
so every split gets a representative share of every grade. Windows are still
built per contiguous (equipment_id, grade-campaign) run, so a window never
splices together two different campaigns, machines, or splits - even when a
split boundary falls in the middle of one campaign. The scaler is fit on the
train split only and reused (unfit) on val/test.

Usage:
    .venv/Scripts/python.exe feature_engineering/data_validation.py
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pandera.errors as pa_errors
from sklearn.preprocessing import StandardScaler
from db_connection import get_sqlalchemy_engine
from schema_validate import CompressorSchema, SENSOR_TAGS

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


def label_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a campaign_id to each row: a new id starts whenever process_grade
    changes within an equipment_id's chronological sequence. Campaigns are the
    atomic unit for splitting, so a split boundary never cuts through a single
    grade's continuous run (which would leak that run's slow autocorrelated
    drift across splits)."""
    df = df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
    new_campaign = df["process_grade"] != df.groupby("equipment_id")["process_grade"].shift()
    campaign_num = new_campaign.groupby(df["equipment_id"]).cumsum()
    df["campaign_id"] = df["equipment_id"] + "_" + campaign_num.astype(str)
    return df


def campaign_stratified_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split so every grade appears in train/val/test at roughly train_frac/val_frac/test_frac.

    Within each grade, rows are cut chronologically at those fractions (a
    grade with only 1-2 campaigns can't otherwise guarantee coverage of all
    three splits from whole campaigns alone - see label_campaigns/
    make_windows_per_campaign for how leakage across the cut is still
    avoided: campaign_id groups rows into contiguous same-equipment,
    same-grade runs, so a cut that lands mid-campaign still only ever
    produces windows within one split's contiguous chunk of that campaign,
    never a window straddling the cut itself.
    """
    df = label_campaigns(df)
    parts = {"train": [], "val": [], "test": []}
    for _, grade_df in df.groupby("process_grade"):
        grade_df = grade_df.sort_values("timestamp")
        n = len(grade_df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        parts["train"].append(grade_df.iloc[:train_end])
        parts["val"].append(grade_df.iloc[train_end:val_end])
        parts["test"].append(grade_df.iloc[val_end:])

    return tuple(
        pd.concat(parts[name]).sort_values(["equipment_id", "timestamp"])
        for name in ("train", "val", "test")
    )


def make_windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = len(arr)
    if n < window:
        # array smaller than one window: return empty (0, window, n_features) so callers can concatenate safely
        return np.empty((0, window, arr.shape[1]), dtype=np.float32)
    starts = range(0, n - window + 1, stride)
    return np.stack([arr[s : s + window] for s in starts]).astype(np.float32)


def make_windows_per_campaign(
    df: pd.DataFrame, scaler: StandardScaler, window: int, stride: int
) -> np.ndarray:
    # windowed per campaign_id: a campaign is by construction one equipment_id's
    # uninterrupted run of one grade, so a sequence never splices together rows
    # from a different machine, a different grade campaign, or across a split boundary
    all_windows = []
    for _, group in df.groupby("campaign_id"):
        group = group.sort_values("timestamp")
        scaled = scaler.transform(group[SENSOR_TAGS].to_numpy()).astype(np.float32)
        all_windows.append(make_windows(scaled, window, stride))
    n_features = len(SENSOR_TAGS)
    if not all_windows:
        return np.empty((0, window, n_features), dtype=np.float32)
    return np.concatenate(all_windows, axis=0)
# adding parser to allow user to specify window and stride lengths, as well as train/val/test split fractions
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=60, help="Window length in minutes")
    parser.add_argument("--stride", type=int, default=15, help="Stride in minutes")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        parser.error("--train-frac + --val-frac must be < 1.0 (remainder becomes the test split)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = get_sqlalchemy_engine()
    print(f"Pulling data from database ({os.environ.get('DB_NAME', '?')})...")
    df = load_data(engine)
    print(f"Pulled {len(df):,} rows")

    print("Validating schema...")
    df = validate_data(df)
    check_continuity(df)
    print("Schema OK")

    train_df, val_df, test_df = campaign_stratified_split(df, args.train_frac, args.val_frac)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        mix = split_df["process_grade"].value_counts(normalize=True).round(3).to_dict()
        print(f"  {name} grade mix: {mix}")

    scaler = StandardScaler()
    scaler.fit(train_df[SENSOR_TAGS].to_numpy())
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        windows = make_windows_per_campaign(split_df, scaler, args.window, args.stride)
        np.savez_compressed(OUT_DIR / f"{name}_windows.npz", X=windows)
        print(f"{name}: windows shape = {windows.shape}")

    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(
            {"sensor_tags": SENSOR_TAGS, "window_minutes": args.window, "stride_minutes": args.stride},
            f,
            indent=2,
        )

    print(f"\nSaved scaler, windows, and feature_columns.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
