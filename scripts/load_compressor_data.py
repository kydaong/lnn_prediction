"""Bulk-load the generated synthetic dataset into LNN_data.

Usage:
    .venv/Scripts/python.exe scripts/load_compressor_data.py
"""
import argparse
import time

import pandas as pd

from db_connection import get_sqlalchemy_engine

TABLE_NAME = "compressor_normal_operation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="data/processed/compressor_normal_operation.parquet",
        help="Parquet file produced by generate_compressor_normal_data.py",
    )
    parser.add_argument("--chunksize", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_parquet(args.source)
    print(f"Loaded {len(df):,} rows from {args.source}")

    engine = get_sqlalchemy_engine()
    start = time.time()
    df.to_sql(
        TABLE_NAME,
        engine,
        schema="dbo",
        if_exists="append",
        index=False,
        chunksize=args.chunksize,
        method=None,  # let fast_executemany (set on the engine) do the batching
    )
    elapsed = time.time() - start
    print(f"Inserted {len(df):,} rows into dbo.{TABLE_NAME} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
