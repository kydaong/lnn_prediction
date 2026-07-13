"""Generate synthetic compressor data in memory and insert it straight into LNN_data.
"""
import argparse
import time

from db_connection import get_sqlalchemy_engine
from generate_compressor_normal_data import generate

TABLE_NAME = "compressor_normal_operation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=182, help="Duration in days (default ~6 months)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start timestamp")
    parser.add_argument("--freq", type=str, default="1min", help="Sample interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (vary this for a different run)")
    parser.add_argument("--chunksize", type=int, default=5000)
    args = parser.parse_args()

    df = generate(n_days=args.days, start=args.start, freq=args.freq, seed=args.seed)
    print(f"Generated {len(df):,} rows x {df.shape[1]} columns "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()})")

    engine = get_sqlalchemy_engine()
    start_time = time.time()
    df.to_sql(
        TABLE_NAME,
        engine,
        schema="dbo",
        if_exists="append",
        index=False,
        chunksize=args.chunksize,
        method=None,  # let fast_executemany (set on the engine) do the batching
    )
    elapsed = time.time() - start_time
    print(f"Inserted {len(df):,} rows into dbo.{TABLE_NAME} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
