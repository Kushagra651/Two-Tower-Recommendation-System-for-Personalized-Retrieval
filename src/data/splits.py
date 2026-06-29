"""
src/data/splits.py
--------------------
Time-based (NOT random) train/val/test split — avoids leakage from the
future into training, and matches the standard "leave-one-out" protocol used
in the NCF paper and most two-tower recsys benchmarks:

    for each user, sorted by timestamp ascending:
        last interaction      -> test
        second-to-last        -> val
        everything before that -> train

Users with too few interactions to support this (< min_interactions) are
kept entirely in train (they simply aren't evaluated on).

Usage:
    python src/data/splits.py \
        --in_path data/processed/processed_interactions.csv \
        --out_dir data/processed
"""

import argparse
import os

import pandas as pd


def time_based_split(df: pd.DataFrame, min_interactions: int = 3) -> tuple:
    df = df.sort_values(["user_idx", "Timestamp"])

    train_rows, val_rows, test_rows = [], [], []

    for _, group in df.groupby("user_idx", sort=False):
        n = len(group)
        if n < min_interactions:
            train_rows.append(group)
            continue

        train_rows.append(group.iloc[: n - 2])
        val_rows.append(group.iloc[n - 2 : n - 1])
        test_rows.append(group.iloc[n - 1 :])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df


def main(in_path: str, out_dir: str, min_interactions: int):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path)
    train_df, val_df, test_df = time_based_split(df, min_interactions)

    print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Saved train.csv / val.csv / test.csv to {out_dir}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-based leave-one-out split")
    parser.add_argument("--in_path", type=str, default="data/processed/processed_interactions.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--min_interactions", type=int, default=3,
                         help="users with fewer interactions than this stay entirely in train")
    args = parser.parse_args()

    main(args.in_path, args.out_dir, args.min_interactions)