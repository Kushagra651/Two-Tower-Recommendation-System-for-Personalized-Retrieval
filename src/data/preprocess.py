"""
src/data/preprocess.py
------------------------
Takes the merged interactions table (output of merge_ml1m.py) and:
  1. Converts explicit ratings -> implicit positive feedback (Rating >= threshold).
  2. Re-indexes raw UserID/MovieID into contiguous 0-indexed user_idx/item_idx
     (required for nn.Embedding lookups).
  3. Provides a NegativeSampler used at training/eval time to draw items a
     user has NOT interacted with (random negatives + popularity-weighted
     "hard" negatives).

Usage:
    python src/data/preprocess.py \
        --in_path data/processed/merged_interactions.csv \
        --out_dir data/processed \
        --rating_threshold 4
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_merged(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"UserID", "MovieID", "Rating", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"merged_interactions.csv is missing columns: {missing}")
    return df


def to_implicit(df: pd.DataFrame, rating_threshold: int = 4) -> pd.DataFrame:
    """Keep only interactions that count as a 'positive' implicit signal."""
    implicit = df[df["Rating"] >= rating_threshold].copy()
    implicit = implicit[["UserID", "MovieID", "Timestamp"]].reset_index(drop=True)
    return implicit


def build_id_mappings(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Map raw sparse UserID/MovieID values to dense 0..N-1 indices."""
    unique_users = sorted(df["UserID"].unique())
    unique_items = sorted(df["MovieID"].unique())

    user2idx = {int(u): i for i, u in enumerate(unique_users)}
    item2idx = {int(m): i for i, m in enumerate(unique_items)}
    return user2idx, item2idx


def apply_id_mappings(df: pd.DataFrame, user2idx: dict, item2idx: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["UserID"].map(user2idx)
    df["item_idx"] = df["MovieID"].map(item2idx)
    return df[["user_idx", "item_idx", "Timestamp"]]


class NegativeSampler:
    """
    Draws item indices a given user has never interacted with.

    - `hard_ratio` fraction of negatives are drawn weighted by item popularity
      (popular-but-not-interacted items are "harder", more informative negatives).
    - the remainder are drawn uniformly at random from the full catalog.
    """

    def __init__(self, interactions: pd.DataFrame, num_items: int, hard_ratio: float = 0.5, seed: int = 42):
        self.num_items = num_items
        self.hard_ratio = hard_ratio
        self.rng = np.random.default_rng(seed)

        # set of items each user has interacted with (for exclusion)
        self.user_pos_items = (
            interactions.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        # popularity distribution over items, used for weighted hard-negative sampling
        item_counts = interactions["item_idx"].value_counts().reindex(
            range(num_items), fill_value=0
        ).values.astype(np.float64)
        item_counts = item_counts + 1.0  # smoothing so zero-count items are still samplable
        self.item_probs = item_counts / item_counts.sum()
        self.all_items = np.arange(num_items)

    def sample(self, user_idx: int, num_negatives: int) -> np.ndarray:
        pos_items = self.user_pos_items.get(user_idx, set())
        n_hard = int(round(num_negatives * self.hard_ratio))
        n_random = num_negatives - n_hard

        negatives = set()
        # popularity-weighted "hard" negatives
        while len(negatives) < n_hard:
            candidates = self.rng.choice(self.all_items, size=n_hard * 2, p=self.item_probs)
            for c in candidates:
                if c not in pos_items and c not in negatives:
                    negatives.add(int(c))
                if len(negatives) >= n_hard:
                    break

        # uniform random negatives
        while len(negatives) < n_hard + n_random:
            candidates = self.rng.integers(0, self.num_items, size=n_random * 2)
            for c in candidates:
                if c not in pos_items and c not in negatives:
                    negatives.add(int(c))
                if len(negatives) >= n_hard + n_random:
                    break

        return np.array(list(negatives)[:num_negatives])


def main(in_path: str, out_dir: str, rating_threshold: int):
    os.makedirs(out_dir, exist_ok=True)

    df = load_merged(in_path)
    print(f"Loaded {len(df)} raw interactions")

    implicit = to_implicit(df, rating_threshold)
    print(f"Kept {len(implicit)} implicit-positive interactions (Rating >= {rating_threshold})")

    user2idx, item2idx = build_id_mappings(implicit)
    print(f"Users: {len(user2idx)} | Items: {len(item2idx)}")

    indexed = apply_id_mappings(implicit, user2idx, item2idx)

    out_csv = os.path.join(out_dir, "processed_interactions.csv")
    indexed.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    id_maps_path = os.path.join(out_dir, "id_maps.json")
    with open(id_maps_path, "w") as f:
        json.dump(
            {
                "user2idx": {str(k): v for k, v in user2idx.items()},
                "item2idx": {str(k): v for k, v in item2idx.items()},
                "num_users": len(user2idx),
                "num_items": len(item2idx),
            },
            f,
        )
    print(f"Saved: {id_maps_path}")

    return indexed, user2idx, item2idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ratings to implicit feedback + build ID maps")
    parser.add_argument("--in_path", type=str, default="data/processed/merged_interactions.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--rating_threshold", type=int, default=4)
    args = parser.parse_args()

    main(args.in_path, args.out_dir, args.rating_threshold)