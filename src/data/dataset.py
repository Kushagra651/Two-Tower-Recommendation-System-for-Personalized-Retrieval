"""
src/data/dataset.py
---------------------
- TwoTowerDataset: positive-only (user_idx, item_idx) pairs for in-batch
  contrastive training. Negatives are NOT materialized here — they come for
  free from the other items in the same training batch (see losses.py).

- build_eval_candidates(): for each user in val/test, builds a
  (user_idx, true_item_idx, [negative_item_idx x N]) candidate set following
  the standard "1 positive + 99 sampled negatives" protocol from the NCF
  paper, so Recall@10/NDCG@10 are directly comparable to published numbers.
"""

from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import NegativeSampler


class TwoTowerDataset(Dataset):
    """Wraps a positives-only interactions dataframe (train.csv)."""

    def __init__(self, df: pd.DataFrame):
        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        return self.user_idx[idx], self.item_idx[idx]


def get_dataloader(df: pd.DataFrame, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = TwoTowerDataset(df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,  # drop last partial batch only during training (keeps in-batch negatives meaningful)
    )


def build_eval_candidates(
    eval_df: pd.DataFrame,
    full_history_df: pd.DataFrame,
    num_items: int,
    num_negatives: int = 99,
    seed: int = 42,
) -> List[Tuple[int, int, List[int]]]:
    """
    full_history_df should be the UNION of train+val+test interactions, so
    sampled negatives are guaranteed to be items the user never touched at
    any point (avoids leaking a "negative" that's secretly a future positive).
    """
    sampler = NegativeSampler(full_history_df, num_items=num_items, hard_ratio=0.5, seed=seed)

    candidates = []
    for row in eval_df.itertuples(index=False):
        user_idx, true_item = int(row.user_idx), int(row.item_idx)
        negatives = sampler.sample(user_idx, num_negatives).tolist()
        candidates.append((user_idx, true_item, negatives))

    return candidates