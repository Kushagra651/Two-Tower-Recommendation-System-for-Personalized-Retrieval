"""
tests/test_dataset.py
------------------------
Run with: pytest tests/test_dataset.py -v
"""

import pandas as pd
import pytest
import torch

from src.data.dataset import TwoTowerDataset, build_eval_candidates, get_dataloader
from src.data.preprocess import NegativeSampler


@pytest.fixture
def toy_interactions():
    # 3 users, 10 items, a handful of interactions each
    return pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1, 2, 2, 2, 2],
        "item_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Timestamp": [100, 200, 300, 100, 200, 100, 200, 300, 400],
    })


class TestTwoTowerDataset:
    def test_length_matches_dataframe(self, toy_interactions):
        ds = TwoTowerDataset(toy_interactions)
        assert len(ds) == len(toy_interactions)

    def test_returns_long_tensors(self, toy_interactions):
        ds = TwoTowerDataset(toy_interactions)
        user_idx, item_idx = ds[0]
        assert user_idx.dtype == torch.long
        assert item_idx.dtype == torch.long

    def test_dataloader_batches_correctly(self, toy_interactions):
        loader = get_dataloader(toy_interactions, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        users, items = batch
        assert users.shape[0] == 4
        assert items.shape[0] == 4


class TestNegativeSampler:
    def test_negatives_never_include_positives(self, toy_interactions):
        sampler = NegativeSampler(toy_interactions, num_items=10, hard_ratio=0.5, seed=0)
        user0_positives = {0, 1, 2}

        for _ in range(20):  # repeat to catch any randomness-related leak
            negatives = sampler.sample(user_idx=0, num_negatives=5)
            assert len(set(negatives.tolist()) & user0_positives) == 0

    def test_returns_requested_count(self, toy_interactions):
        sampler = NegativeSampler(toy_interactions, num_items=10, hard_ratio=0.5, seed=0)
        negatives = sampler.sample(user_idx=1, num_negatives=3)
        assert len(negatives) == 3

    def test_unseen_user_samples_from_full_catalog(self, toy_interactions):
        # a user with no recorded positives should still get valid negatives
        sampler = NegativeSampler(toy_interactions, num_items=10, hard_ratio=0.5, seed=0)
        negatives = sampler.sample(user_idx=999, num_negatives=4)
        assert len(negatives) == 4
        assert all(0 <= n < 10 for n in negatives)


class TestEvalCandidates:
    def test_structure_and_negative_count(self, toy_interactions):
        eval_df = toy_interactions.groupby("user_idx").tail(1)  # last interaction per user as "eval"
        candidates = build_eval_candidates(eval_df, toy_interactions, num_items=10, num_negatives=4, seed=0)

        assert len(candidates) == eval_df.shape[0]
        for user_idx, true_item, negatives in candidates:
            assert isinstance(user_idx, int)
            assert isinstance(true_item, int)
            assert len(negatives) == 4
            assert true_item not in negatives