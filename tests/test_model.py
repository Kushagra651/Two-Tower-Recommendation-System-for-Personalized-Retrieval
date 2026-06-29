"""
tests/test_model.py
----------------------
Run with: pytest tests/test_model.py -v
"""

import torch

from src.models.baseline_mf import MatrixFactorization
from src.models.losses import bpr_loss, in_batch_contrastive_loss
from src.models.two_tower import Tower, TwoTowerModel


class TestTower:
    def test_output_shape(self):
        tower = Tower(num_ids=50, embedding_dim=32, hidden_dims=[64, 32])
        ids = torch.randint(0, 50, (8,))
        out = tower(ids)
        assert out.shape == (8, 32)

    def test_output_is_l2_normalized(self):
        tower = Tower(num_ids=50, embedding_dim=32, hidden_dims=[64, 32])
        ids = torch.randint(0, 50, (8,))
        out = tower(ids)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)


class TestTwoTowerModel:
    def test_forward_shapes(self):
        model = TwoTowerModel(num_users=20, num_items=30, embedding_dim=16, hidden_dims=[32, 16])
        user_idx = torch.randint(0, 20, (8,))
        item_idx = torch.randint(0, 30, (8,))
        user_emb, item_emb = model(user_idx, item_idx)
        assert user_emb.shape == (8, 16)
        assert item_emb.shape == (8, 16)

    def test_score_is_scalar_per_pair(self):
        model = TwoTowerModel(num_users=20, num_items=30, embedding_dim=16)
        user_idx = torch.randint(0, 20, (8,))
        item_idx = torch.randint(0, 30, (8,))
        scores = model.score(user_idx, item_idx)
        assert scores.shape == (8,)
        # cosine similarity of two L2-normalized vectors must be in [-1, 1]
        assert torch.all(scores >= -1.0001) and torch.all(scores <= 1.0001)

    def test_all_item_embeddings_matches_item_tower(self):
        # eval() disables dropout — without it, two forward passes are stochastic
        # and won't match even with identical weights/inputs.
        model = TwoTowerModel(num_users=20, num_items=30, embedding_dim=16)
        model.eval()
        idx = torch.arange(30)
        all_emb = model.all_item_embeddings(idx)
        with torch.no_grad():
            direct_emb = model.item_tower(idx)
        assert torch.allclose(all_emb, direct_emb)


class TestMatrixFactorization:
    def test_output_shape(self):
        mf = MatrixFactorization(num_users=20, num_items=30, embedding_dim=8)
        user_idx = torch.randint(0, 20, (8,))
        item_idx = torch.randint(0, 30, (8,))
        scores = mf(user_idx, item_idx)
        assert scores.shape == (8,)


class TestLosses:
    def test_in_batch_contrastive_loss_is_positive_scalar(self):
        user_emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        item_emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss = in_batch_contrastive_loss(user_emb, item_emb, temperature=0.1)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_in_batch_loss_decreases_when_pairs_are_identical(self):
        # if user_emb == item_emb row-for-row, the diagonal similarity is maximal (=1),
        # so loss should be much lower than for random unrelated embeddings.
        torch.manual_seed(0)
        aligned = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        random_other = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)

        aligned_loss = in_batch_contrastive_loss(aligned, aligned, temperature=0.1)
        misaligned_loss = in_batch_contrastive_loss(aligned, random_other, temperature=0.1)

        assert aligned_loss.item() < misaligned_loss.item()

    def test_bpr_loss_rewards_correct_ranking(self):
        pos_scores = torch.tensor([2.0, 2.0])
        neg_scores = torch.tensor([0.5, 0.5])
        good_loss = bpr_loss(pos_scores, neg_scores)

        # flip pos/neg -> should be a much worse (higher) loss
        bad_loss = bpr_loss(neg_scores, pos_scores)

        assert good_loss.item() < bad_loss.item()