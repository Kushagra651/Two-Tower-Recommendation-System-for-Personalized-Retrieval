"""
src/models/baseline_mf.py
--------------------------
BPR Matrix Factorization baseline.

Implements the same .score() interface as TwoTowerModel so it can be dropped
directly into evaluate_model() from src/evaluate.py without any changes.

    score(user_idx, item_idx) -> dot product of user & item embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, pos_item_idx: torch.Tensor, neg_item_idx: torch.Tensor):
        """Returns BPR loss for a batch of (user, pos_item, neg_item) triples."""
        user_emb = self.user_embedding(user_idx)       # (B, D)
        pos_emb  = self.item_embedding(pos_item_idx)   # (B, D)
        neg_emb  = self.item_embedding(neg_item_idx)   # (B, D)

        pos_score = (user_emb * pos_emb).sum(dim=-1)   # (B,)
        neg_score = (user_emb * neg_emb).sum(dim=-1)   # (B,)

        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Pointwise score — same interface as TwoTowerModel.score().
        Called by evaluate_model() in src/evaluate.py with no changes needed.
        """
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        return (user_emb * item_emb).sum(dim=-1)