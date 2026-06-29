"""
src/models/baseline_mf.py
----------------------------
Simple Matrix Factorization baseline — user embedding + item embedding +
bias terms, scored by dot product. This is the "before deep learning" point
of comparison referenced in notebooks/02_baseline_mf.ipynb and your README's
metrics table.

Train it with bpr_loss() from losses.py + NegativeSampler from preprocess.py
(it needs explicit (user, pos_item, neg_item) triples, unlike the two-tower
model which uses in-batch negatives).
"""

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_idx)
        i = self.item_embedding(item_idx)
        dot = (u * i).sum(dim=-1)
        bias = (
            self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
            + self.global_bias
        )
        return dot + bias