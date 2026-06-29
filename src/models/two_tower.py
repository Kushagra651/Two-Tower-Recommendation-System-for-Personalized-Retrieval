"""
src/models/two_tower.py
-------------------------
Core architecture for the project:

    user_idx -> [User Tower] -> user_embedding  \
                                                   > dot product -> score
    item_idx -> [Item Tower] -> item_embedding   /

Each tower is an ID embedding followed by a small MLP, L2-normalized at the
output so the dot product is a cosine similarity (more stable training with
the in-batch contrastive loss in losses.py, and what FAISS IndexFlatIP/HNSW
expects downstream in src/retrieval/).

Side features (genres, demographics) aren't wired in yet — the `extra_dim`
hook on Tower exists for that extension later without changing call sites.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """A single tower: ID embedding -> MLP -> L2-normalized output embedding."""

    def __init__(self, num_ids: int, embedding_dim: int, hidden_dims: List[int], dropout: float = 0.1, extra_dim: int = 0):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        nn.init.normal_(self.id_embedding.weight, std=0.01)

        layers = []
        in_dim = embedding_dim + extra_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # final projection back to embedding_dim so user/item embeddings are comparable
        self.out_proj = nn.Linear(in_dim, embedding_dim)

    def forward(self, ids: torch.Tensor, extra_features: torch.Tensor = None) -> torch.Tensor:
        x = self.id_embedding(ids)
        if extra_features is not None:
            x = torch.cat([x, extra_features], dim=-1)
        x = self.mlp(x)
        x = self.out_proj(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.user_tower = Tower(num_users, embedding_dim, hidden_dims, dropout)
        self.item_tower = Tower(num_items, embedding_dim, hidden_dims, dropout)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        user_emb = self.user_tower(user_idx)
        item_emb = self.item_tower(item_idx)
        return user_emb, item_emb

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Pointwise score for a single (user, item) batch — dot product of normalized embeddings."""
        user_emb, item_emb = self.forward(user_idx, item_idx)
        return (user_emb * item_emb).sum(dim=-1)

    @torch.no_grad()
    def all_item_embeddings(self, item_idx_batch: torch.Tensor) -> torch.Tensor:
        """Used by retrieval/build_index.py to export the full item embedding matrix."""
        return self.item_tower(item_idx_batch)

    @torch.no_grad()
    def user_embedding_for(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_idx)