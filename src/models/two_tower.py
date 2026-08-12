"""
src/models/two_tower.py
-------------------------
Core architecture for the project:

    user_idx -> [User Tower] -> user_embedding  \
                                                   > dot product -> score
    item_idx -> [Item Tower] -> item_embedding   /

Each tower is an ID embedding followed by a small MLP, L2-normalized at the
output so the dot product is a cosine similarity.

Side features:
    User: gender (1) + age embedding (4) + occupation embedding (8) = 13 dims
    Item: genre multi-hot (18 dims)

When with_features=True, call set_feature_tensors() after construction so
score() and evaluate_model() work without any changes to evaluate.py.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

AGE_BUCKETS   = 7
OCC_BUCKETS   = 21
AGE_EMB_DIM   = 4
OCC_EMB_DIM   = 8
GENDER_DIM    = 1
USER_FEAT_DIM = GENDER_DIM + AGE_EMB_DIM + OCC_EMB_DIM  # 13
ITEM_FEAT_DIM = 18  # genre multi-hot


class Tower(nn.Module):
    def __init__(self, num_ids, embedding_dim, hidden_dims, dropout=0.1, extra_dim=0):
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
        self.out_proj = nn.Linear(in_dim, embedding_dim)

    def forward(self, ids, extra_features=None):
        x = self.id_embedding(ids)
        if extra_features is not None:
            x = torch.cat([x, extra_features], dim=-1)
        x = self.mlp(x)
        x = self.out_proj(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64,
                 hidden_dims=None, dropout=0.1, with_features=False):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.with_features = with_features

        if with_features:
            self.age_embedding = nn.Embedding(AGE_BUCKETS, AGE_EMB_DIM)
            self.occ_embedding = nn.Embedding(OCC_BUCKETS, OCC_EMB_DIM)
            self.user_tower = Tower(num_users, embedding_dim, hidden_dims, dropout, extra_dim=USER_FEAT_DIM)
            self.item_tower = Tower(num_items, embedding_dim, hidden_dims, dropout, extra_dim=ITEM_FEAT_DIM)
        else:
            self.user_tower = Tower(num_users, embedding_dim, hidden_dims, dropout)
            self.item_tower = Tower(num_items, embedding_dim, hidden_dims, dropout)

        # Feature lookup buffers — populated via set_feature_tensors()
        self._user_feat_store = None
        self._item_feat_store = None

    def set_feature_tensors(self, user_features, item_features):
        """Store feature tensors so score() can look them up by index automatically."""
        self._user_feat_store = user_features
        self._item_feat_store = item_features

    def _build_user_extra(self, user_raw_feats):
        gender  = user_raw_feats[:, 0:1]
        age_idx = user_raw_feats[:, 1].long()
        occ_idx = user_raw_feats[:, 2].long()
        age_emb = self.age_embedding(age_idx)
        occ_emb = self.occ_embedding(occ_idx)
        return torch.cat([gender, age_emb, occ_emb], dim=-1)  # (B, 13)

    def forward(self, user_idx, item_idx, user_feats=None, item_feats=None):
        if self.with_features and user_feats is not None and item_feats is not None:
            user_emb = self.user_tower(user_idx, self._build_user_extra(user_feats))
            item_emb = self.item_tower(item_idx, item_feats)
        else:
            user_emb = self.user_tower(user_idx)
            item_emb = self.item_tower(item_idx)
        return user_emb, item_emb

    def score(self, user_idx, item_idx, user_feats=None, item_feats=None):
        """
        Works with evaluate_model() unchanged — looks up features from buffer
        when with_features=True and no explicit feats are passed.
        """
        if self.with_features:
            if user_feats is None and self._user_feat_store is not None:
                user_feats = self._user_feat_store[user_idx].to(user_idx.device)
            if item_feats is None and self._item_feat_store is not None:
                item_feats = self._item_feat_store[item_idx].to(item_idx.device)

        user_emb, item_emb = self.forward(user_idx, item_idx, user_feats, item_feats)
        return (user_emb * item_emb).sum(dim=-1)

    @torch.no_grad()
    def all_item_embeddings(self, item_idx_batch, item_feats=None):
        if self.with_features and item_feats is not None:
            return self.item_tower(item_idx_batch, item_feats)
        return self.item_tower(item_idx_batch)

    @torch.no_grad()
    def user_embedding_for(self, user_idx, user_feats=None):
        if self.with_features and user_feats is not None:
            return self.user_tower(user_idx, self._build_user_extra(user_feats))
        return self.user_tower(user_idx)