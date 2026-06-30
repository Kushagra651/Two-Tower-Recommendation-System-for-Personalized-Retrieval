"""
serving/model_loader.py
--------------------------
Loads the trained two-tower model + FAISS index + ID mappings exactly once,
at FastAPI startup (see the `lifespan` hook in serving/app.py), rather than
on every request.
"""

import json
import os
from dataclasses import dataclass

import faiss
import torch

from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config


@dataclass
class ModelBundle:
    model: TwoTowerModel
    index: faiss.Index
    user2idx: dict     # raw UserID (int) -> internal user_idx
    idx2item: dict     # internal item_idx (int) -> raw MovieID (int)
    top_k_default: int


def load_bundle(
    config_path: str = "configs/train_config.yaml",
    checkpoint_path: str = "checkpoints/best_model.pt",
    index_dir: str = "embeddings",
    id_maps_path: str = None,
) -> ModelBundle:
    cfg = load_config(config_path)

    if id_maps_path is None:
        id_maps_path = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)

    with open(id_maps_path) as f:
        id_maps = json.load(f)

    num_users, num_items = id_maps["num_users"], id_maps["num_items"]
    user2idx = {int(k): v for k, v in id_maps["user2idx"].items()}

    model = TwoTowerModel(num_users, num_items, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    index = faiss.read_index(os.path.join(index_dir, "item_index.faiss"))
    with open(os.path.join(index_dir, "idx2item.json")) as f:
        idx2item_raw = json.load(f)
    idx2item = {int(k): int(v) for k, v in idx2item_raw.items()}

    return ModelBundle(
        model=model,
        index=index,
        user2idx=user2idx,
        idx2item=idx2item,
        top_k_default=cfg.train.top_k,
    )