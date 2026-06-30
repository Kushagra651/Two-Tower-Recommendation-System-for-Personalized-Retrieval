"""
src/retrieval/build_index.py
-------------------------------
Exports the full item embedding matrix from a trained two-tower checkpoint
and builds a FAISS index over it for fast approximate nearest-neighbor (ANN)
retrieval at serving time.

Embeddings are L2-normalized (see Tower.forward), so inner product == cosine
similarity — this is why IndexFlatIP / IndexIVFFlat with METRIC_INNER_PRODUCT
is used instead of L2 distance.

Usage:
    python src/retrieval/build_index.py \
        --config configs/train_config.yaml \
        --retrieval_config configs/retrieval_config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --out_dir embeddings
"""

import argparse
import json
import os

import faiss
import numpy as np
import torch
import yaml

from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config


def load_retrieval_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def export_item_embeddings(model: TwoTowerModel, num_items: int, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        idx = torch.arange(start, end, dtype=torch.long)
        emb = model.all_item_embeddings(idx)
        chunks.append(emb.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype("float32")


def build_faiss_index(item_embeddings: np.ndarray, index_type: str, nlist: int) -> faiss.Index:
    embedding_dim = item_embeddings.shape[1]

    if index_type == "flat":
        # exact search — recommended default at ml-1m/ml-25m scale (<100K items)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(item_embeddings)

    elif index_type == "ivf":
        # approximate search — use once catalog grows large enough that flat search is too slow
        nlist = min(nlist, max(1, item_embeddings.shape[0] // 10))
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(item_embeddings)
        index.add(item_embeddings)

    else:
        raise ValueError(f"Unknown index_type '{index_type}' (expected 'flat' or 'ivf')")

    return index


def main(config_path: str, retrieval_config_path: str, checkpoint_path: str, out_dir: str):
    cfg = load_config(config_path)
    rcfg = load_retrieval_config(retrieval_config_path)

    id_maps_path = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)
    with open(id_maps_path) as f:
        id_maps = json.load(f)
    num_users, num_items = id_maps["num_users"], id_maps["num_items"]

    model = TwoTowerModel(num_users, num_items, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    item_embeddings = export_item_embeddings(model, num_items)
    print(f"Exported item embeddings: {item_embeddings.shape}")

    index = build_faiss_index(item_embeddings, rcfg["index_type"], rcfg.get("nlist", 100))
    if rcfg["index_type"] == "ivf":
        index.nprobe = rcfg.get("nprobe", 10)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "item_index.faiss"))
    np.save(os.path.join(out_dir, "item_embeddings.npy"), item_embeddings)

    # item_idx -> raw MovieID, needed to translate FAISS results back for the API
    idx2item = {v: int(k) for k, v in id_maps["item2idx"].items()}
    with open(os.path.join(out_dir, "idx2item.json"), "w") as f:
        json.dump(idx2item, f)

    print(f"Saved FAISS index + embeddings + idx2item map to: {out_dir}")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from trained item tower")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--retrieval_config", type=str, default="configs/retrieval_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="embeddings")
    args = parser.parse_args()

    main(args.config, args.retrieval_config, args.checkpoint, args.out_dir)