"""
src/retrieval/search.py
--------------------------
ItemRetriever: given a user_idx, computes the user embedding via the trained
user tower and does a FAISS top-K lookup over the pre-built item index.

Also includes a latency benchmark (p50/p99) over a sample of users, saved to
results/latency_benchmark.csv — this is the number that goes in your README
("sub-Xms p99 latency").

Usage:
    python src/retrieval/search.py \
        --config configs/train_config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --index_dir embeddings \
        --num_benchmark_users 200
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import os
import time

import faiss
import numpy as np
import torch

from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config


class ItemRetriever:
    def __init__(self, model: TwoTowerModel, index: faiss.Index, idx2item: dict):
        self.model = model
        self.index = index
        self.idx2item = idx2item
        self.model.eval()

    @torch.no_grad()
    def recommend(self, user_idx: int, k: int = 10):
        user_emb = self.model.user_embedding_for(torch.tensor([user_idx])).numpy().astype("float32")
        scores, indices = self.index.search(user_emb, k)
        results = [
            {"movie_id": self.idx2item[str(i)] if str(i) in self.idx2item else self.idx2item[i],
             "score": float(s)}
            for i, s in zip(indices[0], scores[0])
        ]
        return results


def load_retriever(config_path: str, checkpoint_path: str, index_dir: str) -> ItemRetriever:
    cfg = load_config(config_path)

    id_maps_path = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)
    with open(id_maps_path) as f:
        id_maps = json.load(f)
    num_users, num_items = id_maps["num_users"], id_maps["num_items"]

    model = TwoTowerModel(num_users, num_items, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    index = faiss.read_index(os.path.join(index_dir, "item_index.faiss"))
    with open(os.path.join(index_dir, "idx2item.json")) as f:
        idx2item = json.load(f)

    return ItemRetriever(model, index, idx2item), num_users


def benchmark_latency(retriever: ItemRetriever, user_indices: list, k: int = 10):
    latencies_ms = []
    for u in user_indices:
        start = time.perf_counter()
        retriever.recommend(u, k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies_ms)
    p50, p99 = float(np.percentile(arr, 50)), float(np.percentile(arr, 99))
    return p50, p99, latencies_ms


def main():
    parser = argparse.ArgumentParser(description="ANN search + latency benchmark")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--index_dir", type=str, default="embeddings")
    parser.add_argument("--num_benchmark_users", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    retriever, num_users = load_retriever(args.config, args.checkpoint, args.index_dir)

    # quick sanity check on a single user
    sample_result = retriever.recommend(0, args.top_k)
    print(f"Sample recommendations for user_idx=0: {sample_result}")

    sample_size = min(args.num_benchmark_users, num_users)
    rng = np.random.default_rng(42)
    user_sample = rng.integers(0, num_users, size=sample_size).tolist()

    p50, p99, latencies = benchmark_latency(retriever, user_sample, args.top_k)
    print(f"Latency over {sample_size} queries -> p50={p50:.3f}ms | p99={p99:.3f}ms")

    os.makedirs("results", exist_ok=True)
    import csv
    out_path = "results/latency_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_idx", "latency_ms"])
        for u, lat in zip(user_sample, latencies):
            writer.writerow([u, lat])
    print(f"Saved per-query latencies to {out_path}")


if __name__ == "__main__":
    main()