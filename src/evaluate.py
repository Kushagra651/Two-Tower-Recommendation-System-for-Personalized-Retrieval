"""
src/evaluate.py
-----------------
Ranking metrics computed on the sampled "1 positive + N negatives" candidate
sets built by src/data/dataset.py::build_eval_candidates — the same protocol
used in the NCF paper, so your NDCG@10/Recall@10 numbers are directly
comparable to published two-tower / NCF benchmarks on MovieLens.

Usage (after training):
    python src/evaluate.py --config configs/train_config.yaml \
        --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import torch

from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config
from src.utils.logging import get_logger


def recall_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    if true_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(true_item)  # 0-indexed
    return 1.0 / math.log2(rank + 2)            # +2 because rank is 0-indexed and log2(1)=0


def average_precision(ranked_items: List[int], true_item: int) -> float:
    """With exactly one relevant item, AP reduces to 1/rank (rank is 1-indexed)."""
    if true_item not in ranked_items:
        return 0.0
    rank = ranked_items.index(true_item) + 1
    return 1.0 / rank


@torch.no_grad()
def evaluate_model(
    model: TwoTowerModel,
    eval_candidates: List[Tuple[int, int, List[int]]],
    k: int = 10,
    device: str = "cpu",
) -> dict:
    model.eval()
    recalls, ndcgs, aps = [], [], []

    for user_idx, true_item, negatives in eval_candidates:
        candidate_items = [true_item] + list(negatives)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)
        user_tensor = torch.full((len(candidate_items),), user_idx, dtype=torch.long, device=device)

        scores = model.score(user_tensor, item_tensor)  # (num_candidates,)
        ranked_idx = torch.argsort(scores, descending=True).tolist()
        ranked_items = [candidate_items[i] for i in ranked_idx]

        recalls.append(recall_at_k(ranked_items, true_item, k))
        ndcgs.append(ndcg_at_k(ranked_items, true_item, k))
        aps.append(average_precision(ranked_items, true_item))

    return {
        f"Recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"NDCG@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "MAP": sum(aps) / len(aps) if aps else 0.0,
        "num_eval_users": len(eval_candidates),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained two-tower checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    import pandas as pd
    from src.data.dataset import build_eval_candidates

    cfg = load_config(args.config)
    logger = get_logger("evaluate", cfg.train.log_dir)

    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users, num_items = id_maps["num_users"], id_maps["num_items"]

    model = TwoTowerModel(num_users, num_items, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout)
    model.load_state_dict(torch.load(args.checkpoint, map_location=cfg.train.device))
    model.to(cfg.train.device)

    train_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    eval_df = val_df if args.split == "val" else test_df
    candidates = build_eval_candidates(eval_df, full_history, num_items, cfg.data.num_eval_negatives)

    metrics = evaluate_model(model, candidates, k=cfg.train.top_k, device=cfg.train.device)
    logger.info(f"[{args.split}] {metrics}")

    os.makedirs(cfg.train.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.train.results_dir, f"metrics_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()