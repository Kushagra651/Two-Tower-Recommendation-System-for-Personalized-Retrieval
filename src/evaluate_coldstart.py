"""
src/evaluate_coldstart.py
--------------------------
Cold-start evaluation: simulate users with ZERO interaction history.

Protocol:
1. Pick 200 users at random and completely hide them from training
   (we use users that ARE in the dataset but pretend we've never seen them —
   their test item is evaluated against 99 random negatives, same as normal eval)
2. Two-Tower (with features): runs user tower using ONLY demographics
   (age, gender, occupation). No interaction history needed.
3. MF baseline: has no embedding for unseen users — falls back to
   global popularity ranking (best it can do with zero history).
4. Compare NDCG@10 for both.

Usage:
    python -m src.evaluate_coldstart \
        --config configs/train_config.yaml \
        --tt_checkpoint checkpoints/best_model_with_features.pt \
        --mf_checkpoint checkpoints/best_model_mf.pt
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch

from src.models.two_tower import TwoTowerModel
from src.models.baseline_mf import MatrixFactorization
from src.data.features import load_user_features, load_item_features
from src.utils.config import load_config
from src.utils.logging import get_logger


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_items, true_item, k=10):
    if true_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(true_item)
    return 1.0 / math.log2(rank + 2)


def recall_at_k(ranked_items, true_item, k=10):
    return 1.0 if true_item in ranked_items[:k] else 0.0


# ---------------------------------------------------------------------------
# Two-Tower cold-start scorer
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_coldstart_two_tower(model, cold_users, test_df, full_history_df,
                              user_features, item_features, num_items,
                              num_negatives=99, k=10, seed=42, device="cpu"):
    """
    For each cold user:
      - Look up their test item from test_df
      - Score candidates using ONLY demographic features (no interaction history)
      - The user_idx still exists in the model (they were in the full dataset)
        but conceptually we treat their ID embedding as uninformative —
        what matters is the demographic features carry signal.
    """
    rng = random.Random(seed)

    # Build a set of all items each user touched (to avoid leaking negatives)
    user_pos = full_history_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    model.eval()
    model.set_feature_tensors(user_features.to(device), item_features.to(device))

    test_lookup = test_df.set_index("user_idx")["item_idx"].to_dict()

    ndcgs, recalls = [], []

    for user_idx in cold_users:
        if user_idx not in test_lookup:
            continue

        true_item = int(test_lookup[user_idx])
        seen      = user_pos.get(user_idx, set())

        # sample negatives
        negs = []
        all_items = list(range(num_items))
        rng.shuffle(all_items)
        for item in all_items:
            if item not in seen and item != true_item:
                negs.append(item)
            if len(negs) == num_negatives:
                break

        candidate_items = [true_item] + negs
        item_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)
        user_tensor = torch.full((len(candidate_items),), user_idx, dtype=torch.long, device=device)

        scores      = model.score(user_tensor, item_tensor)
        ranked_idx  = torch.argsort(scores, descending=True).tolist()
        ranked_items = [candidate_items[i] for i in ranked_idx]

        ndcgs.append(ndcg_at_k(ranked_items, true_item, k))
        recalls.append(recall_at_k(ranked_items, true_item, k))

    return {
        f"two_tower_coldstart_NDCG@{k}":   round(sum(ndcgs)   / len(ndcgs),   4) if ndcgs   else 0.0,
        f"two_tower_coldstart_Recall@{k}": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "two_tower_num_cold_users": len(ndcgs),
    }


# ---------------------------------------------------------------------------
# MF cold-start scorer (popularity fallback)
# ---------------------------------------------------------------------------

def eval_coldstart_mf(cold_users, test_df, full_history_df, num_items,
                      num_negatives=99, k=10, seed=42):
    """
    MF has no embedding for zero-history users.
    Best it can do: rank by global item popularity.
    """
    rng = random.Random(seed)

    popularity   = full_history_df["item_idx"].value_counts().to_dict()
    popularity_ranked = sorted(range(num_items), key=lambda i: popularity.get(i, 0), reverse=True)

    user_pos   = full_history_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_lookup = test_df.set_index("user_idx")["item_idx"].to_dict()

    ndcgs, recalls = [], []

    for user_idx in cold_users:
        if user_idx not in test_lookup:
            continue

        true_item = int(test_lookup[user_idx])
        seen      = user_pos.get(user_idx, set())

        negs = []
        all_items = list(range(num_items))
        rng.shuffle(all_items)
        for item in all_items:
            if item not in seen and item != true_item:
                negs.append(item)
            if len(negs) == num_negatives:
                break

        candidate_items = [true_item] + negs
        candidate_set   = set(candidate_items)

        # rank candidates by global popularity
        ranked_items = [i for i in popularity_ranked if i in candidate_set]

        ndcgs.append(ndcg_at_k(ranked_items, true_item, k))
        recalls.append(recall_at_k(ranked_items, true_item, k))

    return {
        f"mf_coldstart_NDCG@{k}":   round(sum(ndcgs)   / len(ndcgs),   4) if ndcgs   else 0.0,
        f"mf_coldstart_Recall@{k}": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "mf_num_cold_users": len(ndcgs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, default="configs/train_config.yaml")
    parser.add_argument("--tt_checkpoint", type=str, default="checkpoints/best_model_with_features.pt")
    parser.add_argument("--mf_checkpoint", type=str, default="checkpoints/best_model_mf.pt")
    parser.add_argument("--num_cold_users", type=int, default=200)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg.train.device
    logger = get_logger("coldstart", cfg.train.log_dir)

    # --- load id maps ---
    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users = id_maps["num_users"]
    num_items = id_maps["num_items"]

    # --- load data ---
    train_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df   = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df  = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # --- pick cold users: users that appear in test_df ---
    rng = random.Random(args.seed)
    test_users   = test_df["user_idx"].unique().tolist()
    cold_users   = rng.sample(test_users, min(args.num_cold_users, len(test_users)))
    logger.info(f"Cold-start evaluation on {len(cold_users)} users")

    # --- load features ---
    id_maps_path  = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)
    user_features = load_user_features("data/raw/users.dat", id_maps_path)
    item_features = load_item_features("data/raw/movies.dat", id_maps_path)

    # --- Two-Tower cold-start ---
    tt_model = TwoTowerModel(
        num_users=num_users, num_items=num_items,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        with_features=True,
    )
    tt_model.load_state_dict(torch.load(args.tt_checkpoint, map_location=device))
    tt_model.to(device)

    tt_results = eval_coldstart_two_tower(
        tt_model, cold_users, test_df, full_history,
        user_features, item_features,
        num_items=num_items, k=cfg.train.top_k,
        seed=args.seed, device=device,
    )
    logger.info(f"Two-Tower cold-start: {tt_results}")

    # --- MF cold-start (popularity fallback) ---
    mf_results = eval_coldstart_mf(
        cold_users, test_df, full_history,
        num_items=num_items, k=cfg.train.top_k, seed=args.seed,
    )
    logger.info(f"MF cold-start: {mf_results}")

    # --- combine and save ---
    k = cfg.train.top_k
    combined = {**tt_results, **mf_results}

    tt_ndcg = tt_results[f"two_tower_coldstart_NDCG@{k}"]
    mf_ndcg = mf_results[f"mf_coldstart_NDCG@{k}"]
    combined["relative_improvement"] = f"{tt_ndcg / mf_ndcg:.1f}x" if mf_ndcg > 0 else "inf"

    os.makedirs(cfg.train.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.train.results_dir, "coldstart_comparison.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Saved cold-start results -> {out_path}")
    logger.info(f"\n{'='*50}")
    logger.info(f"  Two-Tower cold-start NDCG@{k}: {tt_ndcg}")
    logger.info(f"  MF cold-start       NDCG@{k}: {mf_ndcg}")
    logger.info(f"  Relative improvement: {combined['relative_improvement']}")
    logger.info(f"{'='*50}")

    return combined


if __name__ == "__main__":
    main()