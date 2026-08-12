"""
src/train_mf.py
----------------
Training loop for BPR-MF baseline.

Usage:
    python src/train_mf.py --config configs/train_config.yaml

Saves best checkpoint to checkpoints/best_model_mf.pt
Saves metrics    to results/metrics_mf.json
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import build_eval_candidates
from src.models.baseline_mf import MatrixFactorization
from src.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.logging import get_logger


# ---------------------------------------------------------------------------
# BPR dataset: yields (user_idx, pos_item_idx, neg_item_idx)
# ---------------------------------------------------------------------------

class BPRDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, num_items: int):
        self.users     = interactions_df["user_idx"].values
        self.pos_items = interactions_df["item_idx"].values
        self.num_items = num_items
        # build set of positives per user for clean negative sampling
        self.user_pos = (
            interactions_df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos  = self.pos_items[idx]
        # sample a true negative
        neg = random.randint(0, self.num_items - 1)
        while neg in self.user_pos.get(user, set()):
            neg = random.randint(0, self.num_items - 1)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            torch.tensor(neg,  dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config_path: str):
    cfg    = load_config(config_path)
    device = cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
    set_seed(cfg.train.seed)

    logger = get_logger("train_mf", cfg.train.log_dir)

    # --- load data ---
    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users = id_maps["num_users"]
    num_items = id_maps["num_items"]
    logger.info(f"num_users={num_users}  num_items={num_items}")

    train_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df   = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df  = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    train_dataset = BPRDataset(train_df, num_items)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_candidates = build_eval_candidates(
        val_df, full_history, num_items, cfg.data.num_eval_negatives
    )

    # --- model / optimizer ---
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg.model.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.results_dir,    exist_ok=True)

    best_ndcg = -1.0
    best_path = os.path.join(cfg.train.checkpoint_dir, "best_model_mf.pt")
    ndcg_key  = f"NDCG@{cfg.train.top_k}"

    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        # --- train ---
        model.train()
        total_loss, num_batches = 0.0, 0

        for user_idx, pos_idx, neg_idx in train_loader:
            user_idx = user_idx.to(device)
            pos_idx  = pos_idx.to(device)
            neg_idx  = neg_idx.to(device)

            loss = model(user_idx, pos_idx, neg_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f}")
        record = {"epoch": epoch, "train_loss": avg_loss}

        # --- eval ---
        if val_candidates and epoch % cfg.train.eval_every == 0:
            val_metrics = evaluate_model(
                model, val_candidates, k=cfg.train.top_k, device=device
            )
            logger.info(f"Epoch {epoch} | val={val_metrics}")
            record.update(val_metrics)

            if val_metrics[ndcg_key] > best_ndcg:
                best_ndcg = val_metrics[ndcg_key]
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best MF saved ({ndcg_key}={best_ndcg:.4f}) -> {best_path}")

        history.append(record)

    # --- test eval on best checkpoint ---
    logger.info("Loading best checkpoint for test evaluation…")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_candidates = build_eval_candidates(
        test_df, full_history, num_items, cfg.data.num_eval_negatives
    )
    test_metrics = evaluate_model(
        model, test_candidates, k=cfg.train.top_k, device=device
    )
    logger.info(f"[test] {test_metrics}")

    out_path = os.path.join(cfg.train.results_dir, "metrics_mf.json")
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved MF test metrics -> {out_path}")

    # save training history too
    history_path = os.path.join(cfg.train.results_dir, "train_history_mf.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)