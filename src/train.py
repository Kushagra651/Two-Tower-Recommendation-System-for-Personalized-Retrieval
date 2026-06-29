"""
src/train.py
-------------
Config-driven training loop for the two-tower model.

Usage:
    python src/train.py --config configs/train_config.yaml
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_eval_candidates, get_dataloader
from src.models.losses import in_batch_contrastive_loss
from src.models.two_tower import TwoTowerModel
from src.utils.config import config_to_dict, load_config
from src.utils.logging import get_logger
from src.evaluate import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, dataloader, optimizer, temperature, device, logger, epoch):
    model.train()
    total_loss, num_batches = 0.0, 0

    for user_idx, item_idx in dataloader:
        user_idx, item_idx = user_idx.to(device), item_idx.to(device)

        user_emb, item_emb = model(user_idx, item_idx)
        loss = in_batch_contrastive_loss(user_emb, item_emb, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f}")
    return avg_loss


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.train.seed)
    device = cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"

    logger = get_logger("train", cfg.train.log_dir)
    logger.info(f"Config: {config_to_dict(cfg)}")

    # --- load data ---
    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users, num_items = id_maps["num_users"], id_maps["num_items"]
    logger.info(f"num_users={num_users} num_items={num_items}")

    train_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    train_loader = get_dataloader(train_df, cfg.train.batch_size, shuffle=True)
    val_candidates = build_eval_candidates(val_df, full_history, num_items, cfg.data.num_eval_negatives)

    # --- model / optimizer ---
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.results_dir, exist_ok=True)
    best_ndcg, best_path = -1.0, os.path.join(cfg.train.checkpoint_dir, "best_model.pt")

    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.train.temperature, device, logger, epoch)

        record = {"epoch": epoch, "train_loss": train_loss}

        if val_candidates and epoch % cfg.train.eval_every == 0:
            val_metrics = evaluate_model(model, val_candidates, k=cfg.train.top_k, device=device)
            logger.info(f"Epoch {epoch} | val={val_metrics}")
            record.update(val_metrics)

            ndcg_key = f"NDCG@{cfg.train.top_k}"
            if val_metrics[ndcg_key] > best_ndcg:
                best_ndcg = val_metrics[ndcg_key]
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best model saved ({ndcg_key}={best_ndcg:.4f}) -> {best_path}")

        history.append(record)

    # if nothing was ever saved (e.g. no eval candidates), save the final model
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        logger.info(f"Saved final model -> {best_path}")

    metrics_path = os.path.join(cfg.train.results_dir, "train_history.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history -> {metrics_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the two-tower recommender")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    main(args.config)