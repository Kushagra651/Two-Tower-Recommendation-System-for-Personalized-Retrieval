"""
src/train.py
-------------
Config-driven training loop for the two-tower model.

Usage:
    # ID-only (original behaviour):
    python -m src.train --config configs/train_config.yaml

    # With side features (user demographics + item genres):
    python -m src.train --config configs/train_config.yaml --with_features
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


def train_one_epoch(model, dataloader, optimizer, temperature, device, logger, epoch, with_features=False):
    model.train()
    total_loss, num_batches = 0.0, 0

    for batch in dataloader:
        if with_features:
            user_idx, item_idx, user_feats, item_feats = batch
            user_idx  = user_idx.to(device)
            item_idx  = item_idx.to(device)
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            user_emb, item_emb = model(user_idx, item_idx, user_feats, item_feats)
        else:
            user_idx, item_idx = batch
            user_idx, item_idx = user_idx.to(device), item_idx.to(device)
            user_emb, item_emb = model(user_idx, item_idx)

        loss = in_batch_contrastive_loss(user_emb, item_emb, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f}")
    return avg_loss


def main(config_path: str, with_features: bool = False):
    cfg = load_config(config_path)
    set_seed(cfg.train.seed)
    device = cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"

    logger = get_logger("train", cfg.train.log_dir)
    logger.info(f"Config: {config_to_dict(cfg)}  with_features={with_features}")

    # --- load data ---
    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users, num_items = id_maps["num_users"], id_maps["num_items"]
    logger.info(f"num_users={num_users} num_items={num_items}")

    train_df     = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df       = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df      = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # --- load feature tensors if requested ---
    user_features, item_features = None, None
    if with_features:
        from src.data.features import load_user_features, load_item_features
        id_maps_path = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)
        user_features = load_user_features("data/raw/users.dat", id_maps_path)
        item_features = load_item_features("data/raw/movies.dat", id_maps_path)
        logger.info(f"Loaded user_features {tuple(user_features.shape)}, item_features {tuple(item_features.shape)}")

    train_loader   = get_dataloader(train_df, cfg.train.batch_size, shuffle=True,
                                    user_features=user_features, item_features=item_features)
    val_candidates = build_eval_candidates(val_df, full_history, num_items, cfg.data.num_eval_negatives)

    # --- model / optimizer ---
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        with_features=with_features,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Give the model a reference to feature tensors so score() / evaluate_model() work unchanged
    if with_features:
        model.set_feature_tensors(user_features, item_features)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.results_dir,    exist_ok=True)

    suffix    = "_with_features" if with_features else ""
    best_path = os.path.join(cfg.train.checkpoint_dir, f"best_model{suffix}.pt")
    best_ndcg = -1.0
    ndcg_key  = f"NDCG@{cfg.train.top_k}"

    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            cfg.train.temperature, device, logger, epoch,
            with_features=with_features,
        )

        record = {"epoch": epoch, "train_loss": train_loss}

        if val_candidates and epoch % cfg.train.eval_every == 0:
            val_metrics = evaluate_model(model, val_candidates, k=cfg.train.top_k, device=device)
            logger.info(f"Epoch {epoch} | val={val_metrics}")
            record.update(val_metrics)

            if val_metrics[ndcg_key] > best_ndcg:
                best_ndcg = val_metrics[ndcg_key]
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best model saved ({ndcg_key}={best_ndcg:.4f}) -> {best_path}")

        history.append(record)

    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        logger.info(f"Saved final model -> {best_path}")

    # --- test evaluation ---
    logger.info("Loading best checkpoint for test evaluation…")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_candidates = build_eval_candidates(test_df, full_history, num_items, cfg.data.num_eval_negatives)
    test_metrics    = evaluate_model(model, test_candidates, k=cfg.train.top_k, device=device)
    logger.info(f"[test] {test_metrics}")

    out_name = f"metrics_two_tower{suffix}.json"
    out_path = os.path.join(cfg.train.results_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics -> {out_path}")

    history_path = os.path.join(cfg.train.results_dir, f"train_history{suffix}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the two-tower recommender")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--with_features", action="store_true",
                        help="Include user demographics + item genre features in both towers")
    args = parser.parse_args()

    main(args.config, with_features=args.with_features)
