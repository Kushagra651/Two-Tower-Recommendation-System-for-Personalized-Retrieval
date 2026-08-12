"""
src/run_ablation.py
--------------------
Runs three training runs varying only embedding_dim (32, 64, 128).
Logs NDCG@10, Recall@10, MAP and time-per-epoch for each.
Saves results to results/ablation_results.json.

Usage:
    python -m src.run_ablation --config configs/train_config.yaml --with_features
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_eval_candidates, get_dataloader
from src.data.features import load_user_features, load_item_features
from src.models.losses import in_batch_contrastive_loss
from src.models.two_tower import TwoTowerModel
from src.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.logging import get_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_eval(cfg, embedding_dim, user_features, item_features,
                   train_df, val_df, test_df, full_history,
                   with_features, device, logger):

    set_seed(cfg.train.seed)

    num_users = None
    num_items = None
    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)
    num_users = id_maps["num_users"]
    num_items = id_maps["num_items"]

    train_loader = get_dataloader(
        train_df, cfg.train.batch_size, shuffle=True,
        user_features=user_features if with_features else None,
        item_features=item_features if with_features else None,
    )
    val_candidates  = build_eval_candidates(val_df,  full_history, num_items, cfg.data.num_eval_negatives)
    test_candidates = build_eval_candidates(test_df, full_history, num_items, cfg.data.num_eval_negatives)

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        with_features=with_features,
    ).to(device)

    if with_features:
        model.set_feature_tensors(user_features.to(device), item_features.to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    best_ndcg  = -1.0
    best_state = None
    ndcg_key   = f"NDCG@{cfg.train.top_k}"
    epoch_times = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, num_batches = 0.0, 0

        for batch in train_loader:
            if with_features:
                user_idx, item_idx, u_feats, i_feats = batch
                user_idx = user_idx.to(device)
                item_idx = item_idx.to(device)
                u_feats  = u_feats.to(device)
                i_feats  = i_feats.to(device)
                user_emb, item_emb = model(user_idx, item_idx, u_feats, i_feats)
            else:
                user_idx, item_idx = batch
                user_idx, item_idx = user_idx.to(device), item_idx.to(device)
                user_emb, item_emb = model(user_idx, item_idx)

            loss = in_batch_contrastive_loss(user_emb, item_emb, cfg.train.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if epoch % cfg.train.eval_every == 0:
            val_metrics = evaluate_model(model, val_candidates, k=cfg.train.top_k, device=device)
            if val_metrics[ndcg_key] > best_ndcg:
                best_ndcg  = val_metrics[ndcg_key]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info(f"  dim={embedding_dim} | epoch={epoch} | loss={total_loss/max(num_batches,1):.4f} | best_val_NDCG={best_ndcg:.4f}")

    # load best and eval on test
    model.load_state_dict(best_state)
    model.to(device)
    if with_features:
        model.set_feature_tensors(user_features.to(device), item_features.to(device))

    test_metrics = evaluate_model(model, test_candidates, k=cfg.train.top_k, device=device)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    return {
        "embedding_dim":    embedding_dim,
        f"NDCG@{cfg.train.top_k}":   round(test_metrics[f"NDCG@{cfg.train.top_k}"],   4),
        f"Recall@{cfg.train.top_k}": round(test_metrics[f"Recall@{cfg.train.top_k}"], 4),
        "MAP":              round(test_metrics["MAP"], 4),
        "avg_epoch_time_s": round(avg_epoch_time, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="configs/train_config.yaml")
    parser.add_argument("--with_features", action="store_true")
    parser.add_argument("--dims",         type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg.train.device
    logger = get_logger("ablation", cfg.train.log_dir)

    with open(os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)) as f:
        id_maps = json.load(f)

    train_df     = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.train_file))
    val_df       = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.val_file))
    test_df      = pd.read_csv(os.path.join(cfg.data.processed_dir, cfg.data.test_file))
    full_history = pd.concat([train_df, val_df, test_df], ignore_index=True)

    user_features, item_features = None, None
    if args.with_features:
        id_maps_path  = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)
        user_features = load_user_features("data/raw/users.dat", id_maps_path)
        item_features = load_item_features("data/raw/movies.dat", id_maps_path)

    results = []
    for dim in args.dims:
        logger.info(f"\nTraining embedding_dim={dim} ...")
        row = train_and_eval(
            cfg, dim, user_features, item_features,
            train_df, val_df, test_df, full_history,
            with_features=args.with_features,
            device=device, logger=logger,
        )
        results.append(row)
        logger.info(f"  dim={dim} result: {row}")

    os.makedirs(cfg.train.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.train.results_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAblation complete. Results saved to {out_path}")
    logger.info("\n| embedding_dim | NDCG@10 | Recall@10 | MAP  | Avg epoch time |")
    logger.info("|---|---|---|---|---|")
    for r in results:
        logger.info(f"| {r['embedding_dim']} | {r['NDCG@10']} | {r['Recall@10']} | {r['MAP']} | {r['avg_epoch_time_s']}s |")


if __name__ == "__main__":
    main()