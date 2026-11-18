"""serving/model_loader.py"""
import json, os
from dataclasses import dataclass, field
import faiss, torch
from src.models.two_tower import TwoTowerModel
from src.data.features import load_user_features, load_item_features
from src.utils.config import load_config


@dataclass
class ModelBundle:
    model: TwoTowerModel
    index: faiss.Index
    user2idx: dict
    idx2item: dict
    user_features: torch.Tensor
    item_features: torch.Tensor
    seen_items: dict          # raw_user_id -> set of raw_movie_ids
    num_users: int            # original num_users; UNK = num_users
    movie_titles: dict = field(default_factory=dict)
    movie_genres: dict = field(default_factory=dict)
    top_k_default: int = 10


def load_movie_metadata(movies_csv, movies_dat):
    titles, genres = {}, {}
    if os.path.exists(movies_csv):
        import csv
        for row in csv.DictReader(open(movies_csv, encoding="utf-8")):
            mid = int(row["MovieID"])
            titles[mid] = row.get("Title", "")
            genres[mid] = row.get("Genres", "")
        return titles, genres
    if os.path.exists(movies_dat):
        for line in open(movies_dat, encoding="latin-1"):
            p = line.strip().split("::")
            if len(p) >= 2:
                mid = int(p[0])
                titles[mid] = p[1]
                genres[mid] = p[2] if len(p) > 2 else ""
    return titles, genres


def load_bundle(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/best_model_with_features_logq_fv_cold.pt",
    index_dir="embeddings",
    id_maps_path=None,
    movies_csv="data/processed/movies.csv",
    movies_dat="data/raw/movies.dat",
    interactions_csv="data/processed/processed_interactions.csv",
) -> ModelBundle:
    cfg = load_config(config_path)
    if id_maps_path is None:
        id_maps_path = os.path.join(cfg.data.processed_dir, cfg.data.id_maps_file)

    maps = json.load(open(id_maps_path))
    nu, ni = maps["num_users"], maps["num_items"]
    user2idx = {int(k): v for k, v in maps["user2idx"].items()}
    item2idx = {int(k): v for k, v in maps["item2idx"].items()}

    model = TwoTowerModel(nu + 1, ni, cfg.model.embedding_dim,
                          cfg.model.hidden_dims, cfg.model.dropout,
                          with_features=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    uf = load_user_features("data/raw/users.dat", id_maps_path)
    itf = load_item_features("data/raw/movies.dat", id_maps_path)
    model.set_feature_tensors(uf, itf)

    index = faiss.read_index(os.path.join(index_dir, "item_index.faiss"))
    idx2item = {int(k): int(v) for k, v in
                json.load(open(os.path.join(index_dir, "idx2item.json"))).items()}

    # build seen-items lookup (raw user_id -> set of raw movie_ids)
    seen = {}
    if os.path.exists(interactions_csv):
        import pandas as pd
        df = pd.read_csv(interactions_csv)
        raw_item_lookup = {v: int(k) for k, v in maps["item2idx"].items()}
        raw_user_lookup = {v: int(k) for k, v in maps["user2idx"].items()}
        for uid_idx, grp in df.groupby("user_idx"):
            raw_uid = raw_user_lookup.get(uid_idx)
            if raw_uid is not None:
                seen[raw_uid] = set(raw_item_lookup.get(i, -1)
                                    for i in grp["item_idx"])

    titles, genres = load_movie_metadata(movies_csv, movies_dat)
    # at the end of load_bundle, before return
    with torch.no_grad():
        all_items = torch.arange(ni)
        bundle_item_emb = model.all_item_embeddings(all_items, itf)
    return ModelBundle(model=model, index=index, user2idx=user2idx,
                       idx2item=idx2item, user_features=uf, item_features=itf,
                       seen_items=seen, num_users=nu,
                       movie_titles=titles, movie_genres=genres,
                       top_k_default=cfg.train.top_k)