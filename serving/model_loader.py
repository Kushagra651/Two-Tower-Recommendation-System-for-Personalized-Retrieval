"""
serving/model_loader.py
--------------------------
Loads the trained two-tower model + FAISS index + ID mappings + movie titles
exactly once, at FastAPI startup (see the `lifespan` hook in serving/app.py).
"""

import json
import os
from dataclasses import dataclass, field

import faiss
import torch

from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config


@dataclass
class ModelBundle:
    model: TwoTowerModel
    index: faiss.Index
    user2idx: dict              # raw UserID (int) -> internal user_idx
    idx2item: dict               # internal item_idx (int) -> raw MovieID (int)
    movie_titles: dict = field(default_factory=dict)  # raw MovieID (int) -> Title (str)
    movie_genres: dict = field(default_factory=dict)  # raw MovieID (int) -> Genres (str)
    top_k_default: int = 10


def load_movie_metadata(movies_csv_path: str, movies_dat_path: str) -> tuple:
    """
    Prefers data/processed/movies.csv (built by src/data/build_movie_lookup.py
    from the merged interactions table — no dependency on the raw .dat file
    being present at serving time). Falls back to parsing movies.dat directly
    (format: MovieID::Title::Genres) if the CSV isn't there.
    Returns (titles_dict, genres_dict), both empty if neither source exists.
    """
    titles, genres = {}, {}

    if os.path.exists(movies_csv_path):
        import csv
        with open(movies_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = int(row["MovieID"])
                titles[movie_id] = row.get("Title", "")
                genres[movie_id] = row.get("Genres", "")
        return titles, genres

    if os.path.exists(movies_dat_path):
        with open(movies_dat_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 2:
                    movie_id, title = int(parts[0]), parts[1]
                    titles[movie_id] = title
                    genres[movie_id] = parts[2] if len(parts) > 2 else ""

    return titles, genres


def load_bundle(
    config_path: str = "configs/train_config.yaml",
    checkpoint_path: str = "checkpoints/best_model.pt",
    index_dir: str = "embeddings",
    id_maps_path: str = None,
    movies_csv_path: str = "data/processed/movies.csv",
    movies_dat_path: str = "data/raw/movies.dat",
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

    movie_titles, movie_genres = load_movie_metadata(movies_csv_path, movies_dat_path)

    return ModelBundle(
        model=model,
        index=index,
        user2idx=user2idx,
        idx2item=idx2item,
        movie_titles=movie_titles,
        movie_genres=movie_genres,
        top_k_default=cfg.train.top_k,
    )