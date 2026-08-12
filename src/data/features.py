"""
src/data/features.py
---------------------
Parses users.dat and movies.dat into feature tensors.

User features (dim=13):
    - gender:     binary float (M=0, F=1)          → dim 1
    - age:        embedding index (7 buckets)        → dim 4  (embedded in model)
    - occupation: index 0-20                         → dim 8  (embedded in model)
    Raw tensor stored here: [gender, age_idx, occ_idx] (dim=3, embeddings done in model)

Item features (dim=18):
    - genres: multi-hot binary vector over 18 genres

Usage:
    from src.data.features import load_user_features, load_item_features
    user_feats = load_user_features("data/raw/users.dat", "data/processed/id_maps.json")
    item_feats = load_item_features("data/raw/movies.dat", "data/processed/id_maps.json")
"""

import json
import torch

# MovieLens-1M age bucket values → index
AGE_BUCKET_TO_IDX = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

# All 18 possible genres in MovieLens-1M (fixed order)
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}


def load_user_features(users_dat_path: str, id_maps_path: str) -> torch.Tensor:
    """
    Returns a tensor of shape (num_users, 3):
        col 0 — gender    (float: 0=M, 1=F)
        col 1 — age_idx   (long: 0-6)
        col 2 — occ_idx   (long: 0-20)

    Row i corresponds to user_idx i (0-indexed, mapped via id_maps.json).
    Missing users get zeros.
    """
    with open(id_maps_path) as f:
        id_maps = json.load(f)
    user_id_to_idx = {int(k): v for k, v in id_maps["user2idx"].items()}
    num_users = id_maps["num_users"]

    feats = torch.zeros(num_users, 3, dtype=torch.float32)

    with open(users_dat_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            user_id    = int(parts[0])
            gender     = 0.0 if parts[1] == "M" else 1.0
            age_idx    = float(AGE_BUCKET_TO_IDX.get(int(parts[2]), 0))
            occ_idx    = float(int(parts[3]))

            idx = user_id_to_idx.get(user_id)
            if idx is not None:
                feats[idx] = torch.tensor([gender, age_idx, occ_idx])

    return feats  # (num_users, 3)


def load_item_features(movies_dat_path: str, id_maps_path: str) -> torch.Tensor:
    """
    Returns a tensor of shape (num_items, 18) — multi-hot genre vector.
    Row i corresponds to item_idx i (0-indexed, mapped via id_maps.json).
    Missing items get all-zero genre vectors.
    """
    with open(id_maps_path) as f:
        id_maps = json.load(f)
    item_id_to_idx = {int(k): v for k, v in id_maps["item2idx"].items()}
    num_items = id_maps["num_items"]

    feats = torch.zeros(num_items, 18, dtype=torch.float32)

    with open(movies_dat_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts  = line.split("::")
            item_id = int(parts[0])
            genres  = parts[2].split("|")

            idx = item_id_to_idx.get(item_id)
            if idx is not None:
                for g in genres:
                    g_idx = GENRE_TO_IDX.get(g)
                    if g_idx is not None:
                        feats[idx, g_idx] = 1.0

    return feats  # (num_items, 18)


def save_features(users_dat_path: str, movies_dat_path: str, id_maps_path: str, out_dir: str):
    """Parse and save both feature tensors to disk."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    user_feats = load_user_features(users_dat_path, id_maps_path)
    item_feats = load_item_features(movies_dat_path, id_maps_path)

    torch.save(user_feats, os.path.join(out_dir, "user_features.pt"))
    torch.save(item_feats, os.path.join(out_dir, "item_features.pt"))
    print(f"Saved user_features.pt {tuple(user_feats.shape)} and item_features.pt {tuple(item_feats.shape)} to {out_dir}")


if __name__ == "__main__":
    save_features(
        users_dat_path="data/raw/users.dat",
        movies_dat_path="data/raw/movies.dat",
        id_maps_path="data/processed/id_maps.json",
        out_dir="data/processed",
    )