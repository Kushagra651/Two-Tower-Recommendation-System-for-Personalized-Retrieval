"""Re-evaluate existing checkpoints on test under both negative modes."""
import json, os
import pandas as pd
import torch

from src.data.dataset import build_eval_candidates
from src.data.features import load_user_features, load_item_features
from src.evaluate import evaluate_model
from src.models.baseline_mf import MatrixFactorization
from src.models.two_tower import TwoTowerModel
from src.utils.config import load_config

cfg = load_config("configs/train_config.yaml")
P = cfg.data.processed_dir
maps_path = os.path.join(P, cfg.data.id_maps_file)
maps = json.load(open(maps_path))
nu, ni = maps["num_users"], maps["num_items"]

train = pd.read_csv(os.path.join(P, cfg.data.train_file))
val = pd.read_csv(os.path.join(P, cfg.data.val_file))
test = pd.read_csv(os.path.join(P, cfg.data.test_file))
full = pd.concat([train, val, test], ignore_index=True)

uf = load_user_features("data/raw/users.dat", maps_path)
itf = load_item_features("data/raw/movies.dat", maps_path)

def tt(path, with_features):
    m = TwoTowerModel(nu, ni, cfg.model.embedding_dim, cfg.model.hidden_dims,
                      cfg.model.dropout, with_features=with_features)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    if with_features:
        m.set_feature_tensors(uf, itf)
    return m

mf = MatrixFactorization(nu, ni, cfg.model.embedding_dim)
mf.load_state_dict(torch.load("checkpoints/best_model_mf.pt", map_location="cpu"))

models = {
    "MF (BPR)": mf,
    "Two-Tower ID-only": tt("checkpoints/best_model.pt", False),
    "Two-Tower + features": tt("checkpoints/best_model_with_features.pt", True),
}

out = {}
for mode in ("uniform", "popularity_mix"):
    cands = build_eval_candidates(test, full, ni, cfg.data.num_eval_negatives,
                                  negative_mode=mode)
    for name, m in models.items():
        r = evaluate_model(m, cands, k=cfg.train.top_k, device="cpu")
        out.setdefault(name, {})[mode] = {k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in r.items()}
        print(f"{mode:15s} | {name:22s} | {out[name][mode]}")

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/phase1a_negative_modes.json", "w"), indent=2)