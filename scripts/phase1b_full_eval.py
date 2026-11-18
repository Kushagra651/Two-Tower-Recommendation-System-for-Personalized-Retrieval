import json, os
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import build_eval_candidates
from src.data.features import load_user_features, load_item_features
from src.evaluate import evaluate_model, evaluate_full_ranking
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

def tt(path, wf):
    m = TwoTowerModel(nu, ni, cfg.model.embedding_dim, cfg.model.hidden_dims,
                      cfg.model.dropout, with_features=wf)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    if wf:
        m.set_feature_tensors(uf, itf)
    return m

mf = MatrixFactorization(nu, ni, cfg.model.embedding_dim)
mf.load_state_dict(torch.load("checkpoints/best_model_mf.pt", map_location="cpu"))
mf.eval()

# popularity from TRAIN only
pop = torch.zeros(ni)
counts = train["item_idx"].value_counts()
pop[counts.index.values] = torch.tensor(counts.values, dtype=torch.float32)

class PopModel(nn.Module):
    def score(self, u, i):
        return pop[i]

# score_fns -> (B, ni)
all_items = torch.arange(ni)
def mf_fn(u):
    return mf.user_embedding(u) @ mf.item_embedding.weight.T

def tt_fn(m, wf):
    item_emb = m.all_item_embeddings(all_items, itf if wf else None)
    def fn(u):
        ue = m.user_embedding_for(u, uf[u] if wf else None)
        return ue @ item_emb.T
    return fn

def pop_fn(u):
    return pop.unsqueeze(0).expand(len(u), -1)

tt_id, tt_ft = tt("checkpoints/best_model.pt", False), tt("checkpoints/best_model_with_features.pt", True)
models = {
    "Popularity":           (PopModel(), pop_fn),
    "MF (BPR)":             (mf, mf_fn),
    "Two-Tower ID-only":    (tt_id, tt_fn(tt_id, False)),
    "Two-Tower + features": (tt_ft, tt_fn(tt_ft, True)),
}
tt_lq = tt("checkpoints/best_model_with_features_logq.pt", True)
tt_lq_id = tt("checkpoints/best_model_logq.pt", False)
tt_fv = tt("checkpoints/best_model_with_features_logq_fv.pt", True)
models["Two-Tower + feat + logQ (fv)"] = (tt_fv, tt_fn(tt_fv, True))
models["Two-Tower ID-only + logQ"] = (tt_lq_id, tt_fn(tt_lq_id, False))
models["Two-Tower + feat + logQ"] = (tt_lq, tt_fn(tt_lq, True))
cands = build_eval_candidates(test, full, ni, cfg.data.num_eval_negatives, negative_mode="uniform")
out = {}
for name, (m, fn) in models.items():
    s = evaluate_model(m, cands, k=10, device="cpu")
    f = evaluate_full_ranking(fn, test, [train, val], ni, k=10)
    out[name] = {"sampled_uniform": s, "full_catalog": f}
    print(f"{name:22s} | sampled NDCG {s['NDCG@10']:.4f} | full NDCG {f['NDCG@10']:.4f} | full Recall {f['Recall@10']:.4f}")

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/phase1b_full_eval.json", "w"), indent=2)