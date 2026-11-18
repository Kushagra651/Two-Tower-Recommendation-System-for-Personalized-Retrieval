import json, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.data.dataset import build_eval_candidates
from src.evaluate import evaluate_model, evaluate_full_ranking
from src.utils.config import load_config

cfg = load_config("configs/train_config.yaml")
P = cfg.data.processed_dir
maps = json.load(open(os.path.join(P, cfg.data.id_maps_file)))
nu, ni = maps["num_users"], maps["num_items"]
train = pd.read_csv(os.path.join(P, cfg.data.train_file))
val = pd.read_csv(os.path.join(P, cfg.data.val_file))
test = pd.read_csv(os.path.join(P, cfg.data.test_file))
full = pd.concat([train, val, test], ignore_index=True)

R = sp.csr_matrix((np.ones(len(train), dtype=np.float32),
                   (train.user_idx.values, train.item_idx.values)), shape=(nu, ni))

def build_sim(R, shrink=0.0):
    co = (R.T @ R).toarray().astype(np.float32)
    norms = np.sqrt(np.diag(co)) + 1e-8
    sim = co / (np.outer(norms, norms) + shrink)
    np.fill_diagonal(sim, 0.0)
    return torch.tensor(sim)

class ItemKNN(nn.Module):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
    def rows(self, u):
        return torch.tensor(R[u.tolist()].toarray()) @ self.sim   # (B, ni)
    def score(self, u, i):                       # u: all the same user
        return self.rows(u[:1])[0][i]

results = {}
cands = build_eval_candidates(test, full, ni, cfg.data.num_eval_negatives, negative_mode="uniform")
for shrink in (0.0, 10.0):
    m = ItemKNN(build_sim(R, shrink))
    s = evaluate_model(m, cands, k=10, device="cpu")
    f = evaluate_full_ranking(m.rows, test, [train, val], ni, k=10)
    results[f"ItemKNN_shrink{shrink}"] = {"sampled_uniform": s, "full_catalog": f}
    print(f"ItemKNN shrink={shrink:4.1f} | sampled NDCG {s['NDCG@10']:.4f} | "
          f"full NDCG {f['NDCG@10']:.4f} | full Recall {f['Recall@10']:.4f}")

json.dump(results, open("results/phase2a_itemknn.json", "w"), indent=2)