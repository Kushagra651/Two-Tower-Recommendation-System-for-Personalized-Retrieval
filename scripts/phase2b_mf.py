import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import build_eval_candidates
from src.evaluate import evaluate_model, evaluate_full_ranking
from src.models.losses import in_batch_contrastive_loss
from src.utils.config import load_config

cfg = load_config("configs/train_config.yaml")
P = cfg.data.processed_dir
maps = json.load(open(os.path.join(P, cfg.data.id_maps_file)))
nu, ni = maps["num_users"], maps["num_items"]
train = pd.read_csv(os.path.join(P, cfg.data.train_file))
val = pd.read_csv(os.path.join(P, cfg.data.val_file))
test = pd.read_csv(os.path.join(P, cfg.data.test_file))
full = pd.concat([train, val, test], ignore_index=True)

tu = torch.tensor(train.user_idx.values, dtype=torch.long)
ti = torch.tensor(train.item_idx.values, dtype=torch.long)
pos_mat = torch.zeros(nu, ni, dtype=torch.bool)
pos_mat[tu, ti] = True

c = train["item_idx"].value_counts().reindex(range(ni), fill_value=0).values.astype("float64") + 1.0
log_q = torch.tensor(np.log(c / c.sum()), dtype=torch.float32)

test_cands = build_eval_candidates(test, full, ni, cfg.data.num_eval_negatives, negative_mode="uniform")


class MF(nn.Module):
    def __init__(self, dim, normalize):
        super().__init__()
        self.U = nn.Embedding(nu, dim)
        self.I = nn.Embedding(ni, dim)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.I.weight, std=0.01)
        self.normalize = normalize

    def _n(self, x):
        return F.normalize(x, dim=-1) if self.normalize else x

    def users(self, u): return self._n(self.U(u))
    def items(self, i): return self._n(self.I(i))
    def score(self, u, i): return (self.users(u) * self.items(i)).sum(-1)
    def full(self, u): return self.users(u) @ self._n(self.I.weight).T


def run(name, loss_type, dim, lr, wd, K=1, epochs=30, patience=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MF(dim, normalize=(loss_type == "infonce"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best, bad, best_state, t0 = -1.0, 0, None, time.time()
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(tu))
        for s in range(0, len(perm) - 511, 512):
            idx = perm[s:s + 512]
            u, i = tu[idx], ti[idx]
            if loss_type == "bpr":
                neg = torch.randint(0, ni, (len(u), K))
                clash = pos_mat[u[:, None], neg]
                neg = torch.where(clash, torch.randint(0, ni, neg.shape), neg)
                ue, pe, ne = m.U(u)[:, None, :], m.I(i)[:, None, :], m.I(neg)
                loss = -F.logsigmoid((ue * pe).sum(-1) - (ue * ne).sum(-1)).mean()
            else:
                loss = in_batch_contrastive_loss(m.users(u), m.items(i), 0.1, log_q=log_q[i])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        v = evaluate_full_ranking(m.full, val, [train], ni, k=10)["NDCG@10"]
        if v > best:
            best, bad = v, 0
            best_state = {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    s = evaluate_model(m, test_cands, k=10, device="cpu")
    f = evaluate_full_ranking(m.full, test, [train, val], ni, k=10)
    print(f"{name:28s} | ep {ep:2d} | val full {best:.4f} | sampled {s['NDCG@10']:.4f} | "
          f"full NDCG {f['NDCG@10']:.4f} | full Recall {f['Recall@10']:.4f} | {time.time()-t0:.0f}s")
    return {"val_full_ndcg": best, "sampled_uniform": s, "full_catalog": f}


grid = [
    ("bpr K4 lr1e-3 wd1e-5",   "bpr", 64, 1e-3, 1e-5, 4),
    ("bpr K8 lr1e-3 wd1e-5",   "bpr", 64, 1e-3, 1e-5, 8),
    ("bpr K8 lr3e-3 wd1e-5",   "bpr", 64, 3e-3, 1e-5, 8),
    ("bpr K8 lr1e-3 wd1e-4",   "bpr", 64, 1e-3, 1e-4, 8),
    ("bpr K8 lr1e-3 wd1e-5 d128", "bpr", 128, 1e-3, 1e-5, 8),
    ("mf infonce+logQ d64",    "infonce", 64, 1e-3, 1e-5, 1),
]


if __name__ == "__main__":
    out = {}
    for name, lt, dim, lr, wd, K in grid:
        out[name] = run(name, lt, dim, lr, wd, K)
        json.dump(out, open("results/phase2b_mf.json", "w"), indent=2)