import json, os
import numpy as np
import pandas as pd
import torch

from src.data.features import load_user_features, load_item_features
from src.evaluate import evaluate_full_ranking
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
uf = load_user_features("data/raw/users.dat", maps_path)
itf = load_item_features("data/raw/movies.dat", maps_path)
cold = json.load(open("results/cold_users.json"))
cold_set = set(cold)

train_w = train[~train.user_idx.isin(cold_set)]            # what the model saw
test_cold = test[test.user_idx.isin(cold_set)]
test_warm = test[~test.user_idx.isin(cold_set)]
print(f"cold users: {len(test_cold)} | warm users: {len(test_warm)}")

def load(path, n_users):
    m = TwoTowerModel(n_users, ni, cfg.model.embedding_dim, cfg.model.hidden_dims,
                      cfg.model.dropout, with_features=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

m_cold = load("checkpoints/best_model_with_features_logq_fv_cold.pt", nu + 1)
m_old = load("checkpoints/best_model_with_features_logq_fv.pt", nu)
ie_cold = m_cold.all_item_embeddings(torch.arange(ni), itf)
ie_old = m_old.all_item_embeddings(torch.arange(ni), itf)

# --- cold scorers ---
def tt_cold_fn(u):                                          # UNK id + real demographics
    unk = torch.full_like(u, nu)
    return m_cold.user_embedding_for(unk, uf[u]) @ ie_cold.T

cnt = np.bincount(train_w.item_idx.values, minlength=ni).astype(np.float32)
pop = torch.tensor(cnt / cnt.sum())
def global_pop_fn(u):
    return pop.unsqueeze(0).expand(len(u), -1)

g, a, o = (uf[:, i].long().numpy() for i in (0, 1, 2))
warm_users = np.array(sorted(set(range(nu)) - cold_set))
tw = train_w.copy()
seg_cache = {}
def seg_vec(uid, min_users=20):
    for key, same in (((g[uid], a[uid], o[uid]), (g == g[uid]) & (a == a[uid]) & (o == o[uid])),
                      ((g[uid], a[uid]), (g == g[uid]) & (a == a[uid]))):
        members = warm_users[same[warm_users]]
        if len(members) >= min_users:
            if key not in seg_cache:
                sub = tw[tw.user_idx.isin(members)]
                c = np.bincount(sub.item_idx.values, minlength=ni).astype(np.float32)
                seg_cache[key] = torch.tensor(c / max(c.sum(), 1))
            return seg_cache[key] + 1e-9 * pop
    return pop
def seg_pop_fn(u):
    return torch.stack([seg_vec(x) for x in u.tolist()])

masks = [train, val]          # hidden history of cold users, known to the evaluator only
out = {}
for name, fn in (("Global popularity", global_pop_fn),
                 ("Segment popularity (g x age x occ, backoff)", seg_pop_fn),
                 ("Two-Tower UNK + demographics", tt_cold_fn)):
    r = evaluate_full_ranking(fn, test_cold, masks, ni, k=10)
    out[name] = r
    print(f"COLD | {name:46s} | NDCG {r['NDCG@10']:.4f} | Recall {r['Recall@10']:.4f}")

# --- warm-user delta (same warm users for both models) ---
def warm_fn(m, ie):
    return lambda u: m.user_embedding_for(u, uf[u]) @ ie.T
for name, m, ie in (("Two-Tower fv (trained on all users)", m_old, ie_old),
                    ("Two-Tower cold-trained (ID dropout 0.15)", m_cold, ie_cold)):
    r = evaluate_full_ranking(warm_fn(m, ie), test_warm, masks, ni, k=10)
    out["WARM " + name] = r
    print(f"WARM | {name:46s} | NDCG {r['NDCG@10']:.4f} | Recall {r['Recall@10']:.4f}")

rng = np.random.default_rng(0)
tt_n = np.array(out["Two-Tower UNK + demographics"]["per_user_ndcg"])
for other in ("Global popularity", "Segment popularity (g x age x occ, backoff)"):
    d = tt_n - np.array(out[other]["per_user_ndcg"])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Paired diff TT - {other[:22]}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
json.dump(out, open("results/phase3_cold_eval.json", "w"), indent=2)