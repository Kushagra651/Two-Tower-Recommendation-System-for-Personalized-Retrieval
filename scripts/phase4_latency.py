# scripts/phase4_latency.py  — replace with this
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time, requests, numpy as np, json, torch
from src.models.two_tower import TwoTowerModel
from src.data.features import load_user_features
from src.utils.config import load_config
import faiss

cfg = load_config("configs/train_config.yaml")
maps = json.load(open("data/processed/id_maps.json"))
nu, ni = maps["num_users"], maps["num_items"]
uf = load_user_features("data/raw/users.dat", "data/processed/id_maps.json")

model = TwoTowerModel(nu+1, ni, cfg.model.embedding_dim, cfg.model.hidden_dims,
                      cfg.model.dropout, with_features=True)
model.load_state_dict(torch.load("checkpoints/best_model_with_features_logq_fv_cold.pt", map_location="cpu"))
model.eval()
index = faiss.read_index("embeddings/item_index.faiss")

retrieval, e2e = [], []
raw_users = list({int(k) for k in maps["user2idx"]})[:200]
for uid in raw_users:
    u = torch.tensor([maps["user2idx"][str(uid)]])
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = model.user_embedding_for(u, uf[u]).numpy().astype("float32")
    scores, idx = index.search(emb, 10)
    retrieval.append((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    requests.post("http://127.0.0.1:8000/recommend", json={"user_id": uid, "top_k": 10})
    e2e.append((time.perf_counter() - t1) * 1000)

r, e = np.array(retrieval), np.array(e2e)
print(f"Retrieval-only  p50={np.percentile(r,50):.1f}ms  p99={np.percentile(r,99):.1f}ms")
print(f"End-to-end HTTP p50={np.percentile(e,50):.1f}ms  p99={np.percentile(e,99):.1f}ms")