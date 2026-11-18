import json, numpy as np
seeds = [1, 2, 3]
def ms(x): return f"{np.mean(x):.4f} ± {np.std(x, ddof=1):.4f}"
def tt(pattern, key):
    return [json.load(open(pattern.format(s)))["full_catalog"][key] for s in seeds]
mf = json.load(open("results/phase1c_mf_seeds.json"))
rows = {
 "MF InfoNCE+logQ": [[mf[str(s)]["full_catalog"][k] for s in seeds] for k in ("NDCG@10", "Recall@10")],
 "TT+feat+logQ": [tt("results/metrics_two_tower_with_features_logq_fv_s{}.json", k) for k in ("NDCG@10", "Recall@10")],
 "TT ID-only+logQ": [tt("results/metrics_two_tower_logq_fv_s{}.json", k) for k in ("NDCG@10", "Recall@10")],
}
for n, (a, b) in rows.items():
    print(f"{n:18s} NDCG {ms(a)} | Recall {ms(b)}")