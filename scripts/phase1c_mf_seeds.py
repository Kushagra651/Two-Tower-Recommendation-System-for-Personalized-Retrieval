import json
from scripts.phase2b_mf import run

res = {}
for s in (1, 2, 3):
    res[s] = run(f"mf infonce+logQ s{s}", "infonce", 64, 1e-3, 1e-5, 1, seed=s)
    json.dump(res, open("results/phase1c_mf_seeds.json", "w"))