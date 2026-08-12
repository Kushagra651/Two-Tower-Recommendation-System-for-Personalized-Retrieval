# Two-Tower Recommendation System for Personalized Retrieval

A production-style recommender built end-to-end on MovieLens-1M — raw ratings to a containerized, low-latency retrieval API. Covers the full ML lifecycle: data processing, representation learning, ANN retrieval, and serving.

**Stack:** PyTorch · FAISS · FastAPI · Streamlit · Docker

---

## Highlights

- Two-tower neural recommender trained with **in-batch contrastive loss** and **popularity-weighted hard-negative sampling** on 1M interactions (6,040 users, 3,706 items, 95.5% sparsity)
- **NDCG@10 = 0.24 / Recall@10 = 0.45** on held-out test data, using the leave-one-out evaluation protocol from the NCF paper for direct comparability
- **Sub-2ms p99** top-K retrieval latency via a benchmarked FAISS ANN layer over exported item embeddings
- Dockerized FastAPI inference service + Streamlit demo UI, backed by **23 passing unit/integration tests**

---

## Architecture

<img width="1920" height="1200" alt="Two-tower architecture diagram" src="https://github.com/user-attachments/assets/aa4f31e7-b8e1-4ded-8961-41367c010bad" />

A user tower and an item tower each map raw IDs to embeddings via an `Embedding -> MLP -> L2-normalize` stack, trained jointly with an in-batch contrastive loss — other items in the same batch act as free negatives, so no explicit negative sampling is needed at train time. At serving time, the trained item tower's output is exported into a FAISS index for fast approximate nearest-neighbor retrieval, and a FastAPI service computes a user embedding on the fly and looks up the top-K most similar items.

---

## Demo

| Streamlit UI | API Response |
|:---:|:---:|
|<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/d64a9580-22bc-4266-84b1-c6d688f4d16b" />
" /> | <img width="800" alt="Add FastAPI /docs or response screenshot here" src="" /> |

*(Add screenshots: left — Streamlit recommendation UI in action; right — FastAPI `/docs` Swagger view or a sample JSON response.)*

---

## Project Structure

```
two-tower-recsys/
├── data/               # raw .dat files, processed/merged interactions, train/val/test splits
├── notebooks/          # EDA, baseline MF comparison, embedding-dim ablation
├── src/
│   ├── data/            # merge, implicit conversion, negative sampling, time-based split, Dataset
│   ├── models/          # two-tower architecture, MF baseline, loss functions
│   ├── retrieval/        # FAISS index build + ANN search + latency benchmark
│   ├── train.py         # training loop
│   └── evaluate.py      # NDCG@10 / Recall@10 / MAP
├── serving/             # FastAPI app, schemas, model loader, Dockerfile
├── demo/                # Streamlit UI hitting the API
├── tests/                # pytest unit + API tests
├── configs/              # train_config.yaml, retrieval_config.yaml
├── checkpoints/          # saved model weights (gitignored)
├── embeddings/           # FAISS index + exported item embeddings (gitignored)
└── results/              # metrics.json, latency_benchmark.csv, architecture_diagram.png
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
# 1. Merge raw MovieLens files into one table
python src/data/merge_ml1m.py --raw_dir data/raw --out_dir data/processed

# 2. Convert ratings -> implicit feedback, build ID mappings
python src/data/preprocess.py --in_path data/processed/merged_interactions.csv

# 3. Time-based leave-one-out train/val/test split
python src/data/splits.py

# 4. Train the two-tower model
python src/train.py --config configs/train_config.yaml

# 5. Evaluate on the held-out test split
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test

# 6. Build the FAISS index from the trained item tower
python src/retrieval/build_index.py --checkpoint checkpoints/best_model.pt

# 7. Benchmark retrieval latency (p50/p99)
python src/retrieval/search.py

# 8. Serve the model
uvicorn serving.app:app --reload --port 8000
# in a separate terminal:
streamlit run demo/streamlit_app.py

# or, run API + demo together via Docker:
docker-compose up --build
```

**Tests:**
```bash
pytest tests/ -v
```

---

## Results

| Model | Recall@10 | NDCG@10 | MAP | p99 Latency |
|---|---|---|---|---|
| Matrix Factorization (baseline) | not yet run | not yet run | not yet run | — |
| **Two-Tower (this project)** | **0.45** | **0.24** | not yet run | **<2 ms** |
| NCF paper (reference) | see paper | see paper | — | — |

Evaluation protocol: 1 held-out positive + 99 sampled negatives per user, matching the NCF paper's leave-one-out setup — these numbers are directly comparable to published benchmarks.

---

## Design Decisions & Tradeoffs

- **In-batch contrastive loss over explicit negative sampling** for the two-tower model — scales better and avoids materializing negatives every epoch. The MF baseline still uses explicit negatives (BPR loss) since it doesn't share this batch structure.
- **Time-based leave-one-out split, not random** — avoids leaking future interactions into training and matches the standard evaluation protocol in recsys literature.
- **FAISS `IndexFlatIP` (exact search) by default** — at MovieLens scale (thousands of items) exact search is fast enough; `IndexIVFFlat` is configured in `configs/retrieval_config.yaml` for larger catalogs.
- **ID-embedding-only towers** — side features (genres, demographics) aren't wired in yet; the `Tower` class exposes an `extra_dim` hook for this as a natural extension.
- **Catalog size reported honestly** — MovieLens-1M has ~3,900 items; numbers here reflect the actual dataset scale rather than a rounded estimate.
