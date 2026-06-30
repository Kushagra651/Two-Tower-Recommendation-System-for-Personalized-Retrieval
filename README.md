# Two-Tower Recommendation System for Personalized Retrieval

A deep-learning recommender built on MovieLens, going from raw ratings to a live, containerized recommendation API — covering the full ML lifecycle: data processing, model training, ANN retrieval, and serving.

## Architecture

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/aa4f31e7-b8e1-4ded-8961-41367c010bad" />

A user tower and an item tower each map raw IDs to embeddings via an `Embedding -> MLP -> L2-normalize` stack. Both are trained jointly with an **in-batch contrastive loss** (other items in the same batch act as free negatives — no explicit negative sampling needed at train time). At serving time, the trained item tower's output is exported into a **FAISS index** for fast approximate nearest-neighbor retrieval, and a FastAPI service computes a user embedding on the fly and looks up the top-K most similar items.

## Project Structure

```
two-tower-recsys/
├── data/               # raw .dat files, processed/merged interactions, train/val/test splits
├── notebooks/          # EDA, baseline MF comparison, embedding-dim ablation
├── src/
│   ├── data/           # merge, implicit conversion, negative sampling, time-based split, Dataset
│   ├── models/         # two-tower architecture, MF baseline, loss functions
│   ├── retrieval/       # FAISS index build + ANN search + latency benchmark
│   ├── train.py        # training loop
│   └── evaluate.py     # NDCG@10 / Recall@10 / MAP
├── serving/            # FastAPI app, schemas, model loader, Dockerfile
├── demo/               # Streamlit UI hitting the API
├── tests/              # pytest unit + API tests
├── configs/            # train_config.yaml, retrieval_config.yaml
├── checkpoints/        # saved model weights (gitignored)
├── embeddings/         # FAISS index + exported item embeddings (gitignored)
└── results/            # metrics.json, latency_benchmark.csv, architecture_diagram.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage — full pipeline, in order

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

## Tests

```bash
pytest tests/ -v
```

## Metrics

*(Fill in after running step 5 above on the real MovieLens data — these are placeholders.)*

| Model              | Recall@10 | NDCG@10 | MAP   | p99 Latency |
|--------------------|-----------|---------|-------|-------------|
| Matrix Factorization (baseline) | 0.XX | 0.XX | 0.XX | — |
| Two-Tower (this project)        | 0.XX | 0.XX | 0.XX | X ms |
| NCF paper (reference)           | 0.XX | 0.XX | —    | — |

Evaluation protocol: 1 held-out positive + 99 sampled negatives per user (matches the NCF paper, so these numbers are directly comparable to published benchmarks).

## Tradeoffs & Design Notes

- **In-batch contrastive loss over explicit negative sampling for the two-tower model** — scales better, no need to materialize negatives every epoch. The MF baseline still uses explicit negatives (BPR loss) since it doesn't share this batch structure.
- **Time-based leave-one-out split, not random** — avoids leaking future interactions into training, and matches the standard evaluation protocol in recsys literature.
- **FAISS `IndexFlatIP` (exact search) by default** — at MovieLens scale (tens of thousands of items) exact search is fast enough; `IndexIVFFlat` is available in `configs/retrieval_config.yaml` for when the catalog grows much larger.
- **ID-embedding-only towers** — side features (genres, demographics) aren't wired into the model yet; the `Tower` class has an `extra_dim` hook for this as a natural extension.
- **Catalog size honestly reported** — MovieLens-1M has ~3,900 items and ML-25M has ~62,000; resume/README claims should cite the real number, not a round "100K+".
