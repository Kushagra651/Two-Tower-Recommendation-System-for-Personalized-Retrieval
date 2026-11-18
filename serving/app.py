"""serving/app.py"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serving.model_loader import ModelBundle, load_bundle
from serving.schemas import HealthResponse, RecommendationItem, RecommendRequest, RecommendResponse
from src.data.features import AGE_BUCKET_TO_IDX

bundle: ModelBundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    bundle = load_bundle()
    yield
    bundle = None


app = FastAPI(title="Two-Tower Recommender API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=bundle is not None,
                          titles_loaded=bool(bundle and bundle.movie_titles))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if bundle is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()

    # --- resolve user ---
    if req.user_id is not None and req.user_id in bundle.user2idx:
        # known user: use their trained ID + their demographics
        user_idx = bundle.user2idx[req.user_id]
        u_tensor = torch.tensor([user_idx])
        user_feat = bundle.user_features[u_tensor]
        seen = bundle.seen_items.get(req.user_id, set())
    else:
        # unknown user: UNK id + supplied demographics
        if req.gender is None or req.age is None or req.occupation is None:
            raise HTTPException(422, "Unknown user: provide gender, age, occupation.")
        u_tensor = torch.tensor([bundle.num_users])  # UNK row
        gender_f = 0.0 if req.gender.upper() == "M" else 1.0
        age_idx = float(AGE_BUCKET_TO_IDX.get(req.age, 0))
        user_feat = torch.tensor([[gender_f, age_idx, float(req.occupation)]])
        seen = set()

    with torch.no_grad():
        user_emb = bundle.model.user_embedding_for(u_tensor, user_feat).numpy().astype("float32")

    retrieval_ms = (time.perf_counter() - t0) * 1000

    # over-fetch to allow seen-item filtering
    fetch_k = req.top_k + len(seen) + 20
    scores, indices = bundle.index.search(user_emb, fetch_k)

    recs = []
    for i, s in zip(indices[0], scores[0]):
        raw_mid = bundle.idx2item.get(int(i))
        if raw_mid is None or raw_mid in seen:
            continue
        recs.append(RecommendationItem(
            movie_id=raw_mid,
            title=bundle.movie_titles.get(raw_mid),
            genres=bundle.movie_genres.get(raw_mid),
            score=float(s),
        ))
        if len(recs) == req.top_k:
            break

    total_ms = (time.perf_counter() - t0) * 1000
    # log both; headers visible in curl -v
    return RecommendResponse(user_id=req.user_id, recommendations=recs)