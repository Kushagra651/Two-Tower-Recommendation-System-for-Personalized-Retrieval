"""
serving/app.py
----------------
FastAPI service wrapping the trained two-tower model + FAISS index.

Run locally:
    uvicorn serving.app:app --reload --port 8000

Endpoints:
    GET  /health            -> liveness + whether the model loaded
    POST /recommend         -> {"user_id": 1, "top_k": 10} -> top-K movie recommendations
"""
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from serving.model_loader import ModelBundle, load_bundle
from serving.schemas import HealthResponse, RecommendationItem, RecommendRequest, RecommendResponse

bundle: ModelBundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    bundle = load_bundle()
    yield
    bundle = None


app = FastAPI(title="Two-Tower Recommender API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=bundle is not None)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.user_id not in bundle.user2idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {req.user_id}")

    user_idx = bundle.user2idx[req.user_id]

    with torch.no_grad():
        user_emb = bundle.model.user_embedding_for(torch.tensor([user_idx])).numpy().astype("float32")

    scores, indices = bundle.index.search(user_emb, req.top_k)

    recommendations = [
        RecommendationItem(movie_id=bundle.idx2item[int(i)], score=float(s))
        for i, s in zip(indices[0], scores[0])
    ]

    return RecommendResponse(user_id=req.user_id, recommendations=recommendations)