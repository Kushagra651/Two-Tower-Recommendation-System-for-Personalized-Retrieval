"""
tests/test_api.py
--------------------
Mocks load_bundle() so these tests don't depend on a real trained checkpoint
being present — they verify the API contract (status codes, schema, error
handling) in isolation.

Run with: pytest tests/test_api.py -v
"""

import faiss
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import serving.app as app_module
from serving.model_loader import ModelBundle
from src.models.two_tower import TwoTowerModel


@pytest.fixture
def fake_bundle():
    num_users, num_items, dim = 5, 8, 16
    model = TwoTowerModel(num_users+1, num_items, embedding_dim=dim, hidden_dims=[32, 16],with_features=True)
    model.eval()

    user_features = torch.zeros(num_users, 3)    # gender, age_idx, occ_idx
    item_features = torch.zeros(num_items, 18)   # genre multi-hot

    item_emb = model.all_item_embeddings(torch.arange(num_items), item_features).numpy().astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(item_emb)

    user2idx = {101: 0, 102: 1, 103: 2}
    idx2item = {i: 1000 + i for i in range(num_items)}
    seen_items = {101: {1001, 1002}, 102: set(), 103: set()}

    return ModelBundle(model=model, index=index, user2idx=user2idx, idx2item=idx2item,
                       user_features=user_features, item_features=item_features,
                       seen_items=seen_items, num_users=num_users, top_k_default=10)


@pytest.fixture
def client(fake_bundle, monkeypatch):
    monkeypatch.setattr(app_module, "load_bundle", lambda: fake_bundle)
    with TestClient(app_module.app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True


class TestRecommendEndpoint:
    def test_known_user_returns_recommendations(self, client):
        resp = client.post("/recommend", json={"user_id": 101, "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 101
        assert len(body["recommendations"]) == 5
        for rec in body["recommendations"]:
            assert "movie_id" in rec and "score" in rec

    def test_seen_items_are_filtered(self, client):
        r = client.post("/recommend", json={"user_id": 101, "top_k": 6})
        ids = {x["movie_id"] for x in r.json()["recommendations"]}
        assert 1001 not in ids and 1002 not in ids

    def test_unknown_user_without_demographics_is_rejected(self, client):
        resp = client.post("/recommend", json={"user_id": 999999, "top_k": 5})
        assert resp.status_code == 422

    def test_unknown_user_with_demographics_uses_unk(self, client):
        resp = client.post("/recommend", json={"gender": "M", "age": 25, "occupation": 7, "top_k": 5})
        assert resp.status_code == 200

    def test_top_k_is_respected(self, client):
        resp = client.post("/recommend", json={"user_id": 102, "top_k": 3})
        assert len(resp.json()["recommendations"]) == 3

    def test_top_k_out_of_range_is_rejected(self, client):
        resp = client.post("/recommend", json={"user_id": 101, "top_k": 0})
        assert resp.status_code == 422  # pydantic validation error (top_k >= 1)

    def test_movie_ids_are_correctly_mapped(self, client):
        resp = client.post("/recommend", json={"user_id": 101, "top_k": 8})
        movie_ids = {rec["movie_id"] for rec in resp.json()["recommendations"]}
        assert movie_ids.issubset(set(range(1000, 1008)))