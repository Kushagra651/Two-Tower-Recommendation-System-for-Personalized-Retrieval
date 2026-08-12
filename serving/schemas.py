"""
serving/schemas.py
--------------------
Request/response contracts for the /recommend endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Raw UserID as it appears in users.dat / ratings.dat (NOT the internal user_idx)")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")


class RecommendationItem(BaseModel):
    movie_id: int = Field(..., description="Raw MovieID as it appears in movies.dat")
    title: Optional[str] = Field(None, description="Movie title, if movie metadata was available at startup")
    genres: Optional[str] = Field(None, description="Pipe-separated genres, e.g. 'Comedy|Drama'")
    score: float = Field(..., description="Cosine similarity between user and item embeddings")


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    titles_loaded: bool = False