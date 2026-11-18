"""
serving/schemas.py
--------------------
Request/response contracts for the /recommend endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="Raw UserID. Omit for unknown user.")
    gender: Optional[str] = Field(None, description="'M' or 'F'. Required if user_id unknown.")
    age: Optional[int] = Field(None, description="Age bucket: 1 18 25 35 45 50 56.")
    occupation: Optional[int] = Field(None, description="0-20. Required if user_id unknown.")
    top_k: int = Field(10, ge=1, le=100)


class RecommendationItem(BaseModel):
    movie_id: int = Field(..., description="Raw MovieID as it appears in movies.dat")
    title: Optional[str] = Field(None, description="Movie title, if movie metadata was available at startup")
    genres: Optional[str] = Field(None, description="Pipe-separated genres, e.g. 'Comedy|Drama'")
    score: float = Field(..., description="Cosine similarity between user and item embeddings")


class RecommendResponse(BaseModel):
    user_id: Optional[int]
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    titles_loaded: bool = False