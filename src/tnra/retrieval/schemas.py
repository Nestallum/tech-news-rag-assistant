"""Pydantic schemas for the retrieval stage.

A retrieval produces a ranked list of `RetrievalResult` objects. Each one
wraps a chunk's text + metadata together with a relevance score, so that
downstream stages (reranking, generation, evaluation) have a single, typed
contract to work with — regardless of which retriever produced it (dense,
sparse, hybrid, or reranked).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its relevance score and source metadata.

    `score` semantics depend on the retriever that produced it:
        - dense  : similarity in [0, 1]  (higher = better; 1 - cosine_distance)
        - sparse : raw BM25 score        (higher = better; unbounded)
        - hybrid : RRF score             (higher = better; small positive float)
        - reranked: cross-encoder score  (higher = better)
    Because the scale changes, never compare scores across retriever types —
    only compare ranks, or scores within the same retriever.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float

    # --- Source metadata (carried through for citations + filtering) ---
    article_url: str = Field(min_length=1)
    article_title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    feed_name: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    published_at: str | None = None  # ISO 8601 string, or None if feed omitted it
