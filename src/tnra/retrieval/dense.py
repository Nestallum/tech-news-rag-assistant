"""Dense (semantic) retrieval over the ChromaDB index.

Embeds the query with the SAME model used at ingestion time (BGE-large-en-v1.5)
and asks ChromaDB for the nearest chunks by cosine distance.

Why "same model" is non-negotiable: chunks were indexed as vectors produced by
BGE-large. Querying with a different model produces vectors in a different
geometric space — the comparison would be meaningless. The model is pinned in
config and loaded identically on both sides.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from tnra.ingestion.embedding import Embedder
from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------


class DenseRetrieverConfig(BaseModel):
    """Validated config for the dense retriever.

    Built from the `retrieval.hybrid` section of `configs/retrieval.yaml`
    (the `dense_top_k` field).
    """

    top_k: int = Field(gt=0, le=200)


# -----------------------------------------------------------------------------
# Dense retriever
# -----------------------------------------------------------------------------


class DenseRetriever:
    """Semantic retriever backed by a ChromaDB collection.

    Holds a reference to a shared Embedder and a Chroma collection. Both are
    constructed once (Embedder loading is expensive) and reused across queries.
    """

    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve the top_k most semantically similar chunks for a query.

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to return.

        Returns:
            A list of RetrievalResult, ranked best-first. `score` is
            `1 - cosine_distance`, so it lives in roughly [-1, 1] with higher
            meaning more similar (typically 0.5-0.8 for good matches).
        """
        # 1. Embed the query (same BGE model as ingestion).
        query_vector = self.embedder.embed_query(query)

        # 2. Ask Chroma for the nearest chunks.
        #    We request documents + metadatas + distances so we can build
        #    self-contained RetrievalResult objects without a second lookup.
        raw = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 3. Chroma returns parallel lists wrapped in an outer list (one entry
        #    per query). We sent a single query, so we take index [0].
        ids = raw["ids"][0]
        documents = raw["documents"][0]  # type: ignore
        metadatas = raw["metadatas"][0]  # type: ignore
        distances = raw["distances"][0]  # type: ignore

        results: list[RetrievalResult] = []
        for chunk_id, text, meta, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=1.0 - distance,  # distance → similarity (higher = better)
                    article_url=str(meta["article_url"]),
                    article_title=str(meta["article_title"]),
                    source=str(meta["source"]),
                    feed_name=str(meta["feed_name"]),
                    chunk_index=int(meta["chunk_index"]),  # type: ignore
                    published_at=meta.get("published_at"),  # may be absent # type: ignore
                )
            )

        logger.info("Dense retrieval: %d results for query %r", len(results), query[:60])
        return results
