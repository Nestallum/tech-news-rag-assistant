"""Dense (semantic) retrieval over the ChromaDB index.

Embeds the query with the SAME model used at ingestion time (BGE-large-en-v1.5)
and asks ChromaDB for the nearest chunks by cosine distance.

Why "same model" is non-negotiable: chunks were indexed as vectors produced by
BGE-large. Querying with a different model produces vectors in a different
geometric space — the comparison would be meaningless. The model is pinned in
config and loaded identically on both sides.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from tnra.ingestion.embedding import Embedder
from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Dense retriever
# -----------------------------------------------------------------------------


class DenseRetriever:
    """Semantic retriever backed by a Qdrant collection.

    Holds a reference to a shared Embedder and a Qdrant collection. Both are
    constructed once (Embedder loading is expensive) and reused across queries.
    """

    def __init__(self, client: QdrantClient, collection_name: str, embedder: Embedder) -> None:
        self.client = client
        self.collection_name = collection_name
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
        query_vector = self.embedder.embed_query(query)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points

        results: list[RetrievalResult] = []
        for hit in hits:
            assert hit.payload is not None
            p = hit.payload
            results.append(
                RetrievalResult(
                    chunk_id=str(p["chunk_id"]),
                    text=str(p["text"]),
                    score=hit.score,
                    article_url=str(p["article_url"]),
                    article_title=str(p["article_title"]),
                    source=str(p["source"]),
                    feed_name=str(p["feed_name"]),
                    chunk_index=int(p["chunk_index"]),
                    published_at=int(p["published_at"]),
                )
            )

        logger.info("Dense retrieval: %d results for query %r", len(results), query[:60])
        return results
