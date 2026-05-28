"""Retrieval pipeline orchestration.

Wires the five retrieval components into a single entry point:

    query
      ├─► dense retrieval  (ChromaDB, semantic) ──┐
      ├─► sparse retrieval (BM25, keyword)     ──┤
      │                                          ▼
      │                                   RRF fusion
      │                                          ▼
      │                              article-level dedup
      │                                          ▼
      └────────────────────────────────► cross-encoder rerank
                                                  ▼
                                          top_k passages

The `Retriever` class loads every component once and reuses them across
queries — model loading (embedder + reranker) is expensive, so the same
instance is meant to live for the whole app lifetime.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection
from omegaconf import DictConfig
from pydantic import BaseModel

from tnra.ingestion.embedding import Embedder, EmbeddingConfig
from tnra.retrieval.dedup import deduplicate_by_article
from tnra.retrieval.dense import DenseRetriever
from tnra.retrieval.fusion import reciprocal_rank_fusion
from tnra.retrieval.reranker import Reranker, RerankerConfig
from tnra.retrieval.schemas import RetrievalResult
from tnra.retrieval.sparse import SparseRetriever
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------


class HybridConfig(BaseModel):
    """Validated config for the hybrid retrieval stage (the `retrieval.hybrid` block)."""

    dense_top_k: int
    bm25_top_k: int
    rrf_k: int
    fused_top_k: int


class RetrievalConfig(BaseModel):
    """Top-level validated retrieval config (the `retrieval` block of retrieval.yaml)."""

    hybrid: HybridConfig
    reranker: RerankerConfig


# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------


class Retriever:
    """End-to-end retriever: query in, ranked passages out.

    Holds the dense retriever, sparse retriever, and reranker. Build it once
    and reuse it — constructing it loads the embedding model and the reranker
    model (several GB combined), which is slow.
    """

    def __init__(
        self,
        collection: Collection,
        embedder: Embedder,
        retrieval_cfg: RetrievalConfig,
    ) -> None:
        self.cfg = retrieval_cfg

        # Dense: semantic search over ChromaDB, using the shared embedder.
        self.dense = DenseRetriever(collection=collection, embedder=embedder)

        # Sparse: BM25 index built from the texts stored in the same collection.
        self.sparse = SparseRetriever.from_collection(collection)

        # Reranker: built only if enabled, to avoid loading a 2.3 GB model for nothing.
        self.reranker: Reranker | None = None
        if retrieval_cfg.reranker.enabled:
            self.reranker = Reranker(retrieval_cfg.reranker)
        else:
            logger.info("Reranker disabled in config — skipping model load")

        logger.info("Retriever ready (reranker=%s)", "on" if self.reranker else "off")

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Run the full retrieval pipeline for a query.

        Steps:
            1. Dense + sparse retrieval (each returns its own top_k).
            2. RRF fusion of the two ranked lists.
            3. Article-level deduplication (one chunk per article).
            4. Cross-encoder reranking → final top_k (if reranker enabled;
               otherwise the deduped list is simply truncated).

        Args:
            query: The user's natural-language question.

        Returns:
            A ranked list of RetrievalResult, best first, length = reranker.top_k.
        """
        hybrid = self.cfg.hybrid
        final_k = self.cfg.reranker.top_k

        # --- 1. Dense + sparse ---
        dense_results = self.dense.retrieve(query, top_k=hybrid.dense_top_k)
        sparse_results = self.sparse.retrieve(query, top_k=hybrid.bm25_top_k)

        # --- 2. RRF fusion ---
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=hybrid.rrf_k)

        # --- 3. Article-level dedup ---
        deduped = deduplicate_by_article(fused)
        deduped = deduped[: hybrid.fused_top_k]

        # --- 4. Rerank (or truncate if reranker disabled) ---
        if self.reranker is not None:
            final = self.reranker.rerank(query, deduped, top_k=final_k)
        else:
            final = deduped[:final_k]

        logger.info("Retrieval complete: query=%r → %d passages", query[:60], len(final))
        return final


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------


def build_retriever(
    collection: Collection,
    retrieval_cfg: DictConfig,
    embeddings_cfg: DictConfig,
) -> Retriever:
    """Construct a Retriever from raw OmegaConf config sections.

    Args:
        collection: An open ChromaDB collection (the indexed corpus).
        retrieval_cfg: The `retrieval` block of configs/retrieval.yaml.
        embeddings_cfg: The `embeddings` block of configs/ingestion.yaml.
            Required because the query MUST be embedded with the exact same
            model used at ingestion time.

    Returns:
        A ready-to-use Retriever.
    """
    # Validate both config sections through their Pydantic schemas.
    validated_retrieval = RetrievalConfig(**retrieval_cfg)  # type: ignore
    validated_embeddings = EmbeddingConfig(**embeddings_cfg)  # type: ignore

    # Build the shared embedder (same model as ingestion — non-negotiable).
    embedder = Embedder(validated_embeddings)

    return Retriever(
        collection=collection,
        embedder=embedder,
        retrieval_cfg=validated_retrieval,
    )
