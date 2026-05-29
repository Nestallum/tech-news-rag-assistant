"""Sparse (keyword) retrieval with BM25.

Complements dense retrieval: where dense search captures meaning but can dilute
rare exact terms, BM25 matches literal words and never misses an exact token
(product names, version numbers, acronyms).

Unlike ChromaDB, BM25 has no persistent store. The index is an in-memory
structure built from the corpus texts. We rebuild it at startup by reading all
documents from the Chroma collection — fast for our corpus size (a few ms for
hundreds of chunks). For much larger corpora, the index would be persisted or
delegated to a dedicated engine (OpenSearch/Qdrant) — noted in the README.

Critical detail — tokenization must be IDENTICAL for indexing and querying.
BM25 does exact token matching: if "Apple" is indexed but "apple" is queried,
they won't match. A single `tokenize()` function is used on both sides.
"""

from __future__ import annotations

import re

from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

# Token pattern: runs of letters/digits. Splits on everything else (spaces,
# punctuation, symbols). Simple and language-agnostic enough for English news.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


# -----------------------------------------------------------------------------
# Tokenization (used for BOTH indexing and querying)
# -----------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Used identically on corpus chunks (at index build) and on queries (at
    search time). Keeping a single function guarantees the consistency BM25
    needs for exact matching.

    Example:
        "Apple's M5 chip!" -> ["apple", "s", "m5", "chip"]
    """
    return _TOKEN_PATTERN.findall(text.lower())


# -----------------------------------------------------------------------------
# Sparse retriever
# -----------------------------------------------------------------------------


class SparseRetriever:
    """BM25 keyword retriever built from the chunks stored in ChromaDB.

    The BM25 index is built once at construction time. If the underlying corpus
    changes (after a fresh ingestion), build a new SparseRetriever.
    """

    def __init__(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Build the BM25 index from parallel lists of chunk data.

        Args:
            chunk_ids: Chunk IDs, parallel to `documents` and `metadatas`.
            documents: Raw chunk texts.
            metadatas: Chroma metadata dicts (one per chunk).

        Use `from_collection()` instead of calling this directly in most cases.
        """
        if not (len(chunk_ids) == len(documents) == len(metadatas)):
            raise ValueError("chunk_ids, documents, and metadatas must have equal length")
        if not documents:
            raise ValueError("Cannot build a BM25 index from an empty corpus")

        self.chunk_ids = chunk_ids
        self.documents = documents
        self.metadatas = metadatas

        # Tokenize every chunk, then build the BM25 index over those token lists.
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built over %d chunks", len(documents))

    @classmethod
    def from_collection(cls, collection: Collection) -> SparseRetriever:
        """Build a SparseRetriever by reading all chunks from a Chroma collection.

        `collection.get()` with no filter returns the entire collection. For our
        corpus size this is cheap; for very large corpora this is where you'd
        switch to a persisted BM25 index instead.
        """
        raw = collection.get(include=["documents", "metadatas"])
        return cls(
            chunk_ids=raw["ids"],
            documents=raw["documents"],  # type: ignore
            metadatas=raw["metadatas"],  # type: ignore
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve the top_k chunks with the highest BM25 score for a query.

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to return.

        Returns:
            A list of RetrievalResult ranked best-first. `score` is the raw
            BM25 score (unbounded, higher = better). Do NOT compare these
            scores against dense scores — different scale entirely.
        """
        query_tokens = tokenize(query)

        # BM25 scores the query against EVERY document in the corpus.
        scores = self.bm25.get_scores(query_tokens)

        # Rank by score, keep the top_k indices.
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            meta = self.metadatas[idx]
            results.append(
                RetrievalResult(
                    chunk_id=self.chunk_ids[idx],
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    article_url=str(meta["article_url"]),
                    article_title=str(meta["article_title"]),
                    source=str(meta["source"]),
                    feed_name=str(meta["feed_name"]),
                    chunk_index=int(meta["chunk_index"]),
                    published_at=int(meta["published_at"]),
                )
            )

        logger.info("Sparse retrieval: %d results for query %r", len(results), query[:60])
        return results
