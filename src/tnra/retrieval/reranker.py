"""Cross-encoder reranking of retrieval candidates.

Hybrid retrieval (dense + BM25 + RRF) optimizes recall: it casts a wide net and
rarely misses a relevant chunk, but lets through some noise. The reranker
optimizes precision: it re-scores the shortlist and surfaces the truly best
passages.

Bi-encoder vs cross-encoder — the core distinction:
  - Bi-encoder (our embeddings): query and chunk are encoded SEPARATELY into
    fixed vectors, then compared. Fast, vectors precomputable — but each side
    is summarized before knowing what it will be matched against.
  - Cross-encoder (this reranker): query and chunk are fed TOGETHER through the
    model in one pass, producing a direct relevance score. The model attends
    across both texts — far more accurate, but one model pass PER PAIR.

Because it costs one pass per pair, the cross-encoder is only viable on a small
shortlist. The pipeline is therefore two-stage: fast hybrid retrieval over the
whole corpus, then precise reranking over the ~dozen candidates it returns.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger
from tnra.utils.paths import HF_CACHE_DIR, ensure_dir

logger = get_logger(__name__)

DeviceChoice = Literal["auto", "cuda", "mps", "cpu"]


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------


class RerankerConfig(BaseModel):
    """Validated config for the reranker.

    Built from the `retrieval.reranker` section of `configs/retrieval.yaml`.
    """

    enabled: bool = True
    model: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=100)  # final number of passages to keep
    device: DeviceChoice = "auto"
    batch_size: int = Field(gt=0, le=256)


# -----------------------------------------------------------------------------
# Device resolution (same logic as embedding.py)
# -----------------------------------------------------------------------------


def _resolve_device(choice: DeviceChoice) -> str:
    """Pick a concrete torch device string from the config preference."""
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return choice


# -----------------------------------------------------------------------------
# Reranker
# -----------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker over retrieval candidates.

    The model is loaded once at construction (a few hundred MB) and reused
    across queries. Keep the Reranker alive for the whole app lifetime.
    """

    def __init__(self, cfg: RerankerConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        cache_dir = ensure_dir(HF_CACHE_DIR)

        logger.info("Loading reranker model: %s on device=%s", cfg.model, self.device)
        # CrossEncoder is a distinct sentence-transformers class from
        # SentenceTransformer: it scores text PAIRS rather than encoding singles.
        self.model = CrossEncoder(
            cfg.model,
            device=self.device,
            model_kwargs={"cache_dir": str(cache_dir)},
        )
        logger.info("Reranker loaded")

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Re-score candidates with the cross-encoder and return the top_k.

        Args:
            query: The user's natural-language question.
            candidates: The shortlist to rerank (output of fusion + dedup).
            top_k: Number of results to keep after reranking.

        Returns:
            A new list of RetrievalResult, sorted by descending cross-encoder
            score, truncated to top_k. The `score` field holds the
            cross-encoder relevance score (higher = better; not comparable to
            dense / sparse / RRF scores).
        """
        if not candidates:
            logger.warning("rerank() called with no candidates")
            return []

        # Build (query, chunk_text) pairs — the cross-encoder's input format.
        pairs = [(query, c.text) for c in candidates]

        # One forward pass per pair, batched on the GPU.
        scores = self.model.predict(
            pairs,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
        )

        # Attach scores, sort best-first, truncate.
        rescored = [
            candidate.model_copy(update={"score": float(score)})
            for candidate, score in zip(candidates, scores, strict=False)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        top_results = rescored[:top_k]

        logger.info(
            "Reranking: %d candidates → top %d (best score=%.3f)",
            len(candidates),
            len(top_results),
            top_results[0].score if top_results else 0.0,
        )
        return top_results
