"""Hybrid fusion of dense and sparse results via Reciprocal Rank Fusion (RRF).

Dense and sparse retrievers produce scores on incomparable scales (dense in
[0,1], BM25 unbounded). RRF sidesteps this entirely: it ignores raw scores and
fuses on RANKS only.

For each chunk, across every ranked list it appears in:

    rrf_score(chunk) = sum over lists of  1 / (k + rank_in_that_list)

with rank starting at 1. Properties:
  - higher rank (closer to 1) contributes more
  - k (=60, the standard from Cormack et al. 2009) dampens the gap between
    adjacent ranks, making fusion stable and tuning-free
  - a chunk found by BOTH retrievers accumulates both contributions, so
    cross-retriever agreement naturally floats to the top

This is the industry-default hybrid fusion: no score normalization, no
per-corpus calibration.
"""

from __future__ import annotations

from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# RRF fusion
# -----------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse several ranked result lists into one, using Reciprocal Rank Fusion.

    Args:
        ranked_lists: A list of ranked result lists (e.g. [dense_results,
            sparse_results]). Each inner list must already be sorted best-first.
        k: The RRF damping constant. 60 is the canonical default.

    Returns:
        A single list of RetrievalResult, sorted by descending RRF score.
        Each result's `score` field is replaced with its RRF score (a small
        positive float — not comparable to dense/sparse scores).

    Notes:
        Chunks are deduplicated by `chunk_id`: the same chunk appearing in
        multiple input lists is fused into one entry whose RRF score is the
        sum of its per-list contributions. The result object (text, metadata)
        is taken from the first list in which the chunk was seen.
    """
    # chunk_id -> accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id -> the RetrievalResult object (kept from first occurrence)
    seen: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            contribution = 1.0 / (k + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + contribution
            if result.chunk_id not in seen:
                seen[result.chunk_id] = result

    # Rebuild RetrievalResult objects with the RRF score in the score field.
    fused: list[RetrievalResult] = []
    for chunk_id, rrf_score in rrf_scores.items():
        original = seen[chunk_id]
        fused.append(original.model_copy(update={"score": rrf_score}))

    # Sort by descending RRF score (best first).
    fused.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        "RRF fusion: %d input lists → %d unique chunks",
        len(ranked_lists),
        len(fused),
    )
    return fused
