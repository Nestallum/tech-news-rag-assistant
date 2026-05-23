"""Anti-hallucination guard for the generation stage.

Before calling the LLM, we check whether retrieval actually found anything
relevant. Retrieval always returns results, even for off-topic questions —
just with low scores. If the best retrieved passage scores below a threshold,
we skip the LLM entirely and refuse, rather than feeding it weak passages it
might hallucinate an answer from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tnra.retrieval.schemas import RetrievalResult


class GuardConfig(BaseModel):
    """Validated schema for the `guard` section of generation.yaml."""

    min_retrieval_score: float = Field(ge=0.0, le=1.0)


def passes_guard(results: list[RetrievalResult], cfg: GuardConfig) -> bool:
    """Decide whether retrieval is strong enough to call the LLM.

    Args:
        results: Retrieved passages, ranked best-first (output of Phase 2).
        cfg: Guard configuration holding the minimum score threshold.

    Returns:
        True if the best passage scores at or above the threshold (proceed to
        the LLM), False otherwise (skip the LLM and refuse).
    """
    if not results:
        return False
    best_score = results[0].score
    return best_score >= cfg.min_retrieval_score
