"""Tests for the anti-hallucination guard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tnra.generation.guard import GuardConfig, passes_guard
from tnra.retrieval.schemas import RetrievalResult


def _result(score: float) -> RetrievalResult:
    """Build a minimal RetrievalResult with the given score."""
    return RetrievalResult(
        chunk_id="chunk_x",
        text="some text",
        score=score,
        article_url="https://example.com/x",
        article_title="Some article",
        source="The Verge",
        feed_name="the_verge",
        chunk_index=0,
        published_at="2026-06-10",
    )


@pytest.fixture
def cfg() -> GuardConfig:
    """A guard config with a 0.3 threshold."""
    return GuardConfig(min_retrieval_score=0.3, refusal_message="No information.")


def test_passes_guard_strong_retrieval(cfg: GuardConfig) -> None:
    """Top score above the threshold: the guard lets the query through."""
    results = [_result(0.95), _result(0.40)]
    assert passes_guard(results, cfg) is True


def test_passes_guard_weak_retrieval(cfg: GuardConfig) -> None:
    """Top score below the threshold: the guard blocks."""
    results = [_result(0.20), _result(0.10)]
    assert passes_guard(results, cfg) is False


def test_passes_guard_empty_results(cfg: GuardConfig) -> None:
    """No results at all: the guard blocks."""
    assert passes_guard([], cfg) is False


def test_passes_guard_score_exactly_at_threshold(cfg: GuardConfig) -> None:
    """A score exactly equal to the threshold passes (boundary case)."""
    assert passes_guard([_result(0.30)], cfg) is True


def test_guard_config_rejects_out_of_range_score() -> None:
    """A threshold outside [0, 1] is rejected by the schema."""
    with pytest.raises(ValidationError):
        GuardConfig(min_retrieval_score=1.5, refusal_message="x")
