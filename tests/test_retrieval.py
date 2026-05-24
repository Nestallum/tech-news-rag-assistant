"""Tests for the pure functions of the retrieval stage."""

from __future__ import annotations

import pytest

from tnra.retrieval.dedup import deduplicate_by_article
from tnra.retrieval.fusion import reciprocal_rank_fusion
from tnra.retrieval.schemas import RetrievalResult
from tnra.retrieval.sparse import tokenize


def test_tokenize_lowercases() -> None:
    """Tokens are lowercased."""
    assert tokenize("Apple GROQ") == ["apple", "groq"]


def test_tokenize_splits_on_punctuation() -> None:
    """Punctuation and symbols are separators, not kept."""
    assert tokenize("Apple's M5 chip!") == ["apple", "s", "m5", "chip"]


def test_tokenize_keeps_alphanumeric_tokens() -> None:
    """Digits and letters survive; version-like tokens stay intact."""
    assert tokenize("Llama 3.3 70b") == ["llama", "3", "3", "70b"]


def test_tokenize_empty_string() -> None:
    """An empty string yields no tokens."""
    assert tokenize("") == []


def _result(chunk_id: str, article_url: str, score: float) -> RetrievalResult:
    """Build a minimal RetrievalResult for dedup tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text="some text",
        score=score,
        article_url=article_url,
        article_title="Some article",
        source="The Verge",
        feed_name="the_verge",
        chunk_index=0,
        published_at="2026-06-10",
    )


def test_dedup_keeps_one_chunk_per_article() -> None:
    """Two chunks from the same article collapse to the first one."""
    results = [
        _result("a_0", "https://example.com/a", 0.90),
        _result("a_1", "https://example.com/a", 0.80),
        _result("b_0", "https://example.com/b", 0.70),
    ]
    deduped = deduplicate_by_article(results)
    assert [r.chunk_id for r in deduped] == ["a_0", "b_0"]


def test_dedup_first_seen_wins() -> None:
    """The chunk kept is the one appearing first in the input order."""
    results = [
        _result("a_top", "https://example.com/a", 0.95),
        _result("a_low", "https://example.com/a", 0.10),
    ]
    deduped = deduplicate_by_article(results)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == "a_top"


def test_dedup_preserves_order() -> None:
    """Dedup keeps the relative order of the surviving chunks."""
    results = [
        _result("a_0", "https://example.com/a", 0.90),
        _result("b_0", "https://example.com/b", 0.80),
        _result("a_1", "https://example.com/a", 0.70),
        _result("c_0", "https://example.com/c", 0.60),
    ]
    deduped = deduplicate_by_article(results)
    assert [r.chunk_id for r in deduped] == ["a_0", "b_0", "c_0"]


def test_dedup_empty_list() -> None:
    """An empty list yields an empty list."""
    assert deduplicate_by_article([]) == []


def _ranked(chunk_id: str) -> RetrievalResult:
    """Build a minimal RetrievalResult identified by chunk_id."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        score=0.0,
        article_url=f"https://example.com/{chunk_id}",
        article_title="Some article",
        source="The Verge",
        feed_name="the_verge",
        chunk_index=0,
        published_at="2026-06-10",
    )


def test_rrf_single_list_preserves_order() -> None:
    """With one list, RRF keeps the original ranking."""
    fused = reciprocal_rank_fusion([[_ranked("a"), _ranked("b"), _ranked("c")]], k=1)
    assert [r.chunk_id for r in fused] == ["a", "b", "c"]


def test_rrf_score_uses_one_based_rank() -> None:
    """Top result of a single list scores 1/(k+1)."""
    fused = reciprocal_rank_fusion([[_ranked("a"), _ranked("b")]], k=1)
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["a"] == 1 / 2  # rank 1 -> 1/(1+1)
    assert scores["b"] == 1 / 3  # rank 2 -> 1/(1+2)


def test_rrf_chunk_in_both_lists_sums_contributions() -> None:
    """A chunk present in two lists accumulates both contributions."""
    dense = [_ranked("a"), _ranked("b")]
    sparse = [_ranked("b"), _ranked("c")]
    fused = reciprocal_rank_fusion([dense, sparse], k=1)
    scores = {r.chunk_id: r.score for r in fused}
    # b: rank 2 in dense (1/3) + rank 1 in sparse (1/2) = 5/6
    assert scores["b"] == pytest.approx(1 / 3 + 1 / 2)
    # b outscores a (1/2) and c (1/3): it appears in both lists.
    assert fused[0].chunk_id == "b"


def test_rrf_deduplicates_by_chunk_id() -> None:
    """A chunk in both lists yields a single fused entry."""
    fused = reciprocal_rank_fusion([[_ranked("a")], [_ranked("a")]], k=1)
    assert len(fused) == 1
    assert fused[0].chunk_id == "a"


def test_rrf_empty_lists() -> None:
    """Fusing empty lists yields an empty result."""
    assert reciprocal_rank_fusion([[], []], k=1) == []
