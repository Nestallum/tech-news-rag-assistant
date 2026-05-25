"""Tests for the retrieval metrics."""

from __future__ import annotations

from tnra.evaluation.metrics import recall_at_k, reciprocal_rank

_RETRIEVED = ["url_a", "url_b", "url_c", "url_d", "url_e"]


def test_recall_hit_within_k() -> None:
    """An expected URL inside the top-k counts as a hit."""
    assert recall_at_k(_RETRIEVED, ["url_b"], k=3) == 1.0


def test_recall_miss_beyond_k() -> None:
    """An expected URL retrieved but beyond the top-k is a miss."""
    assert recall_at_k(_RETRIEVED, ["url_e"], k=3) == 0.0


def test_recall_miss_not_retrieved() -> None:
    """An expected URL never retrieved is a miss."""
    assert recall_at_k(_RETRIEVED, ["url_x"], k=5) == 0.0


def test_recall_multi_article_any_hit() -> None:
    """With several expected URLs, finding any one within k is a hit."""
    assert recall_at_k(_RETRIEVED, ["url_x", "url_c"], k=3) == 1.0


def test_reciprocal_rank_first_position() -> None:
    """An expected URL in first position scores 1.0."""
    assert reciprocal_rank(_RETRIEVED, ["url_a"]) == 1.0


def test_reciprocal_rank_third_position() -> None:
    """An expected URL in third position scores 1/3."""
    assert reciprocal_rank(_RETRIEVED, ["url_c"]) == 1 / 3


def test_reciprocal_rank_not_found() -> None:
    """An expected URL never retrieved scores 0.0."""
    assert reciprocal_rank(_RETRIEVED, ["url_x"]) == 0.0


def test_reciprocal_rank_multi_article_uses_best() -> None:
    """With several expected URLs, the best-ranked one is used."""
    assert reciprocal_rank(_RETRIEVED, ["url_d", "url_b"]) == 0.5
