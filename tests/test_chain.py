"""Tests for the pure functions of the generation chain."""

from __future__ import annotations

import pytest

from tnra.generation.chain import extract_sources, strip_markers
from tnra.retrieval.schemas import RetrievalResult


@pytest.fixture
def results() -> list[RetrievalResult]:
    """Three minimal retrieved passages, labelled by index."""
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            text=f"text {i}",
            score=0.9,
            article_url=f"https://example.com/article-{i}",
            article_title=f"Article {i}",
            source="The Verge",
            feed_name="the_verge",
            chunk_index=0,
            published_at="2026-06-10",
        )
        for i in (1, 2, 3)
    ]


def test_extract_sources_basic(results: list[RetrievalResult]) -> None:
    """Two distinct markers map to their two articles, in order of appearance."""
    sources = extract_sources("Claim A [1]. Claim B [2].", results)
    assert [s.marker for s in sources] == [1, 2]
    assert sources[0].article_title == "Article 1"


def test_extract_sources_deduplicates_repeated_marker(
    results: list[RetrievalResult],
) -> None:
    """A marker cited several times yields a single source."""
    sources = extract_sources("A [1]. More on A [1] again.", results)
    assert [s.marker for s in sources] == [1]


def test_extract_sources_ignores_out_of_range_marker(
    results: list[RetrievalResult],
) -> None:
    """A marker pointing past the results list is silently ignored."""
    sources = extract_sources("Valid [3]. Invalid [9].", results)
    assert [s.marker for s in sources] == [3]


def test_extract_sources_no_markers(results: list[RetrievalResult]) -> None:
    """An answer with no markers yields no sources."""
    sources = extract_sources("No citations here.", results)
    assert sources == []


def test_strip_markers_removes_single_marker() -> None:
    """A lone marker is removed, punctuation stays clean."""
    assert strip_markers("The chip is fast [1].") == "The chip is fast."


def test_strip_markers_removes_marker_run_with_comma() -> None:
    """A run like '[2], [3]' is removed without leaving an orphan comma."""
    assert strip_markers("It targets AI [2], [3].") == "It targets AI."


def test_strip_markers_removes_marker_run_with_space() -> None:
    """A run like '[4] [5]' is removed without leaving a double space."""
    assert strip_markers("See here [4] [5] now.") == "See here now."


def test_strip_markers_leaves_clean_text_untouched() -> None:
    """Text with no markers is returned unchanged."""
    assert strip_markers("Nothing to remove here.") == "Nothing to remove here."
