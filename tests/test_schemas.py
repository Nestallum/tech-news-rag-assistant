"""Tests for the generation response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tnra.generation.schemas import RAGResponse, Source


def test_source_builds_with_valid_fields() -> None:
    """A Source is created from valid field values."""
    source = Source(
        marker=1,
        article_title="Apple unveils the M5 chip",
        article_url="https://example.com/m5",
        source="The Verge",
    )
    assert source.marker == 1
    assert source.source == "The Verge"


def test_source_rejects_marker_below_one() -> None:
    """A marker must be >= 1; zero is rejected."""
    with pytest.raises(ValidationError):
        Source(
            marker=0,
            article_title="x",
            article_url="https://example.com/x",
            source="The Verge",
        )


def test_rag_response_defaults() -> None:
    """A RAGResponse built with only an answer gets sane defaults."""
    response = RAGResponse(answer="Some answer.")
    assert response.sources == []
    assert response.guard_triggered is False


def test_rag_response_with_sources() -> None:
    """A RAGResponse carries its list of cited sources."""
    source = Source(
        marker=1,
        article_title="Apple unveils the M5 chip",
        article_url="https://example.com/m5",
        source="The Verge",
    )
    response = RAGResponse(answer="The M5 is fast.", sources=[source])
    assert len(response.sources) == 1
    assert response.sources[0].marker == 1


def test_rag_response_requires_answer() -> None:
    """The answer field is mandatory — building without it fails."""
    with pytest.raises(ValidationError):
        RAGResponse()  # type: ignore
