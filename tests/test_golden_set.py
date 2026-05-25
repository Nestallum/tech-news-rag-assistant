"""Tests for golden set loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tnra.evaluation.golden_set import GoldenQuestion, load_golden_set


def test_golden_question_builds_with_valid_fields() -> None:
    """A GoldenQuestion is created from valid field values."""
    q = GoldenQuestion(
        id="q01",
        question="How much will Anthropic pay xAI?",
        expected_article_urls=["https://example.com/a"],
    )
    assert q.id == "q01"
    assert len(q.expected_article_urls) == 1


def test_golden_question_rejects_empty_url_list() -> None:
    """A question with no expected URLs is rejected."""
    with pytest.raises(ValidationError):
        GoldenQuestion(id="q01", question="Some question?", expected_article_urls=[])


def test_load_golden_set_reads_all_questions() -> None:
    """The real golden set file loads into 15 validated questions."""
    questions = load_golden_set(Path("eval/golden_set/golden_set.json"))
    assert len(questions) == 15
    assert all(isinstance(q, GoldenQuestion) for q in questions)


def test_load_golden_set_has_unique_ids() -> None:
    """Every question in the golden set has a distinct id."""
    questions = load_golden_set(Path("eval/golden_set/golden_set.json"))
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))
