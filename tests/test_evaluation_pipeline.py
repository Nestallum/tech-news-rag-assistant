"""Tests for the evaluation aggregation logic."""

from __future__ import annotations

import pytest

from tnra.evaluation.judge import JudgeScore
from tnra.evaluation.pipeline import QuestionResult, aggregate_results


def _qr(recall_5: float, rr: float, faith: int, rel: int) -> QuestionResult:
    """Build a minimal QuestionResult for aggregation tests."""
    return QuestionResult(
        id="qx",
        question="some question?",
        recall_at_1=recall_5,
        recall_at_3=recall_5,
        recall_at_5=recall_5,
        reciprocal_rank=rr,
        judge=JudgeScore(faithfulness=faith, relevance=rel, reasoning="x"),
    )


def test_aggregate_computes_means() -> None:
    """Means are the average of the per-question values."""
    results = [_qr(1.0, 1.0, 5, 5), _qr(0.0, 0.0, 3, 1)]
    report = aggregate_results(results)
    assert report.n_questions == 2
    assert report.mean_recall_at_5 == 0.5
    assert report.mean_reciprocal_rank == 0.5
    assert report.mean_faithfulness == 4.0
    assert report.mean_relevance == 3.0


def test_aggregate_keeps_per_question_detail() -> None:
    """The report keeps every per-question result."""
    results = [_qr(1.0, 1.0, 5, 5), _qr(1.0, 0.5, 4, 4)]
    report = aggregate_results(results)
    assert len(report.per_question) == 2


def test_aggregate_empty_raises() -> None:
    """Aggregating an empty list raises a clear error."""
    with pytest.raises(ValueError):
        aggregate_results([])
