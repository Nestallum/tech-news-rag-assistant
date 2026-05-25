"""Tests for the LLM-as-judge pure helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tnra.evaluation.judge import JudgeScore, _extract_json


def test_extract_json_plain() -> None:
    """A bare JSON object is returned unchanged."""
    raw = '{"faithfulness": 4, "relevance": 5, "reasoning": "ok"}'
    assert _extract_json(raw) == raw


def test_extract_json_strips_code_fences() -> None:
    """JSON wrapped in markdown code fences is unwrapped."""
    raw = '```json\n{"faithfulness": 3, "relevance": 3, "reasoning": "x"}\n```'
    assert _extract_json(raw) == '{"faithfulness": 3, "relevance": 3, "reasoning": "x"}'


def test_extract_json_strips_surrounding_text() -> None:
    """JSON preceded by chatter is still extracted."""
    raw = 'Here is my evaluation: {"faithfulness": 2, "relevance": 4, "reasoning": "y"}'
    assert _extract_json(raw) == '{"faithfulness": 2, "relevance": 4, "reasoning": "y"}'


def test_extract_json_raises_when_absent() -> None:
    """A reply with no JSON object raises a clear error."""
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_judge_score_valid() -> None:
    """A JudgeScore is built from valid scores."""
    score = JudgeScore(faithfulness=4, relevance=5, reasoning="solid")
    assert score.faithfulness == 4


def test_judge_score_rejects_out_of_range() -> None:
    """A score outside the 1-5 range is rejected."""
    with pytest.raises(ValidationError):
        JudgeScore(faithfulness=7, relevance=3, reasoning="too high")
