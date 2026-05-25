"""Golden set loading for the evaluation stage.

The golden set is a JSON file of fact-based questions, each mapped to the
article URL(s) that should be retrieved. This module loads it into validated
Python objects.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    """One golden-set entry: a question and its expected article URL(s)."""

    id: str
    question: str
    expected_article_urls: list[str] = Field(min_length=1)


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Load and validate the golden set from a JSON file.

    Args:
        path: Path to the golden set JSON file.

    Returns:
        The list of golden questions, validated.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = [GoldenQuestion.model_validate(q) for q in raw["questions"]]
    return questions
