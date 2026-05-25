"""LLM-as-judge for the evaluation stage.

Scores a generated answer against its source passages using an LLM. The judge
rates faithfulness (is the answer grounded in the passages?) and relevance
(does it answer the question?) on a 1-5 scale, returning structured JSON.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from tnra.generation.llm import LLMClient
from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


class JudgeScore(BaseModel):
    """Structured result of one LLM-as-judge evaluation."""

    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str


_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of question-answering \
systems. You are given a QUESTION, an ANSWER produced by a system, and the \
PASSAGES the system was given as sources.

Rate the answer on two criteria, each from 1 (poor) to 5 (excellent):
- faithfulness: is every claim in the answer supported by the passages? An \
answer that adds facts absent from the passages scores low, even if those \
facts are true.
- relevance: does the answer actually address the question?

Reply with ONLY a JSON object, no other text, in exactly this form:
{"faithfulness": <int>, "relevance": <int>, "reasoning": "<one short sentence>"}"""


def _build_user_prompt(question: str, answer: str, passages: list[RetrievalResult]) -> str:
    """Assemble the judge's user message from the evaluation inputs."""
    passages_block = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, 1))
    return f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nPASSAGES:\n{passages_block}"


def _extract_json(text: str) -> str:
    """Pull the JSON object out of the LLM reply, tolerating extra wrapping."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge reply: {text!r}")
    return text[start : end + 1]


def judge_answer(
    question: str,
    answer: str,
    passages: list[RetrievalResult],
    llm: LLMClient,
) -> JudgeScore:
    """Score a generated answer with the LLM-as-judge.

    Args:
        question: The original question.
        answer: The answer produced by the RAG system.
        passages: The passages the system used as sources.
        llm: The loaded LLM client.

    Returns:
        A JudgeScore with faithfulness, relevance, and short reasoning.
    """

    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(question, answer, passages)),
    ]
    raw = llm.invoke(messages)
    payload = json.loads(_extract_json(raw))
    score = JudgeScore.model_validate(payload)
    logger.info("Judge: faithfulness=%d relevance=%d", score.faithfulness, score.relevance)
    return score
