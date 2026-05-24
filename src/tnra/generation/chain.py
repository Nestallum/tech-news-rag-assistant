"""RAG chain for the generation stage.

Wires the pieces together: retrieve → guard → prompt → LLM → typed response.
Built incrementally, one helper at a time.
"""

from __future__ import annotations

import re

from tnra.generation.guard import GuardConfig, passes_guard
from tnra.generation.llm import LLMClient
from tnra.generation.prompt import PROMPT, format_context
from tnra.generation.schemas import RAGResponse, Source
from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

# Matches a citation marker like [1], [12]... and captures the number inside.
_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def extract_sources(answer: str, results: list[RetrievalResult]) -> list[Source]:
    """Map the [n] markers used in the answer back to their source articles.

    The LLM cites passages by number; passage [n] is the n-th retrieved
    result (markers are 1-based, list indices are 0-based, so [n] -> results[n - 1]).

    Markers that fall outside the results list are ignored: the LLM can emit
    a number that was never provided. Each cited source appears once, in the
    order it first appears in the answer.

    Args:
        answer: The LLM answer text, containing [n] markers.
        results: The retrieved passages, in the SAME order used to build the
            prompt (this ordering is what makes the markers meaningful).

    Returns:
        The cited sources, one Source per distinct valid marker.
    """
    sources: list[Source] = []
    seen: set[int] = set()

    for match in _MARKER_PATTERN.finditer(answer):
        marker = int(match.group(1))

        # Skip markers already handled, or pointing outside the results list.
        if marker in seen or not (1 <= marker <= len(results)):
            continue
        seen.add(marker)

        result = results[marker - 1]
        sources.append(
            Source(
                marker=marker,
                article_title=result.article_title,
                article_url=result.article_url,
                source=result.source,
            )
        )

    return sources


def generate_answer(
    question: str,
    results: list[RetrievalResult],
    llm: LLMClient,
) -> str:
    """Call the LLM on the retrieved passages and return its raw answer text.

    Assembles the prompt from the numbered passages and the question, sends it
    to the LLM, and returns the reply as-is — still containing [n] markers,
    not yet parsed into sources.

    Args:
        question: The user's question.
        results: Retrieved passages, ranked best-first. Their order must match
            the order used everywhere else in the chain (the markers depend
            on it).
        llm: The loaded LLM client.

    Returns:
        The LLM's raw answer text.
    """
    context = format_context(results)
    messages = PROMPT.format_messages(context=context, question=question)
    return llm.invoke(messages)  # type: ignore


def answer_question(
    question: str,
    results: list[RetrievalResult],
    llm: LLMClient,
    guard_cfg: GuardConfig,
) -> RAGResponse:
    """Run the full generation pipeline for one question.

    Steps: check the retrieval guard; if it fails, refuse without calling the
    LLM. Otherwise call the LLM, map its citation markers to sources, and pack
    everything into a RAGResponse.

    Args:
        question: The user's question.
        results: Retrieved passages, ranked best-first (output of Phase 2).
        llm: The loaded LLM client.
        guard_cfg: Guard configuration (threshold and refusal message).

    Returns:
        A complete RAGResponse — either a real answer or a guard refusal.
    """
    # Guard: if retrieval is too weak, refuse without calling the LLM.
    if not passes_guard(results, guard_cfg):
        logger.info("Guard triggered for question %r — refusing", question[:60])
        return RAGResponse(
            answer=guard_cfg.refusal_message,
            sources=[],
            guard_triggered=True,
        )

    # Retrieval is strong enough: call the LLM and build the cited sources.
    raw_answer = generate_answer(question, results, llm)
    sources = extract_sources(raw_answer, results)

    logger.info("Answered question %r — %d source(s) cited", question[:60], len(sources))
    return RAGResponse(
        answer=raw_answer,
        sources=sources,
        guard_triggered=False,
    )
