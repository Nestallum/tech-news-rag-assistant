"""RAG chain for the generation stage.

Wires the pieces together: retrieve → guard → prompt → LLM → typed response.
Built incrementally, one helper at a time.
"""

from __future__ import annotations

import re

from tnra.generation.schemas import Source
from tnra.retrieval.schemas import RetrievalResult

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
