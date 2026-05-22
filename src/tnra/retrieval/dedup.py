"""Article-level deduplication of retrieval results.

A long article is split into several chunks at ingestion time. Multiple chunks
of the same article can match a single query, so a ranked result list often
contains the same article more than once.

This is a problem downstream:
  - the LLM gets fewer distinct articles than the top_k suggests
  - the same article would be cited twice in the final answer

`deduplicate_by_article` keeps only the best-ranked chunk per article. Because
the input list is already sorted best-first (it comes out of RRF fusion), the
first chunk seen for a given article URL is its best one — so we simply keep
the first occurrence and drop the rest. No score comparison needed.
"""

from __future__ import annotations

from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


def deduplicate_by_article(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep only the best-ranked chunk per article.

    Args:
        results: A ranked list of RetrievalResult, sorted best-first. The
            ordering assumption is essential: the first chunk seen for an
            article is treated as its best one.

    Returns:
        A new ranked list with at most one chunk per `article_url`, preserving
        the original relative order.
    """
    seen_urls: set[str] = set()
    deduped: list[RetrievalResult] = []

    for result in results:
        if result.article_url in seen_urls:
            continue
        seen_urls.add(result.article_url)
        deduped.append(result)

    dropped = len(results) - len(deduped)
    logger.info(
        "Article-level dedup: %d → %d results (dropped %d same-article chunks)",
        len(results),
        len(deduped),
        dropped,
    )
    return deduped
