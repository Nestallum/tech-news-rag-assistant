"""Retrieval metrics for the evaluation stage.

Pure functions comparing the article URLs a retriever returned against the
article URLs a golden question expected. Built incrementally.
"""

from __future__ import annotations


def recall_at_k(retrieved_urls: list[str], expected_urls: list[str], k: int) -> float:
    """Recall@k for a single question.

    Returns 1.0 if at least one expected article URL appears within the top-k
    retrieved URLs, else 0.0. When several articles are expected, finding any
    one of them counts as a hit.

    Args:
        retrieved_urls: Article URLs returned by the retriever, ranked best-first.
        expected_urls: Article URLs the golden question expects (at least one).
        k: How many of the top retrieved URLs to consider.

    Returns:
        1.0 on a hit, 0.0 otherwise.
    """
    top_k = retrieved_urls[:k]
    hit = any(url in top_k for url in expected_urls)
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_urls: list[str], expected_urls: list[str]) -> float:
    """Reciprocal rank for a single question.

    Finds the rank of the first expected article URL within the retrieved
    list (rank 1 = first position) and returns 1 / rank. Returns 0.0 if no
    expected URL is retrieved. When several articles are expected, the
    best-ranked (earliest) one is used.

    Args:
        retrieved_urls: Article URLs returned by the retriever, ranked best-first.
        expected_urls: Article URLs the golden question expects (at least one).

    Returns:
        1 / rank of the first expected URL found, or 0.0 if none is found.
    """
    expected = set(expected_urls)
    for rank, url in enumerate(retrieved_urls, start=1):
        if url in expected:
            return 1.0 / rank
    return 0.0
