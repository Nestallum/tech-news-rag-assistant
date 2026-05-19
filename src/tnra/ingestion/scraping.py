"""HTML scraping of article pages.

RSS feeds in 2026 ship excerpts only — the real article body lives on the
publisher's HTML page. This module fetches each article URL and extracts the
clean main content using `trafilatura`, the reference library for boilerplate
removal (used in Common Crawl, academic NLP pipelines, and most production
RAG systems on news data).

Pipeline:
    FeedEntry  ──HTTP GET──►  raw HTML  ──trafilatura──►  clean text  ──►  RawArticle

Failures (network error, paywall, empty extraction) are logged and the entry
is dropped from the pipeline. One bad article should never kill the batch.
"""

from __future__ import annotations

import httpx
import trafilatura
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from tnra.ingestion.schemas import FeedEntry, RawArticle
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum text length below which we consider the extraction failed.
# Articles shorter than this are almost always paywalls, error pages, or
# extraction failures masquerading as success.
_MIN_EXTRACTED_CHARS = 500


# -----------------------------------------------------------------------------
# HTTP fetch (retried, distinct UA from RSS to mimic a regular reader)
# -----------------------------------------------------------------------------


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Decide whether to retry a failed HTTP call.

    Retry on:
      - Transport errors (timeout, DNS, connection refused)
      - 5xx server errors (likely transient overload)
    Do NOT retry on:
      - 4xx client errors (403 Forbidden, 404 Not Found, 429 Too Many Requests):
        the server is explicitly rejecting us. Retrying makes things worse.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_html(url: str, *, user_agent: str, timeout_s: float) -> str:
    """GET an article URL and return its HTML as a string.

    Retried only on transient errors (transport failures, 5xx). Client errors
    (403, 404, 429) are not retried — the server is rejecting us deliberately
    and hammering it won't help.
    """
    response = httpx.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=timeout_s,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.text


# -----------------------------------------------------------------------------
# Trafilatura extraction
# -----------------------------------------------------------------------------


def _extract_main_content(html: str, *, source_url: str) -> str | None:
    """Extract the article's main body from raw HTML using trafilatura.

    We request markdown output: it preserves paragraph breaks and lists,
    which help downstream chunking respect natural article structure.

    Trafilatura returns None if it can't confidently identify article content
    (which happens on paywalled pages, infinite-scroll homepages, or unusual
    layouts). We propagate None — the caller handles the skip.
    """
    text = trafilatura.extract(
        html,
        url=source_url,  # helps trafilatura with relative URL resolution and date heuristics
        output_format="markdown",  # preserves paragraph structure for the chunker
        include_comments=False,  # comments are noise for our RAG use case
        include_tables=True,  # tables can carry valuable info (specs, benchmarks)
        favor_precision=True,  # prefer dropping ambiguous content over including boilerplate
        deduplicate=True,  # removes repeated boilerplate blocks if any leak through
    )
    return text if text else None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def scrape_article(
    entry: FeedEntry,
    *,
    user_agent: str,
    timeout_s: float = 30.0,
) -> RawArticle | None:
    """Scrape one article: FeedEntry (metadata only) → RawArticle (with full content).

    Args:
        entry: The RSS metadata record pointing to the article URL.
        user_agent: HTTP User-Agent string.
        timeout_s: HTTP timeout per attempt.

    Returns:
        A RawArticle with `content_html` filled with clean extracted text,
        or None if scraping/extraction failed for any reason (network error,
        paywall, empty extraction, content too short).
    """
    url_str = str(entry.url)

    try:
        html = _fetch_html(url_str, user_agent=user_agent, timeout_s=timeout_s)
    except httpx.HTTPError as e:
        logger.warning("HTTP error scraping %s: %s", url_str, e)
        return None

    text = _extract_main_content(html, source_url=url_str)
    if text is None:
        logger.warning("Trafilatura extracted nothing from %s", url_str)
        return None

    if len(text) < _MIN_EXTRACTED_CHARS:
        logger.warning(
            "Extracted text too short (%d chars, min=%d) from %s — likely paywall or failure",
            len(text),
            _MIN_EXTRACTED_CHARS,
            url_str,
        )
        return None

    try:
        return RawArticle(
            url=url_str,  # type: ignore[arg-type]
            title=entry.title,
            content_html=text,
            source=entry.source,
            feed_name=entry.feed_name,
            published_at=entry.published_at,
            fetched_at=entry.fetched_at,
            author=entry.author,
        )
    except Exception as e:
        logger.warning("Failed to build RawArticle for %s: %s", url_str, e)
        return None


def scrape_articles(
    entries: list[FeedEntry],
    *,
    user_agent: str,
    timeout_s: float = 30.0,
    delay_s: float = 0.25,
) -> list[RawArticle]:
    """Scrape a list of feed entries sequentially with a polite delay between calls.

    Sequential (not async/parallel) on purpose for v1:
      - Simpler code, easier to reason about + debug
      - Polite to publishers: no risk of getting rate-limited or blocked
      - Ingestion runs offline (background task), latency is not critical
      - 100 articles × ~1.5s + 0.25s delay = 3 min total — perfectly acceptable
    """  # noqa: RUF002
    import time

    logger.info("Scraping %d articles (delay=%.2fs between calls)...", len(entries), delay_s)
    articles: list[RawArticle] = []
    for i, entry in enumerate(entries, start=1):
        article = scrape_article(entry, user_agent=user_agent, timeout_s=timeout_s)
        if article is not None:
            articles.append(article)
        if i < len(entries):
            time.sleep(delay_s)
        if i % 10 == 0 or i == len(entries):
            logger.info("Scraping progress: %d/%d (%d successful)", i, len(entries), len(articles))
    logger.info("Scraping done: %d/%d articles successfully scraped", len(articles), len(entries))
    return articles
