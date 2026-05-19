"""RSS feed parsing.

This module turns RSS feed URLs into validated `FeedEntry` objects — lightweight
metadata records (url, title, date, source). The actual article body is NOT
fetched here: in 2026, virtually all major tech publishers serve excerpt-only
RSS feeds, so full-text content is retrieved separately by `scraping.py`.

Network calls are retried with exponential backoff via `tenacity`. Parse
errors on individual entries are logged and skipped — one bad entry should
never bring down the ingestion of an entire feed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import struct_time
from typing import Any

import feedparser
import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tnra.ingestion.schemas import FeedEntry
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# HTTP fetch (retried)
# -----------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _http_get(url: str, *, user_agent: str, timeout_s: float) -> bytes:
    """GET a URL with retries on transient network errors.

    `tenacity` decorator: retries up to 3 times on connection errors or HTTP 5xx,
    with exponential backoff (2s, 4s, 8s...). Bubbles up the final exception
    after exhausting attempts so the caller can decide what to do.
    """
    response = httpx.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/rss+xml, application/xml, */*",
        },
        timeout=timeout_s,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.content


# -----------------------------------------------------------------------------
# Per-entry parsing
# -----------------------------------------------------------------------------


def _extract_summary(entry: Any) -> str:
    """Pull the short excerpt advertised by the RSS feed.

    Stored on FeedEntry as a fallback only — the real content is fetched
    separately by scraping.py. We still keep it because:
      (a) it's useful for debugging ("did the RSS feed even mention X?")
      (b) it can serve as last-resort content if HTML scraping fails

    We prefer `summary` over `description` (synonyms in most feeds) and skip
    the multi-value `content` field entirely — it's almost never present
    on modern excerpt-only feeds.
    """
    for attr in ("summary", "description"):
        value = getattr(entry, attr, "")
        if value:
            return str(value)
    return ""


def _parse_published_at(entry: Any) -> datetime | None:
    """Convert feedparser's `published_parsed` (struct_time) to a UTC datetime.

    Feedparser normalizes most date formats into a `time.struct_time` in UTC.
    Some feeds skip the date entirely or set malformed values — we return None
    in that case rather than guessing.
    """
    parsed: struct_time | None = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if parsed is None:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=UTC)
    except TypeError, ValueError:
        return None


def _entry_to_feed_entry(
    entry: Any,
    *,
    source: str,
    feed_name: str,
    fetched_at: datetime,
) -> FeedEntry | None:
    """Convert a feedparser entry into a validated FeedEntry, or None on failure.

    Returns None (and logs a warning) if the entry is missing essential fields
    (url or title) or fails Pydantic validation. The caller skips Nones and
    keeps processing the rest of the feed.
    """
    url = getattr(entry, "link", "")
    title = getattr(entry, "title", "")

    if not url or not title:
        logger.warning(
            "Skipping entry with missing url/title in feed=%s (has_url=%s, has_title=%s)",
            feed_name,
            bool(url),
            bool(title),
        )
        return None

    try:
        return FeedEntry(
            url=url,  # type: ignore[arg-type]
            title=title,
            summary=_extract_summary(entry),
            source=source,
            feed_name=feed_name,
            published_at=_parse_published_at(entry),
            fetched_at=fetched_at,
            author=getattr(entry, "author", None) or None,
        )
    except ValidationError as e:
        logger.warning("Pydantic validation failed for entry in feed=%s: %s", feed_name, e)
        return None


# -----------------------------------------------------------------------------
# Public API: fetch one feed
# -----------------------------------------------------------------------------


def fetch_feed(
    url: str,
    *,
    feed_name: str,
    source: str,
    user_agent: str,
    timeout_s: float = 30.0,
) -> list[FeedEntry]:
    """Fetch and parse an RSS feed, returning validated metadata entries.

    Args:
        url: RSS/Atom feed URL.
        feed_name: Internal identifier of the feed (e.g. "techcrunch_ai").
        source: Human-readable publisher name (e.g. "TechCrunch").
        user_agent: HTTP User-Agent string to identify ourselves to servers.
        timeout_s: HTTP timeout per attempt.

    Returns:
        List of valid FeedEntry objects (metadata only — no full content).
        Malformed entries are logged and dropped silently; we never raise on
        a single bad entry.

    Raises:
        httpx.HTTPError: if all HTTP retries fail (network down, 5xx, etc.).
        ValueError: if the feed is malformed AND returned no entries at all.
    """
    logger.info("Fetching feed: %s (%s)", feed_name, url)
    raw_bytes = _http_get(url, user_agent=user_agent, timeout_s=timeout_s)

    # feedparser.parse accepts bytes, str, file objects, or URLs. We pass bytes
    # so it doesn't redo the HTTP request itself.
    parsed = feedparser.parse(raw_bytes)
    if parsed.bozo and not parsed.entries:
        # `bozo` flags malformed feeds; if we also have no entries, it's fatal.
        raise ValueError(
            f"Feed {feed_name} is malformed and has no entries: {parsed.bozo_exception}"
        )

    fetched_at = datetime.now(tz=UTC)
    entries: list[FeedEntry] = []
    for entry in parsed.entries:
        feed_entry = _entry_to_feed_entry(
            entry, source=source, feed_name=feed_name, fetched_at=fetched_at
        )
        if feed_entry is not None:
            entries.append(feed_entry)

    logger.info(
        "Parsed feed: %s — %d valid entries out of %d total",
        feed_name,
        len(entries),
        len(parsed.entries),
    )
    return entries
