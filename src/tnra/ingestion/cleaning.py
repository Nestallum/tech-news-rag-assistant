"""Article cleaning and deduplication.

Takes the raw scraped articles produced by `scraping.py` and turns them into
deduplicated, normalized `CleanedArticle` objects ready for chunking.

Two-level deduplication:
    1. Canonical URL (stripped of tracking params): catches the common case of
       the same article shared with different UTM/ref query strings.
    2. Content hash (SHA-256 of cleaned text): catches republications under
       a different URL — rare but real (CMS migrations, syndication).

In both cases, the FIRST occurrence wins. This is intentional: we trust the
earliest fetch and ignore later duplicates.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tnra.ingestion.schemas import CleanedArticle, RawArticle
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

# Query-string parameters considered tracking metadata and stripped from URLs
# before deduplication. Lowercased, prefix-matched (so "utm_anything" is caught).
_TRACKING_PARAM_PREFIXES: tuple[str, ...] = (
    "utm_",  # Google Analytics (utm_source, utm_medium, utm_campaign, ...)
    "fbclid",  # Facebook click ID
    "gclid",  # Google ads click ID
    "mc_",  # Mailchimp (mc_cid, mc_eid)
    "ref",  # generic referrer tag (also catches "ref_src")
    "ref_",
    "src",
    "_hsenc",  # HubSpot
    "_hsmi",  # HubSpot
)

# Whitespace normalization pattern: collapse 3+ consecutive newlines into 2
# (preserves paragraph breaks, removes only excessive blank lines).
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Collapse runs of spaces/tabs (but NOT newlines — paragraph structure matters
# for the chunker, which uses "\n\n" as its top-priority separator).
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


# -----------------------------------------------------------------------------
# URL canonicalization
# -----------------------------------------------------------------------------


def canonicalize_url(url: str) -> str:
    """Normalize an article URL for deduplication.

    Operations:
      - Lowercase the scheme and host
      - Drop URL fragment (#section)
      - Drop tracking query params (utm_*, fbclid, gclid, etc.)
      - Drop trailing slash from path

    Example:
            https://TechCrunch.com/article/?utm_source=rss#comments
          → https://techcrunch.com/article

    Args:
        url: Any valid URL string.

    Returns:
        Canonical form, suitable as a deduplication key.
    """
    parts = urlsplit(url)

    # Scheme + host lowercased; path preserved (case can be meaningful in URLs)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")

    # Filter query string: keep only non-tracking params
    kept_params = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(k)
    ]
    query = urlencode(kept_params)

    # Drop fragment entirely (it's never relevant for content identity)
    return urlunsplit((scheme, netloc, path, query, ""))


def _is_tracking_param(key: str) -> bool:
    """True if a query param key matches any tracking prefix."""
    k = key.lower()
    return any(k.startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES)


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Clean residual whitespace and Unicode noise from extracted article text.

    Trafilatura already produces clean text — this is a safety net for edge
    cases (PDF-converted articles, copy-pasted layouts) that may still have:
      - Non-breaking spaces (U+00A0) and other Unicode whitespace
      - Runs of 3+ consecutive newlines (visual padding from source HTML)
      - Multiple tabs/spaces within a line

    We deliberately preserve paragraph breaks ("\\n\\n") because the chunker
    uses them as its top-priority split boundary.
    """
    # Replace common non-standard whitespace with regular space.
    # U+00A0 = NBSP, U+200B = zero-width space, U+FEFF = BOM.
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # Collapse excessive blank lines (3+) into exactly 2 (preserves paragraph break)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # Collapse runs of spaces/tabs into a single space
    text = _MULTI_SPACE.sub(" ", text)

    # Trim leading/trailing whitespace globally
    return text.strip()


# -----------------------------------------------------------------------------
# Cleaning + deduplication
# -----------------------------------------------------------------------------


def clean_article(article: RawArticle) -> CleanedArticle:
    """Convert one RawArticle into a CleanedArticle.

    Computes the content hash from normalized text — i.e. two articles with
    minor whitespace differences but identical actual content will hash equal
    and be deduplicated downstream.
    """
    cleaned_content = normalize_text(article.content_html)
    canonical_url = canonicalize_url(str(article.url))
    content_hash = CleanedArticle.compute_hash(cleaned_content)

    return CleanedArticle(
        url=canonical_url,  # type: ignore[arg-type]
        title=article.title,
        content=cleaned_content,
        content_hash=content_hash,
        source=article.source,
        feed_name=article.feed_name,
        published_at=article.published_at,
        fetched_at=article.fetched_at,
        author=article.author,
    )


def clean_and_deduplicate(articles: list[RawArticle]) -> list[CleanedArticle]:
    """Clean a batch of articles and remove duplicates (URL + content hash).

    Two-level deduplication strategy:
      1. Canonical URL match (after stripping tracking params)
      2. Content hash match (catches republications under a different URL)

    First-seen wins in both cases. We log how many duplicates were dropped
    and why, which is useful when debugging feed overlap (e.g. an article
    that appears in both techcrunch_main and techcrunch_ai).

    Args:
        articles: List of scraped RawArticle objects.

    Returns:
        List of unique CleanedArticle objects, in input order.
    """
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    cleaned: list[CleanedArticle] = []
    dropped_url = 0
    dropped_hash = 0

    for article in articles:
        cleaned_article = clean_article(article)

        if cleaned_article.url in seen_urls:  # type: ignore[operator]
            dropped_url += 1
            continue
        if cleaned_article.content_hash in seen_hashes:
            dropped_hash += 1
            continue

        seen_urls.add(str(cleaned_article.url))
        seen_hashes.add(cleaned_article.content_hash)
        cleaned.append(cleaned_article)

    logger.info(
        "Cleaning done: %d → %d articles (dropped %d duplicate URLs, %d duplicate contents)",
        len(articles),
        len(cleaned),
        dropped_url,
        dropped_hash,
    )
    return cleaned
