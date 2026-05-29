"""Pydantic schemas for the ingestion pipeline.

Data flows through four stages, each with its own schema:

    RSS feed entry ──► FeedEntry ──► RawArticle ──► CleanedArticle ──► Chunk(s) ──► ChromaDB
                       (parsing)    (HTML scraping)   (cleaning)      (chunking)

Each schema is the contract between two adjacent stages. Downstream modules
import the type they consume, validate at the boundary, and produce the next
type. This makes the pipeline self-documenting and catches malformed data
at the earliest possible point.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class FeedEntry(BaseModel):
    """A lightweight entry parsed from an RSS feed, BEFORE full-text scraping.

    RSS feeds only carry metadata + short excerpts in 2026 — the real article
    body lives on the publisher's HTML page. This schema captures the metadata
    we need to (a) deduplicate before scraping, (b) preserve published date
    even if the HTML page omits it, (c) fetch the right URL.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    url: HttpUrl  # canonical article URL (will be scraped)
    title: str = Field(min_length=1)
    summary: str = ""  # short excerpt from RSS, kept for fallback
    source: str = Field(min_length=1)
    feed_name: str = Field(min_length=1)
    published_at: datetime | None = None
    fetched_at: datetime
    author: str | None = None


class RawArticle(BaseModel):
    """An article as it comes out of the RSS parser, before cleaning.

    At this stage we trust the feed minimally: we keep the raw HTML content
    and only check that the essential fields (url, title, content) are present
    and non-empty. Date parsing is permissive — some feeds use non-standard
    formats and we fall back to `fetched_at` if parsing fails upstream.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    url: HttpUrl  # canonical article URL (deduplication key)
    title: str = Field(min_length=1)
    content_html: str = Field(min_length=1)  # raw HTML from <content:encoded> or <description>
    source: str = Field(min_length=1)  # human-readable feed source, e.g. "TechCrunch"
    feed_name: str = Field(min_length=1)  # internal feed identifier, e.g. "techcrunch_ai"
    published_at: datetime | None = None  # may be missing on some feeds
    fetched_at: datetime  # when WE retrieved it (always set)
    author: str | None = None


class CleanedArticle(BaseModel):
    """An article after HTML stripping, whitespace normalization, and deduplication.

    `content` is plain text ready to be chunked. `content_hash` is computed once
    here and used as a secondary deduplication key (same URL with edited content
    is treated as the same article — we keep the first version).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    url: HttpUrl
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)  # cleaned plain text
    content_hash: str = Field(min_length=64, max_length=64)  # SHA-256 hex digest
    source: str = Field(min_length=1)
    feed_name: str = Field(min_length=1)
    published_at: datetime | None = None
    fetched_at: datetime
    author: str | None = None

    @field_validator("content_hash")
    @classmethod
    def _hash_is_hex(cls, v: str) -> str:
        """Reject anything that isn't a valid lowercase hex SHA-256 digest."""
        try:
            int(v, 16)
        except ValueError as e:
            raise ValueError("content_hash must be a hex SHA-256 digest") from e
        return v.lower()

    @staticmethod
    def compute_hash(content: str) -> str:
        """SHA-256 of the cleaned content, lowercase hex. Use this to fill content_hash."""
        return sha256(content.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """A passage produced by the chunker, ready to be embedded and indexed.

    Each chunk carries enough metadata to (a) be retrieved with full context,
    (b) be cited back to its source article in the LLM response, and
    (c) support per-source / per-date filtering in the vector store.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)  # stable unique id, e.g. "<url_hash>_<idx>"
    article_url: HttpUrl  # back-pointer to source article
    article_title: str = Field(min_length=1)
    text: str = Field(min_length=1)  # the actual passage content
    chunk_index: int = Field(ge=0)  # position of this chunk inside its article
    source: str = Field(min_length=1)
    feed_name: str = Field(min_length=1)
    published_at: datetime | None = None
    fetched_at: datetime

    def to_chroma_metadata(self) -> dict[str, str | int]:
        """Flatten to a Chroma-compatible metadata dict.

        ChromaDB metadata values must be primitives (str, int, float, bool, None).
        Dates are stored as Unix timestamps (seconds) so that downstream
        filters can use Chroma's numeric range operators. When the feed
        didn't provide a publication date, we fall back to `fetched_at` so
        `published_at` is always present.
        """
        published = self.published_at if self.published_at is not None else self.fetched_at
        return {
            "article_url": str(self.article_url),
            "article_title": self.article_title,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "feed_name": self.feed_name,
            "fetched_at": int(self.fetched_at.timestamp()),
            "published_at": int(published.timestamp()),
        }


# -----------------------------------------------------------------------------
# Feed validation report (used by ingest.py --validate-feeds)
# -----------------------------------------------------------------------------

FeedStatus = Literal["full_text", "partial", "excerpt", "error"]


class FeedValidationResult(BaseModel):
    """Output of empirical full-text validation for a single feed.

    Used by `ingest.py --validate-feeds` to decide whether to keep a candidate
    feed (Engadget, 9to5Mac) in the corpus or drop it for being excerpt-only.
    """

    feed_name: str
    url: str
    status: FeedStatus
    sample_size: int  # how many entries were checked
    avg_content_chars: int  # average plain-text length across sample
    error: str | None = None  # filled when status == "error"


# -----------------------------------------------------------------------------
# Retention policy (used by the indexing stage to purge expired chunks)
# -----------------------------------------------------------------------------


class RetentionConfig(BaseModel):
    """Configuration for the corpus retention policy.

    The pipeline purges chunks older than `days` before each new indexing
    pass, so the corpus stays a rolling window of recent tech news rather
    than growing indefinitely.
    """

    model_config = ConfigDict(frozen=True)

    days: int = Field(gt=0, description="Maximum chunk age in days")
