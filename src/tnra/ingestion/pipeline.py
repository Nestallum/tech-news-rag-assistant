"""Ingestion pipeline orchestration.

Wires together the ingestion stages built in the sibling modules:

    feeds ──► scraping ──► cleaning ──► chunking ──► embedding ──► indexing

Two public entry points:
    - run_ingestion()  : full pipeline, RSS feeds → ChromaDB index
    - validate_feeds() : dry-run that only checks scrape success per feed

Both are called by `scripts/ingest.py`. Keeping the logic here (rather than
in the script) makes it importable by tests and by the future scheduled
ingestion job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from omegaconf import DictConfig

from tnra.ingestion.chunking import ChunkingConfig, chunk_articles
from tnra.ingestion.cleaning import clean_and_deduplicate
from tnra.ingestion.embedding import Embedder, EmbeddingConfig
from tnra.ingestion.feeds import fetch_feed
from tnra.ingestion.indexing import (
    IndexConfig,
    get_chroma_client,
    get_or_create_collection,
    index_chunks,
)
from tnra.ingestion.schemas import FeedEntry, FeedValidationResult, RawArticle, RetentionConfig
from tnra.ingestion.scraping import scrape_articles
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Result summary
# -----------------------------------------------------------------------------


@dataclass
class IngestionReport:
    """Summary of an ingestion run, returned to the caller for logging/printing."""

    feeds_processed: int = 0
    entries_found: int = 0
    articles_scraped: int = 0
    articles_after_dedup: int = 0
    chunks_indexed: int = 0
    collection_total: int = 0
    per_feed_entries: dict[str, int] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Stage 1+2: fetch RSS + scrape HTML, feed by feed
# -----------------------------------------------------------------------------


def _collect_articles(
    cfg: DictConfig, *, max_articles: int | None
) -> tuple[list[RawArticle], dict[str, int]]:
    """Fetch every configured feed and scrape its articles.

    Returns the flat list of successfully scraped articles plus a per-feed
    count of RSS entries (useful for the run report and for spotting a feed
    that suddenly went silent).

    When `max_articles` is set (dev/test runs), the cap is spread *evenly*
    across feeds rather than truncating the flat list — otherwise we'd only
    ever sample the first feed. Each feed contributes at most
    `ceil(max_articles / n_feeds)` entries.
    """

    user_agent = cfg.fetch.user_agent
    timeout_s = cfg.fetch.timeout_s
    n_feeds = len(cfg.feeds)

    # Per-feed budget when a global cap is requested (spread evenly).
    per_feed_budget: int | None = None
    if max_articles is not None and n_feeds > 0:
        per_feed_budget = math.ceil(max_articles / n_feeds)

    all_entries: list[FeedEntry] = []
    per_feed: dict[str, int] = {}

    for feed in cfg.feeds:
        try:
            entries = fetch_feed(
                feed.url,
                feed_name=feed.name,
                source=feed.source,
                user_agent=user_agent,
                timeout_s=timeout_s,
            )
        except Exception as e:
            # A single broken feed must not abort the whole ingestion.
            logger.error("Feed %s failed entirely: %s", feed.name, e)
            per_feed[feed.name] = 0
            continue

        # Per-feed cap from config (e.g. 9to5Mac publishes 100 entries —
        # avoid Apple over-representation in the corpus).
        max_entries = feed.get("max_entries", None)
        if max_entries is not None:
            entries = entries[:max_entries]

        # Dev/test global cap, spread evenly across feeds.
        if per_feed_budget is not None:
            entries = entries[:per_feed_budget]

        per_feed[feed.name] = len(entries)
        all_entries.extend(entries)

    # Hard trim in case rounding pushed us slightly over max_articles.
    if max_articles is not None:
        all_entries = all_entries[:max_articles]

    logger.info("Collected %d entries across %d feeds — scraping...", len(all_entries), n_feeds)
    articles = scrape_articles(all_entries, user_agent=user_agent, timeout_s=timeout_s)
    return articles, per_feed


# -----------------------------------------------------------------------------
# Stage 6.5: purge expired chunks (retention policy)
# -----------------------------------------------------------------------------


def _purge_expired_chunks(collection, retention_cfg: RetentionConfig) -> int:
    """Delete chunks older than the retention window from the collection.

    Filters on `published_at`, which is guaranteed present and stored as a
    Unix timestamp in the chunk metadata. Returns the number of chunks
    deleted, for logging.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=retention_cfg.days)
    cutoff_ts = int(cutoff.timestamp())

    before = collection.count()
    collection.delete(where={"published_at": {"$lt": cutoff_ts}})
    after = collection.count()
    deleted = before - after

    if deleted > 0:
        logger.info(
            "Retention: purged %d chunks older than %d days (cutoff=%s)",
            deleted,
            retention_cfg.days,
            cutoff.isoformat(),
        )
    else:
        logger.info("Retention: no chunks older than %d days to purge", retention_cfg.days)

    return deleted


# -----------------------------------------------------------------------------
# Public: full ingestion
# -----------------------------------------------------------------------------


def run_ingestion(cfg: DictConfig, *, max_articles: int | None = None) -> IngestionReport:
    """Run the full ingestion pipeline: RSS feeds → ChromaDB index.

    Args:
        cfg: Merged ingestion config (base.yaml + ingestion.yaml + overrides).
        max_articles: Optional global cap on articles, for fast dev runs.

    Returns:
        An IngestionReport summarizing what happened.
    """
    report = IngestionReport(feeds_processed=len(cfg.feeds))

    # --- Stage 1+2: fetch + scrape ---
    articles, per_feed = _collect_articles(cfg, max_articles=max_articles)
    report.per_feed_entries = per_feed
    report.entries_found = sum(per_feed.values())
    report.articles_scraped = len(articles)

    if not articles:
        logger.warning("No articles scraped — nothing to index. Stopping.")
        return report

    # --- Stage 3: clean + deduplicate ---
    cleaned = clean_and_deduplicate(articles)
    report.articles_after_dedup = len(cleaned)

    # --- Stage 4: chunk ---
    chunking_cfg = ChunkingConfig(**cfg.chunking)
    chunks = chunk_articles(cleaned, chunking_cfg)
    if not chunks:
        logger.warning("No chunks produced — nothing to index. Stopping.")
        return report

    # --- Stage 5: embed ---
    embedding_cfg = EmbeddingConfig(**cfg.embeddings)
    embedder = Embedder(embedding_cfg)
    embeddings = embedder.embed_chunks(chunks)

    # --- Stage 6: index ---
    index_cfg = IndexConfig(**cfg.index)
    client = get_chroma_client()
    collection = get_or_create_collection(client, index_cfg)

    # Apply the retention policy before adding new chunks.
    retention_cfg = RetentionConfig(**cfg.retention)
    _purge_expired_chunks(collection, retention_cfg)

    index_chunks(chunks, embeddings, collection)

    report.chunks_indexed = len(chunks)
    report.collection_total = collection.count()
    return report


# -----------------------------------------------------------------------------
# Public: feed validation (dry-run, no indexing)
# -----------------------------------------------------------------------------


def validate_feeds(cfg: DictConfig, *, sample_size: int = 5) -> list[FeedValidationResult]:
    """Dry-run: check scrape success + content length for each configured feed.

    Does NOT clean, chunk, embed, or index. Use this to (re)assess whether a
    feed still delivers full-text content before committing it to the corpus.

    Args:
        cfg: Merged ingestion config.
        sample_size: Number of articles to scrape per feed for the check.

    Returns:
        One FeedValidationResult per feed.
    """
    user_agent = cfg.fetch.user_agent
    timeout_s = cfg.fetch.timeout_s
    min_chars = cfg.fetch.validate_min_content_chars

    results: list[FeedValidationResult] = []

    for feed in cfg.feeds:
        try:
            entries = fetch_feed(
                feed.url,
                feed_name=feed.name,
                source=feed.source,
                user_agent=user_agent,
                timeout_s=timeout_s,
            )
        except Exception as e:
            results.append(
                FeedValidationResult(
                    feed_name=feed.name,
                    url=feed.url,
                    status="error",
                    sample_size=0,
                    avg_content_chars=0,
                    error=str(e),
                )
            )
            continue

        sample = entries[:sample_size]
        articles = scrape_articles(sample, user_agent=user_agent, timeout_s=timeout_s)

        if not articles:
            status: str = "error"
            avg_chars = 0
        else:
            lengths = [len(a.content_html) for a in articles]
            avg_chars = sum(lengths) // len(lengths)
            success_rate = len(articles) / len(sample)
            if success_rate >= 0.8 and avg_chars >= min_chars:
                status = "full_text"
            elif avg_chars >= min_chars // 2:
                status = "partial"
            else:
                status = "excerpt"

        results.append(
            FeedValidationResult(
                feed_name=feed.name,
                url=feed.url,
                status=status,  # type: ignore[arg-type]
                sample_size=len(sample),
                avg_content_chars=avg_chars,
            )
        )
        logger.info(
            "Feed %s: status=%s, avg_chars=%d (%d/%d scraped)",
            feed.name,
            status,
            avg_chars,
            len(articles),
            len(sample),
        )

    return results
