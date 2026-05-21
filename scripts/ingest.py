"""Ingestion entry point.

Builds (or refreshes) the ChromaDB index from the configured RSS feeds.

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --config configs/ingestion.yaml
    uv run python scripts/ingest.py --max-articles 12          # fast dev run
    uv run python scripts/ingest.py --override chunking.chunk_size=256
    uv run python scripts/ingest.py --validate-feeds           # dry-run, no indexing

This script is intentionally thin: it parses CLI args, sets up logging and
environment, then delegates to `tnra.ingestion.pipeline`. The real logic lives
there so it stays importable by tests and the scheduled ingestion job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make `tnra` importable when this script is run directly (scripts/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnra.ingestion.pipeline import run_ingestion, validate_feeds
from tnra.utils.config import load_config, save_config
from tnra.utils.logger import get_logger, get_run_log_file
from tnra.utils.paths import LOGS_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest RSS feeds into the ChromaDB vector index.")
    parser.add_argument(
        "--config",
        type=str,
        default="ingestion.yaml",
        help="Stage config file (relative to configs/ or an explicit path).",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, e.g. --override chunking.chunk_size=256",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Global cap on articles (spread evenly across feeds). For fast dev runs.",
    )
    parser.add_argument(
        "--validate-feeds",
        action="store_true",
        help="Dry-run: only check scrape success per feed, do not index anything.",
    )
    return parser.parse_args()


def main() -> int:
    """Script entry point. Returns a process exit code (0 = success)."""
    args = parse_args()

    # Load environment variables (HF_TOKEN, etc.) from a local .env if present.
    load_dotenv()

    # Set up run logging: console (rich) + a timestamped file under logs/.
    stage = "validate" if args.validate_feeds else "ingest"
    log_file = get_run_log_file(stage)
    logger = get_logger("tnra", log_file=log_file)
    logger.info("Run log: %s", log_file)

    # Load + merge config (base.yaml + stage config + CLI overrides).
    cfg = load_config(args.config, overrides=args.override)

    # --- Validation mode: dry-run, no indexing ---
    if args.validate_feeds:
        logger.info("Running feed validation (dry-run)...")
        results = validate_feeds(cfg)
        logger.info("=" * 60)
        logger.info("FEED VALIDATION SUMMARY")
        logger.info("=" * 60)
        for r in results:
            logger.info(
                "  %-18s %-12s avg=%6d chars  %s",
                r.feed_name,
                r.status,
                r.avg_content_chars,
                f"(error: {r.error})" if r.error else "",
            )
        return 0

    # --- Full ingestion mode ---
    # Snapshot the resolved config for reproducibility (which config built this index?).
    snapshot_dir = LOGS_DIR / log_file.stem
    config_path = save_config(cfg, snapshot_dir)
    logger.info("Config snapshot saved: %s", config_path)

    logger.info("Starting ingestion...")
    report = run_ingestion(cfg, max_articles=args.max_articles)

    # Final summary
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Feeds processed     : %d", report.feeds_processed)
    logger.info("  RSS entries found   : %d", report.entries_found)
    logger.info("  Articles scraped    : %d", report.articles_scraped)
    logger.info("  After deduplication : %d", report.articles_after_dedup)
    logger.info("  Chunks indexed      : %d", report.chunks_indexed)
    logger.info("  Collection total    : %d", report.collection_total)
    for feed_name, count in report.per_feed_entries.items():
        logger.info("    - %-18s %d entries", feed_name, count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
