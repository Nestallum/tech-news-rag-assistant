"""Structured logging with console (rich) and optional file output.

Usage:
    from tnra.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Fetched %d entries", count)

Loggers are cached by name: calling get_logger("foo") twice returns the same
logger, so configuration is applied only once.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

from tnra.utils.paths import LOGS_DIR, ensure_dir

_DEFAULT_LEVEL = logging.INFO
_DEFAULT_FORMAT = "%(name)s | %(message)s"  # rich handles time + level itself
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "tnra",
    level: int | str = _DEFAULT_LEVEL,
    log_file: Path | None = None,
) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name. Use `__name__` from the caller for hierarchical naming
            (e.g. `tnra.ingestion.feeds` will inherit config from `tnra`).
        level: Log level (logging.INFO, "DEBUG", etc.).
        log_file: Optional file path to also write logs to. If None, console only.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Idempotency: if this logger is already configured, return as-is.
    # Without this guard, calling get_logger() multiple times would stack handlers
    # and each log line would print N times.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs via the root logger

    # --- Console handler (rich: colored, formatted, with timestamp + level) ---
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        log_time_format="[%X]",
    )
    console_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(console_handler)

    # --- File handler (plain text, full format with date) ---
    if log_file is not None:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)

    return logger


def get_run_log_file(stage: str) -> Path:
    """Build a timestamped log file path for a pipeline run.

    Args:
        stage: Pipeline stage name (e.g. "ingest", "evaluate", "app").

    Returns:
        Path like `logs/ingest_20260518_142301.log`. The parent dir is NOT created
        here — pass this path to `get_logger(log_file=...)` which will create it.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{stage}_{timestamp}.log"
