"""Persistent vector indexing with ChromaDB.

Stores chunk embeddings + metadata in a local Chroma collection that persists
across runs (under `chroma_db/`). The indexing is idempotent: re-running on
the same chunks updates rather than duplicates, thanks to deterministic
chunk_id values produced by chunking.py.

ChromaDB internals (high level):
    - Vectors: HNSW index (hierarchical k-NN), in-memory + persisted to disk
    - Metadata: SQLite under chroma_db/chroma.sqlite3
    - Filtering during search supported via metadata predicates
    - Distance: cosine (configurable at collection creation)
"""

from __future__ import annotations

from typing import Literal

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from pydantic import BaseModel, Field

from tnra.ingestion.schemas import Chunk
from tnra.utils.logger import get_logger
from tnra.utils.paths import CHROMA_DIR, ensure_dir

logger = get_logger(__name__)

DistanceChoice = Literal["cosine", "l2", "ip"]


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------


class IndexConfig(BaseModel):
    """Validated index config.

    Built from the `index` section of `configs/ingestion.yaml`.
    """

    backend: Literal["chromadb"] = "chromadb"
    collection_name: str = Field(min_length=1)
    distance: DistanceChoice = "cosine"


# -----------------------------------------------------------------------------
# Chroma client and collection
# -----------------------------------------------------------------------------


def get_chroma_client() -> ClientAPI:
    """Build a persistent Chroma client pointing at `chroma_db/`.

    The directory is created on first call. Subsequent calls reuse the same
    on-disk store, so previously indexed chunks survive across runs.
    """
    persist_dir = ensure_dir(CHROMA_DIR)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),  # disable Chroma's telemetry ping
    )


def get_or_create_collection(client: ClientAPI, cfg: IndexConfig) -> Collection:
    """Open the Chroma collection, creating it if absent.

    The distance metric is set at creation time and CANNOT be changed later
    without rebuilding the index. We assert it matches the config on every
    open, so a mismatched run fails loudly rather than silently using the
    wrong metric.
    """
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": cfg.distance},  # "cosine", "l2", or "ip"
    )

    existing_space = collection.metadata.get("hnsw:space") if collection.metadata else None
    if existing_space and existing_space != cfg.distance:
        raise ValueError(
            f"Collection '{cfg.collection_name}' was created with distance="
            f"{existing_space!r} but config requests {cfg.distance!r}. "
            f"Delete chroma_db/ to rebuild, or change the config back."
        )

    logger.info(
        "Chroma collection ready: name=%s, distance=%s, current_count=%d",
        cfg.collection_name,
        cfg.distance,
        collection.count(),
    )
    return collection


# -----------------------------------------------------------------------------
# Upsert
# -----------------------------------------------------------------------------


def index_chunks(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    collection: Collection,
    *,
    upsert_batch_size: int = 256,
) -> None:
    """Insert (or update) chunks + embeddings into the Chroma collection.

    Uses `upsert` (not `add`) so re-ingesting the same chunks updates them
    in place rather than raising on duplicate IDs. Idempotency relies on
    chunking.py producing stable chunk_id values from (article_url, index).

    Args:
        chunks: Validated chunks. Must align 1:1 with rows of `embeddings`.
        embeddings: shape (N, D), float32, L2-normalized if using cosine.
        collection: Open Chroma collection.
        upsert_batch_size: Chunks sent per upsert call. Chroma is fine with
            large batches, but we cap to give nicer log progress and avoid
            ballooning memory on very large ingestions.

    Raises:
        ValueError: if lengths don't match or embeddings shape is wrong.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
        )
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D), got shape {embeddings.shape}")
    if not chunks:
        logger.warning("index_chunks called with empty input — nothing to do")
        return

    total = len(chunks)
    logger.info("Indexing %d chunks into collection '%s'...", total, collection.name)

    for start in range(0, total, upsert_batch_size):
        end = min(start + upsert_batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeds = embeddings[start:end]

        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeds.tolist(),  # Chroma expects list[list[float]]
            documents=[c.text for c in batch_chunks],  # raw text, used for BM25 + LLM context
            metadatas=[c.to_chroma_metadata() for c in batch_chunks],  # type: ignore
        )

        logger.info("Upsert progress: %d/%d", end, total)

    logger.info("Indexing done. Collection now has %d chunks total.", collection.count())
