"""Persistent vector indexing with Qdrant Cloud.

Stores chunk embeddings + metadata in a Qdrant collection. Idempotent:
re-running on the same chunks updates rather than duplicates, thanks to
deterministic chunk_id values produced by chunking.py.
"""

from __future__ import annotations

import os
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from tnra.ingestion.schemas import Chunk
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------


class IndexConfig(BaseModel):
    """Validated index config.

    Built from the `index` section of `configs/ingestion.yaml`.
    """

    backend: Literal["qdrant"] = "qdrant"
    collection_name: str = Field(min_length=1)


# -----------------------------------------------------------------------------
# Qdrant client and collection
# -----------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """Build a Qdrant Cloud client from environment variables."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def ensure_collection(client: QdrantClient, cfg: IndexConfig) -> None:
    """Create the Qdrant collection if it doesn't exist yet.

    Vector name 'dense' anticipates future hybrid search (sparse + dense).
    Dimension 1024 matches BAAI/bge-large-en-v1.5.
    """
    existing = {c.name for c in client.get_collections().collections}
    if cfg.collection_name not in existing:
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        client.create_payload_index(
            collection_name=cfg.collection_name,
            field_name="published_at",
            field_schema=PayloadSchemaType.INTEGER,
        )
        logger.info("Created Qdrant collection: %s", cfg.collection_name)
        logger.info("Created payload index on 'published_at'")
    else:
        logger.info("Qdrant collection already exists: %s", cfg.collection_name)


# -----------------------------------------------------------------------------
# Upsert
# -----------------------------------------------------------------------------


def index_chunks(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    client: QdrantClient,
    collection_name: str,
    *,
    upsert_batch_size: int = 256,
) -> None:
    """Insert (or update) chunks + embeddings into the Qdrant collection.

    Uses `upsert` (not `add`) so re-ingesting the same chunks updates them
    in place rather than raising on duplicate IDs. Idempotency relies on
    chunking.py producing stable chunk_id values from (article_url, index).

    Args:
        chunks: Validated chunks. Must align 1:1 with rows of `embeddings`.
        embeddings: shape (N, D), float32, L2-normalized if using cosine.
        client: Qdrant client.
        collection_name: Target collection name.
        upsert_batch_size: Chunks sent per upsert call. Capped to give nicer
            log progress and avoid ballooning memory on very large ingestions.

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
    logger.info("Indexing %d chunks into collection '%s'...", total, collection_name)

    for start in range(0, total, upsert_batch_size):
        end = min(start + upsert_batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeds = embeddings[start:end]

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector={"dense": batch_embeds[i].tolist()},
                payload=c.to_metadata(),
            )
            for i, c in enumerate(batch_chunks)
        ]
        client.upsert(collection_name=collection_name, points=points)

        logger.info("Upsert progress: %d/%d", end, total)
