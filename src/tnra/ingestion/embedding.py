"""Chunk embedding with sentence-transformers (BGE-large-en-v1.5).

Transforms each Chunk's text into a 1024-dimensional vector suitable for
cosine-similarity retrieval in ChromaDB.

Why sentence-transformers directly (rather than via langchain-huggingface):
    - Direct access to batch_size, device, show_progress_bar
    - Single dependency, less abstraction noise when debugging
    - We still use the LangChain wrapper later in indexing.py to plug into
      the Chroma vector store cleanly. Best of both worlds.

Why BGE-large-en-v1.5:
    - Top-tier on MTEB benchmarks, well-established baseline
    - 1024-dim, 512-token context — matches our chunking
    - Public, MIT license, no API key required
    - Light enough to run on CPU if no GPU is available
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from tnra.ingestion.schemas import Chunk
from tnra.utils.logger import get_logger
from tnra.utils.paths import HF_CACHE_DIR, ensure_dir

logger = get_logger(__name__)

DeviceChoice = Literal["auto", "cuda", "mps", "cpu"]


# -----------------------------------------------------------------------------
# Config schema (Pydantic validation of the YAML sub-section)
# -----------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """Validated embedding config.

    Built from the `embeddings` section of `configs/ingestion.yaml`.
    """

    model: str = Field(min_length=1)
    device: DeviceChoice = "auto"
    batch_size: int = Field(gt=0, le=512)
    normalize: bool = True
    query_prefix: str = ""  # BGE v1.5 doesn't need one; v1.0 did ("Represent this...")


# -----------------------------------------------------------------------------
# Device resolution
# -----------------------------------------------------------------------------


def _resolve_device(choice: DeviceChoice) -> str:
    """Pick a concrete torch device string from the config preference.

    Priority for "auto": cuda → mps (Apple Silicon) → cpu.
    Explicit choices are honored as-is (no fallback) so a misconfigured GPU
    fails loudly instead of silently embedding on CPU at 1/10th the speed.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return choice


# -----------------------------------------------------------------------------
# Embedder
# -----------------------------------------------------------------------------


class Embedder:
    """Wraps a SentenceTransformer model with batched encoding.

    The model is loaded once at construction and reused across calls — loading
    BGE-large takes ~3s on warm cache, 2-5 min on first download. Keep the
    Embedder alive for the whole pipeline run.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        cache_dir = ensure_dir(HF_CACHE_DIR)

        logger.info(
            "Loading embedding model: %s on device=%s (cache=%s)",
            cfg.model,
            self.device,
            cache_dir,
        )
        self.model = SentenceTransformer(
            cfg.model,
            device=self.device,
            cache_folder=str(cache_dir),
        )
        # Embedding dimension is fixed by the model — expose it for downstream
        # consumers (Chroma collection schema, sanity checks).
        self.dim: int = self.model.get_embedding_dimension()  # type: ignore[assignment]
        logger.info("Model loaded: dim=%d, max_seq_length=%d", self.dim, self.model.max_seq_length)

    def embed_texts(self, texts: list[str], *, show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into a (N, dim) float32 numpy array.

        Args:
            texts: Raw strings to embed.
            show_progress: Display tqdm progress bar (set False in tight loops).

        Returns:
            np.ndarray of shape (len(texts), self.dim), L2-normalized iff
            cfg.normalize is True.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        # sentence-transformers returns float32 by default; assert to catch surprises.
        assert embeddings.dtype == np.float32, f"Unexpected dtype: {embeddings.dtype}"
        return embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Encode a list of Chunks (uses chunk.text as the input)."""
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single user query into a (dim,) vector.

        Applies `query_prefix` if set in config (e.g. "Represent this sentence
        for searching relevant passages: " for older BGE v1.0). BGE v1.5
        doesn't need a prefix.
        """
        text = f"{self.cfg.query_prefix}{query}" if self.cfg.query_prefix else query
        vec = self.embed_texts([text], show_progress=False)
        return vec[0]
