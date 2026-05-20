"""Article chunking into retrievable passages.

Splits cleaned articles into overlapping passages sized to fit within the
embedding model's context window (512 tokens for BGE-large-en-v1.5).

Uses LangChain's `RecursiveCharacterTextSplitter`, which respects natural text
boundaries: it tries paragraphs first ("\\n\\n"), then lines, then sentences,
then words — going recursively to coarser splits only when needed. This
preserves semantic coherence within each chunk, which is critical for
retrieval quality.

Token counting via `tiktoken` (OpenAI's cl100k_base). It's not the exact
tokenizer used by BGE, but matches within ~5% and is the industry standard
for chunk-size estimation across embedding models.
"""

from __future__ import annotations

from hashlib import sha256

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from tnra.ingestion.schemas import Chunk, CleanedArticle
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Config schema (Pydantic validation of the YAML sub-section)
# -----------------------------------------------------------------------------


class ChunkingConfig(BaseModel):
    """Validated chunking config.

    Built from the `chunking` section of `configs/ingestion.yaml`. Validates
    bounds at module entry, so misconfigurations fail loudly with a clear
    Pydantic error rather than producing silently-broken chunks downstream.
    """

    splitter: str = Field(default="recursive_character")
    chunk_size: int = Field(gt=0, le=2048)  # in tokens
    chunk_overlap: int = Field(ge=0)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


# -----------------------------------------------------------------------------
# Splitter construction
# -----------------------------------------------------------------------------


def build_splitter(cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Construct a token-aware RecursiveCharacterTextSplitter from validated config.

    `from_tiktoken_encoder` configures the splitter to count chunk size in
    *tokens* (via tiktoken cl100k_base) rather than characters. Without this,
    chunk_size=512 would mean 512 characters — way under our actual target.
    """
    if cfg.splitter != "recursive_character":
        raise ValueError(
            f"Unsupported splitter: {cfg.splitter!r}. Only 'recursive_character' is implemented."
        )

    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError(
            f"chunk_overlap ({cfg.chunk_overlap}) must be smaller than "
            f"chunk_size ({cfg.chunk_size})."
        )

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )


# -----------------------------------------------------------------------------
# Chunking pipeline
# -----------------------------------------------------------------------------


def _make_chunk_id(article_url: str, chunk_index: int) -> str:
    """Build a stable, content-independent chunk ID.

    Format: <url_sha256_prefix>_<index>, e.g. "a3f9c1e2b4d5e6f7_0".

    The same article re-ingested on a future run will produce the same chunk
    IDs. ChromaDB uses these as primary keys, so this guarantees idempotent
    upserts (re-running ingestion doesn't duplicate entries).
    """
    url_hash = sha256(article_url.encode("utf-8")).hexdigest()[:16]
    return f"{url_hash}_{chunk_index}"


def chunk_article(article: CleanedArticle, splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Split one cleaned article into a list of validated Chunk objects.

    Each chunk carries enough metadata (title, url, source, position) to be
    cited back to its origin in the final LLM response.
    """
    texts = splitter.split_text(article.content)
    url_str = str(article.url)

    chunks: list[Chunk] = []
    for idx, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(url_str, idx),
                article_url=article.url,
                article_title=article.title,
                text=text,
                chunk_index=idx,
                source=article.source,
                feed_name=article.feed_name,
                published_at=article.published_at,
                fetched_at=article.fetched_at,
            )
        )
    return chunks


def chunk_articles(articles: list[CleanedArticle], cfg: ChunkingConfig) -> list[Chunk]:
    """Chunk a batch of articles into a single flat list of passages.

    Logs aggregate stats to help spot pathological cases:
      - very short articles producing 1 chunk (could indicate scraping failure)
      - very long articles producing many chunks (could indicate boilerplate
        leakage from trafilatura)
    """
    splitter = build_splitter(cfg)

    all_chunks: list[Chunk] = []
    for article in articles:
        article_chunks = chunk_article(article, splitter)
        all_chunks.extend(article_chunks)

    if not articles:
        logger.warning("chunk_articles called with empty article list")
        return all_chunks

    chunks_per_article = [
        sum(1 for c in all_chunks if str(c.article_url) == str(a.url)) for a in articles
    ]
    logger.info(
        "Chunking done: %d articles → %d chunks (avg %.1f, min %d, max %d per article)",
        len(articles),
        len(all_chunks),
        sum(chunks_per_article) / len(chunks_per_article),
        min(chunks_per_article),
        max(chunks_per_article),
    )
    return all_chunks
