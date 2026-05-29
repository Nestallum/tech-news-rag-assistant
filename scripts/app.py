"""FastAPI server for the tech-news RAG assistant.

Exposes the RAG pipeline via a POST /ask endpoint and serves the static
frontend (index.html, style.css, script.js).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnra.generation.pipeline import Generator, build_generator
from tnra.retrieval.pipeline import Retriever, build_retriever
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class AskRequest(BaseModel):
    """Incoming question from the frontend."""

    question: str


class SourceItem(BaseModel):
    """One cited source, structured for the frontend."""

    title: str
    url: str
    publication: str


class AskResponse(BaseModel):
    """Answer returned to the frontend."""

    answer: str
    sources: list[SourceItem]
    error: bool = False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a rate-limit / quota error from the LLM provider.

    LangChain wraps provider exceptions, so we can't rely on isinstance against
    a specific class. We inspect the exception (and its chain) for a 429 status
    code or rate-limit keywords in the message.
    """
    for e in _walk_exception_chain(exc):
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
        if status == 429:
            return True
        message = str(e).lower()
        if "rate limit" in message or "quota" in message or "too many requests" in message:
            return True
    return False


def _walk_exception_chain(exc: BaseException) -> list[BaseException]:
    """Return exc and all its __cause__ / __context__ ancestors."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _build_pipelines() -> tuple[Retriever, Generator]:
    """Build the retriever and generator once, at startup."""
    base_cfg = OmegaConf.load("configs/base.yaml")
    ingestion_cfg = OmegaConf.load("configs/ingestion.yaml")
    retrieval_cfg = OmegaConf.load("configs/retrieval.yaml")
    generation_cfg = OmegaConf.load("configs/generation.yaml")

    client = chromadb.PersistentClient(path=base_cfg.paths.chroma_dir)
    collection = client.get_collection(ingestion_cfg.index.collection_name)  # type: ignore

    retriever = build_retriever(collection, retrieval_cfg.retrieval, ingestion_cfg.embeddings)
    generator = build_generator(generation_cfg)  # type: ignore
    return retriever, generator


load_dotenv()
logger.info("Building pipelines...")
_retriever, _generator = _build_pipelines()
logger.info("Pipelines ready")

app = FastAPI(title="Tech News RAG Assistant")


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    """Run the RAG pipeline for one question and return a structured answer."""
    question = request.question.strip()
    if not question:
        return AskResponse(answer="Please enter a question.", sources=[], error=True)

    try:
        passages = _retriever.retrieve(question)
        response = _generator.generate(question, passages)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            logger.warning("LLM rate limit reached")
            return AskResponse(
                answer=(
                    "The language model is temporarily unavailable "
                    "(usage limit reached). Please try again later."
                ),
                sources=[],
                error=True,
            )
        logger.exception("Unexpected error while answering")
        return AskResponse(
            answer="The assistant is temporarily unavailable. Please try again later.",
            sources=[],
            error=True,
        )

    sources = [
        SourceItem(title=s.article_title, url=s.article_url, publication=s.source)
        for s in response.sources
    ]
    logger.info("Answered a question — %d source(s)", len(sources))
    return AskResponse(answer=response.answer, sources=sources)


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend entry page."""
    return FileResponse(_FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=_FRONTEND_DIR), name="static")
