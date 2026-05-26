"""FastAPI server for the tech-news RAG assistant.

Exposes the RAG pipeline via a POST /ask endpoint and serves the static
frontend (index.html, style.css, script.js).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import RateLimitError
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


def _format_retry_delay(message: str) -> str:
    """Extract a retry delay from a Groq rate-limit message and round it.

    Returns an empty string if no delay can be found.
    """
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return ""
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    total_minutes = minutes + (1 if seconds > 0 else 0)
    if total_minutes <= 1:
        return "about a minute"
    return f"about {total_minutes} minutes"


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
    except RateLimitError as exc:
        delay = _format_retry_delay(str(exc))
        logger.warning("Groq rate limit reached")
        wait = f" Please try again in {delay}." if delay else " Please try again later."
        return AskResponse(
            answer=(f"The language model is temporarily unavailable (usage limit reached).{wait}"),
            sources=[],
            error=True,
        )
    except Exception:
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
