"""Gradio demo for the tech-news RAG assistant.

A web interface where a user asks a question and gets a grounded answer with
its cited sources. Built incrementally — the RAG-to-UI bridge first.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import gradio as gr
from dotenv import load_dotenv
from groq import RateLimitError
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnra.generation.pipeline import Generator, build_generator
from tnra.generation.schemas import RAGResponse
from tnra.retrieval.pipeline import Retriever, build_retriever
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


def _format_sources(response: RAGResponse) -> str:
    """Render the cited sources as a Markdown list.

    Each source shows its publication, title, and a clickable link. Returns a
    placeholder line when no sources were cited (e.g. a guard refusal).
    """
    if not response.sources:
        return "_No sources._"
    lines = ["**Sources**", ""]
    for s in response.sources:
        lines.append(f"- [{s.article_title}]({s.article_url}) — *{s.source}*")
    return "\n".join(lines)


def _format_retry_delay(message: str) -> str:
    """Extract a retry delay from a Groq rate-limit message and round it.

    Groq messages contain a phrase like 'try again in 12m11.808s'. This pulls
    out the minutes (and seconds) and renders a human-readable estimate.
    Returns an empty string if no delay can be found.
    """
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)  # type: ignore
    if not match:
        return ""
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    # Round up to the next minute for a clean estimate.
    total_minutes = minutes + (1 if seconds > 0 else 0)
    if total_minutes <= 1:
        return "about a minute"
    return f"about {total_minutes} minutes"


def answer(question: str, retriever: Retriever, generator: Generator) -> tuple[str, str]:
    """Run the full RAG pipeline for one question.

    Args:
        question: The user's question.
        retriever: The retrieval pipeline.
        generator: The generation pipeline.

    Returns:
        A pair (answer_text, sources_markdown) for the UI to display.
        On a Groq rate-limit error, returns a friendly message instead of failing.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    try:
        passages = retriever.retrieve(question)
        response = generator.generate(question, passages)
    except RateLimitError as exc:
        delay = _format_retry_delay(str(exc))
        logger.warning("Groq rate limit reached")
        wait = f" Please try again in {delay}." if delay else " Please try again later."
        return (
            f"The language model is temporarily unavailable (usage limit reached).{wait}",
            "",
        )
    except Exception:
        logger.exception("Unexpected error while answering")
        return (
            "The assistant is temporarily unavailable. Please try again later.",
            "",
        )

    logger.info("Answered a question — %d source(s)", len(response.sources))
    return response.answer, _format_sources(response)


def build_pipelines() -> tuple[Retriever, Generator]:
    """Build the retriever and generator once, at app startup."""
    base_cfg = OmegaConf.load("configs/base.yaml")
    ingestion_cfg = OmegaConf.load("configs/ingestion.yaml")
    retrieval_cfg = OmegaConf.load("configs/retrieval.yaml")
    generation_cfg = OmegaConf.load("configs/generation.yaml")

    client = chromadb.PersistentClient(path=base_cfg.paths.chroma_dir)
    collection = client.get_collection(ingestion_cfg.index.collection_name)  # type: ignore

    retriever = build_retriever(collection, retrieval_cfg.retrieval, ingestion_cfg.embeddings)
    generator = build_generator(generation_cfg)  # type: ignore
    return retriever, generator


_EXAMPLES = [
    "What did Airbnb expand into beyond home rentals?",
    "When might OpenAI's IPO happen?",
    "Why did China ban the Nvidia RTX 5090D V2?",
]


def build_demo(retriever: Retriever, generator: Generator) -> gr.Blocks:
    """Build the Gradio interface, wired to the RAG pipelines."""

    def respond(question: str) -> tuple[str, str]:
        """Bridge the UI to the RAG pipeline (closes over the pipelines)."""
        return answer(question, retriever, generator)

    with gr.Blocks(title="Tech News Assistant") as demo:
        gr.Markdown("# Tech News Assistant")
        gr.Markdown("Ask a question about recent tech news — grounded answers with sources.")

        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. How much will Anthropic pay xAI per month?",
            lines=2,
        )
        ask_button = gr.Button("Ask", variant="primary")

        gr.Examples(examples=_EXAMPLES, inputs=question_box)

        answer_box = gr.Markdown(label="Answer")
        sources_box = gr.Markdown(label="Sources")

        ask_button.click(
            fn=respond,
            inputs=question_box,
            outputs=[answer_box, sources_box],
        )

    return demo


def main() -> None:
    """Build the pipelines and launch the Gradio app."""
    load_dotenv()
    logger.info("Building pipelines...")
    retriever, generator = build_pipelines()
    demo = build_demo(retriever, generator)
    logger.info("Launching Gradio app")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
