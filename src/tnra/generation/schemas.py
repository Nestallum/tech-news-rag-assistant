"""Typed response schema for the generation stage.

Defines the shape of what the RAG system returns. The LLM only produces raw
prose with numeric markers ([1], [2]...); the chain (sub-step 3.5) fills these
structures by attaching real article metadata to those markers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single cited source — what a marker like [1] actually refers to.

    Built from the metadata of a retrieved passage, never from LLM output,
    so titles and URLs are always accurate.
    """

    marker: int = Field(ge=1, description="Citation number used in the answer text")
    article_title: str
    article_url: str
    source: str = Field(description="Publication name, e.g. 'The Verge'")


class RAGResponse(BaseModel):
    """The complete result of a RAG query.

    This is the target structure the generation chain produces and the demo
    UI consumes.
    """

    answer: str = Field(description="The answer text, with [n] citation markers")
    sources: list[Source] = Field(
        default_factory=list,
        description="Sources actually cited in the answer, one per used marker",
    )
    query_language: str = Field(
        default="en",
        description="Language of the user's question (for the multilingual wrapper)",
    )
    guard_triggered: bool = Field(
        default=False,
        description="True if the anti-hallucination guard refused before the LLM",
    )
