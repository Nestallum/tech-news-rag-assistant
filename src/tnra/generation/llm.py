"""Groq-backed LLM client for the generation stage.

Wraps LangChain's ChatGroq behind a thin, load-once class. The model itself
runs remotely on Groq's infrastructure; this client only holds configuration
and issues requests.
"""

from __future__ import annotations

import os

from langchain_cerebras import ChatCerebras
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from tnra.utils.logger import get_logger

logger = get_logger(__name__)


class LLMConfig(BaseModel):
    """Validated schema for the `llm` section of generation.yaml."""

    provider: str = Field(pattern="^cerebras$")
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)


class LLMClient:
    """Load-once wrapper around a Cerebras chat model.

    Instantiated a single time at startup and reused for every query
    (load once, serve many).
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._chat = self._build_chat(cfg)
        logger.info("LLMClient ready (provider=%s, model=%s)", cfg.provider, cfg.model)

    @staticmethod
    def _build_chat(cfg: LLMConfig) -> ChatCerebras:
        """Instantiate the underlying ChatCerebras client."""
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY is not set — add it to your .env file.")
        return ChatCerebras(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
            api_key=api_key,  # type: ignore
        )

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Send chat messages to the LLM and return its text response.

        Args:
            messages: Chat messages, e.g. produced by a ChatPromptTemplate
                (a system message plus a user message).

        Returns:
            The model's reply as plain text.
        """
        response = self._chat.invoke(messages)  # type: ignore
        return response.content  # type: ignore
