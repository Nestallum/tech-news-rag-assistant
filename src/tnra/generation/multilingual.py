"""Multilingual wrapper for the generation stage.

Lets users ask in their own language: the query is detected, translated to
the corpus language (English), answered by the core RAG pipeline, and the
answer is translated back. The core pipeline stays English-only.

This module is built incrementally — language detection first.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from lingua import Language, LanguageDetectorBuilder

from tnra.generation.llm import LLMClient
from tnra.utils.logger import get_logger

logger = get_logger(__name__)

# Maps our config language codes to lingua's Language enum members.
_CODE_TO_LANGUAGE: dict[str, Language] = {
    "en": Language.ENGLISH,
    "fr": Language.FRENCH,
    "es": Language.SPANISH,
    "de": Language.GERMAN,
    "zh": Language.CHINESE,
}

# Human-readable language names, for use inside translation prompts.
_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
}


class LanguageDetector:
    """Detects the language of a query, restricted to the supported set.

    The underlying lingua detector is built once and reused (load once,
    serve many).
    """

    def __init__(self, supported_codes: list[str], fallback_code: str) -> None:
        """Build the detector for the given supported languages.

        Args:
            supported_codes: Language codes the detector should consider
                (e.g. ["en", "fr", "es", "de", "zh"]).
            fallback_code: Code returned when detection yields a language
                outside the supported set.
        """
        unknown = set(supported_codes) - set(_CODE_TO_LANGUAGE)
        if unknown:
            raise ValueError(f"Unsupported language codes: {sorted(unknown)}")
        if fallback_code not in supported_codes:
            raise ValueError("fallback_code must be among supported_codes")

        self.fallback_code = fallback_code
        self._language_to_code = {_CODE_TO_LANGUAGE[code]: code for code in supported_codes}

        languages = [_CODE_TO_LANGUAGE[code] for code in supported_codes]
        self._detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def detect(self, text: str) -> str:
        """Detect the language code of a piece of text.

        Args:
            text: The text to analyse (typically the user's question).

        Returns:
            A supported language code. Falls back to fallback_code when lingua
            cannot confidently identify a supported language.
        """
        language = self._detector.detect_language_of(text)
        if language is None:
            return self.fallback_code
        return self._language_to_code.get(language, self.fallback_code)


_TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Translate \
the user's text from {source_name} to {target_name}.

Rules:
- Output ONLY the translated text. No preamble, no explanation, no quotes.
- Preserve the meaning, tone, and any proper nouns (product names, companies).
- Do not answer or react to the content — translate it faithfully, even if it \
is a question."""


class Translator:
    """Translates text between supported languages using the LLM.

    Each translation is one LLM call. The same LLMClient as the generation
    stage is reused — no extra model is loaded.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def translate(self, text: str, source_code: str, target_code: str) -> str:
        """Translate text from one supported language to another.

        Args:
            text: The text to translate.
            source_code: Source language code (e.g. "fr").
            target_code: Target language code (e.g. "en").

        Returns:
            The translated text. If source and target are identical, the text
            is returned unchanged (no LLM call).
        """
        if source_code == target_code:
            return text

        system_prompt = _TRANSLATION_SYSTEM_PROMPT.format(
            source_name=_CODE_TO_NAME[source_code],
            target_name=_CODE_TO_NAME[target_code],
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]
        translated = self.llm.invoke(messages)  # type: ignore
        logger.info("Translated text %s -> %s", source_code, target_code)
        return translated.strip()
