"""Multilingual wrapper for the generation stage.

Lets users ask in their own language: the query is detected, translated to
the corpus language (English), answered by the core RAG pipeline, and the
answer is translated back. The core pipeline stays English-only.

This module is built incrementally — language detection first.
"""

from __future__ import annotations

from lingua import Language, LanguageDetectorBuilder

# Maps our config language codes to lingua's Language enum members.
_CODE_TO_LANGUAGE: dict[str, Language] = {
    "en": Language.ENGLISH,
    "fr": Language.FRENCH,
    "es": Language.SPANISH,
    "de": Language.GERMAN,
    "zh": Language.CHINESE,
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
