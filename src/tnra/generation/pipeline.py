"""Generation pipeline orchestration.

Assembles the generation stage into a single reusable object. The Generator
holds the load-once components (LLM client, guard config) and answers
questions; build_generator constructs it from raw config.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from tnra.generation.chain import answer_question
from tnra.generation.guard import GuardConfig
from tnra.generation.llm import LLMClient, LLMConfig
from tnra.generation.schemas import RAGResponse
from tnra.retrieval.schemas import RetrievalResult
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


class Generator:
    """The generation stage as a single reusable object.

    Holds the load-once components and answers questions. Instantiated once at
    startup; use build_generator() to construct it from raw config.
    """

    def __init__(self, llm: LLMClient, guard_cfg: GuardConfig) -> None:
        self.llm = llm
        self.guard_cfg = guard_cfg

    def generate(self, question: str, results: list[RetrievalResult]) -> RAGResponse:
        """Answer one question from its retrieved passages.

        Args:
            question: The user's question.
            results: Retrieved passages, ranked best-first (output of Phase 2).

        Returns:
            A complete RAGResponse — a real answer or a guard refusal.
        """
        return answer_question(question, results, self.llm, self.guard_cfg)


def build_generator(cfg: DictConfig) -> Generator:
    """Build a Generator from a raw (OmegaConf) generation config.

    Validates each config section with its Pydantic schema, then assembles the
    load-once components.

    Args:
        cfg: The raw generation config (parsed generation.yaml), with `llm`
            and `guard` sections.

    Returns:
        A ready-to-use Generator.
    """
    llm_cfg = LLMConfig.model_validate(OmegaConf.to_container(cfg.llm), from_attributes=True)
    guard_cfg = GuardConfig.model_validate(OmegaConf.to_container(cfg.guard), from_attributes=True)

    llm = LLMClient(llm_cfg)
    logger.info("Generator ready")
    return Generator(llm=llm, guard_cfg=guard_cfg)
