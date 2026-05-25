"""Evaluation pipeline orchestration.

Runs the golden set end-to-end: retrieve, score retrieval, generate, judge.
Built incrementally — single-question evaluation first.
"""

from __future__ import annotations

from pydantic import BaseModel

from tnra.evaluation.golden_set import GoldenQuestion
from tnra.evaluation.judge import JudgeScore, judge_answer
from tnra.evaluation.metrics import recall_at_k, reciprocal_rank
from tnra.generation.llm import LLMClient
from tnra.generation.pipeline import Generator
from tnra.retrieval.pipeline import Retriever
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionResult(BaseModel):
    """Evaluation outcome for a single golden-set question."""

    id: str
    question: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    judge: JudgeScore


class EvaluationReport(BaseModel):
    """Aggregated evaluation results over the whole golden set."""

    n_questions: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_reciprocal_rank: float
    mean_faithfulness: float
    mean_relevance: float
    per_question: list[QuestionResult]


def evaluate_question(
    golden: GoldenQuestion,
    retriever: Retriever,
    generator: Generator,
    judge_llm: LLMClient,
) -> QuestionResult:
    """Evaluate one golden-set question end-to-end.

    Steps: retrieve passages, score retrieval (Recall@k, MRR), generate an
    answer, and judge it.

    Args:
        golden: The golden-set question with its expected article URLs.
        retriever: The retrieval pipeline (Phase 2).
        generator: The generation pipeline (Phase 3).
        judge_llm: The LLM client used by the judge.

    Returns:
        A QuestionResult bundling all metrics for this question.
    """
    # Retrieval.
    passages = retriever.retrieve(golden.question)
    retrieved_urls = [p.article_url for p in passages]
    expected = golden.expected_article_urls

    # Retrieval metrics.
    recall_1 = recall_at_k(retrieved_urls, expected, k=1)
    recall_3 = recall_at_k(retrieved_urls, expected, k=3)
    recall_5 = recall_at_k(retrieved_urls, expected, k=5)
    rr = reciprocal_rank(retrieved_urls, expected)

    # Generation.
    response = generator.generate(golden.question, passages)

    # Judge the generated answer against the retrieved passages.
    judge_score = judge_answer(golden.question, response.answer, passages, judge_llm)

    logger.info(
        "Evaluated %s — recall@5=%.0f rr=%.2f judge=%d/%d",
        golden.id,
        recall_5,
        rr,
        judge_score.faithfulness,
        judge_score.relevance,
    )
    return QuestionResult(
        id=golden.id,
        question=golden.question,
        recall_at_1=recall_1,
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        reciprocal_rank=rr,
        judge=judge_score,
    )


def aggregate_results(results: list[QuestionResult]) -> EvaluationReport:
    """Average the per-question results into a single evaluation report.

    Args:
        results: One QuestionResult per golden-set question.

    Returns:
        An EvaluationReport with mean metrics and the per-question detail.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty results list")

    n = len(results)

    def mean(values: list[float]) -> float:
        return sum(values) / n

    return EvaluationReport(
        n_questions=n,
        mean_recall_at_1=mean([r.recall_at_1 for r in results]),
        mean_recall_at_3=mean([r.recall_at_3 for r in results]),
        mean_recall_at_5=mean([r.recall_at_5 for r in results]),
        mean_reciprocal_rank=mean([r.reciprocal_rank for r in results]),
        mean_faithfulness=mean([float(r.judge.faithfulness) for r in results]),
        mean_relevance=mean([float(r.judge.relevance) for r in results]),
        per_question=results,
    )


def evaluate_golden_set(
    golden_questions: list[GoldenQuestion],
    retriever: Retriever,
    generator: Generator,
    judge_llm: LLMClient,
) -> EvaluationReport:
    """Evaluate every golden-set question and aggregate the results.

    Args:
        golden_questions: The full golden set.
        retriever: The retrieval pipeline (Phase 2).
        generator: The generation pipeline (Phase 3).
        judge_llm: The LLM client used by the judge.

    Returns:
        The aggregated EvaluationReport.
    """
    results: list[QuestionResult] = []
    for i, golden in enumerate(golden_questions, start=1):
        logger.info("Evaluating question %d/%d (%s)", i, len(golden_questions), golden.id)
        results.append(evaluate_question(golden, retriever, generator, judge_llm))
    return aggregate_results(results)
