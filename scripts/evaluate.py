"""Entry point for the evaluation stage.

Runs the full golden set through retrieval, generation, and the LLM-as-judge,
then prints an aggregated report.

Usage:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnra.evaluation.golden_set import load_golden_set
from tnra.evaluation.pipeline import EvaluationReport, evaluate_golden_set
from tnra.generation.llm import LLMClient, LLMConfig
from tnra.generation.pipeline import build_generator
from tnra.retrieval.pipeline import build_retriever
from tnra.utils.logger import get_logger

logger = get_logger(__name__)


def _print_report(report: EvaluationReport) -> None:
    """Print the evaluation report in a readable form."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT — {report.n_questions} questions")
    print("=" * 60)
    print("\n  Retrieval")
    print(f"    Recall@1 : {report.mean_recall_at_1:.3f}")
    print(f"    Recall@3 : {report.mean_recall_at_3:.3f}")
    print(f"    Recall@5 : {report.mean_recall_at_5:.3f}")
    print(f"    MRR      : {report.mean_reciprocal_rank:.3f}")
    print("\n  Generation (LLM-as-judge, 1-5)")
    print(f"    Faithfulness : {report.mean_faithfulness:.2f}")
    print(f"    Relevance    : {report.mean_relevance:.2f}")
    print("\n  Per-question")
    for r in report.per_question:
        print(
            f"    {r.id}: R@5={r.recall_at_5:.0f} RR={r.reciprocal_rank:.2f} "
            f"judge={r.judge.faithfulness}/{r.judge.relevance}"
        )
    print("=" * 60 + "\n")


def main() -> None:
    """Build the pipelines and run the full evaluation."""
    load_dotenv()

    base_cfg = OmegaConf.load("configs/base.yaml")
    ingestion_cfg = OmegaConf.load("configs/ingestion.yaml")
    retrieval_cfg = OmegaConf.load("configs/retrieval.yaml")
    generation_cfg = OmegaConf.load("configs/generation.yaml")

    client = chromadb.PersistentClient(path=base_cfg.paths.chroma_dir)
    collection = client.get_collection(ingestion_cfg.index.collection_name)  # type: ignore

    retriever = build_retriever(collection, retrieval_cfg.retrieval, ingestion_cfg.embeddings)
    generator = build_generator(generation_cfg)  # type: ignore

    llm_cfg = LLMConfig.model_validate(
        OmegaConf.to_container(generation_cfg.llm), from_attributes=True
    )
    judge_llm = LLMClient(llm_cfg)

    golden_path = Path(base_cfg.paths.eval_dir) / "golden_set" / "golden_set.json"
    golden_questions = load_golden_set(golden_path)

    logger.info("Starting evaluation on %d questions", len(golden_questions))
    report = evaluate_golden_set(golden_questions, retriever, generator, judge_llm)
    _print_report(report)


if __name__ == "__main__":
    main()
