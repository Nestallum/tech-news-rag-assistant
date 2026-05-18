# Tech News RAG Assistant

> 🚧 **WIP** — End-to-end RAG assistant answering questions on the latest tech news: AI, product launches, funding rounds, big tech moves.

A production-grade Retrieval-Augmented Generation system that ingests TechCrunch and VentureBeat
RSS feeds daily, indexes them with BGE embeddings into ChromaDB, and answers questions about
AI breakthroughs, product launches (iPhone, Galaxy, Pixel...), funding announcements, and the
tech industry at large — using Groq-hosted Llama 3.3 70B with cited sources.

## Stack

- **Orchestration:** LangChain
- **Embeddings:** BAAI/bge-large-en-v1.5
- **Vector store:** ChromaDB (persistent, local)
- **Reranker:** BAAI/bge-reranker-base (cross-encoder)
- **LLM:** Groq API (Llama 3.3 70B), with local Qwen 2.5 7B fallback
- **Demo:** Gradio on HuggingFace Spaces
- **Eval:** Retrieval metrics (Recall@k, MRR) + LLM-as-judge on a 15-question golden set
- **MLOps:** Docker, GitHub Actions CI (ruff + pytest)

## Setup

\`\`\`bash
uv sync --extra dev
cp .env.example .env  # then add your GROQ_API_KEY
\`\`\`

## Usage

\`\`\`bash
# 1. Ingest latest articles (incremental)
uv run python scripts/ingest.py --config configs/ingestion.yaml

# 2. Evaluate the system on the golden set
uv run python scripts/evaluate.py --config configs/eval.yaml

# 3. Launch the Gradio demo
uv run python scripts/app.py
\`\`\`

## Project structure

\`\`\`
src/tech_news_rag_assistant/
├── ingestion/     # RSS parsing, deduplication, chunking, embedding, indexing
├── retrieval/     # Dense + BM25 hybrid retrieval, cross-encoder reranking
├── generation/    # LLM clients, prompt templates, RAG chain assembly
├── evaluation/    # Retrieval metrics, LLM-as-judge, golden set runner
└── utils/         # Config, logger, paths, seed

scripts/  ingest.py   evaluate.py   app.py
configs/  base.yaml + per-stage overrides
eval/     golden_set/ (curated Q&A pairs)
tests/    pytest sanity checks
\`\`\`

## License

MIT
