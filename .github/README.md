# Tech News RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about
recent tech news, grounding every answer in real articles and citing its
sources.

**🟢 Live demo:** https://huggingface.co/spaces/Nestallum/tech-news-rag-assistant

![Demo screenshot](assets/figures/demo.png)

## Features

- **End-to-end RAG pipeline** — from news ingestion to a grounded, sourced answer.
- **Hybrid retrieval** — dense (embeddings) and sparse (BM25) search fused with
  Reciprocal Rank Fusion, then refined by a cross-encoder reranker.
- **Grounded generation** — answers are built only from retrieved passages, with
  an anti-hallucination retrieval-score guard that withholds an answer when the
  evidence is too weak.
- **Cited sources** — every answer lists the articles it draws from.
- **Measured quality** — evaluated on a hand-curated golden set with retrieval
  metrics and an LLM-as-judge.

## Architecture

The system is a four-stage pipeline:

![Architecture diagram](assets/figures/architecture.svg)

1. **Ingestion** — tech news articles are scraped from RSS feeds, cleaned,
   split into chunks, embedded with BGE-large, and indexed in ChromaDB.
2. **Retrieval** — for a question, dense and BM25 searches run in parallel;
   their results are fused with RRF, deduplicated at the article level, and
   reranked by a cross-encoder to keep the most relevant passages.
3. **Generation** — the question and retrieved passages are passed to a large
   language model (Llama 3.3 70B via Groq), which writes an answer grounded in
   the passages. A retrieval-score guard blocks answers when retrieval is weak.
4. **Evaluation** — a golden set of fact-based questions measures retrieval
   quality (Recall@k, MRR) and answer quality (LLM-as-judge).

## Evaluation

The system is evaluated on a hand-curated golden set of 15 fact-based
questions, each mapped to the article(s) that should be retrieved.

Retrieval is scored with **Recall@k** (is a correct article among the top *k*?)
and **MRR** (how highly is it ranked?). Answer quality is scored by an
**LLM-as-judge** rating faithfulness and relevance on a 1–5 scale.

| Metric            | Score        |
| ----------------- | ------------ |
| Recall@1          | _TBD_        |
| Recall@5          | _TBD_        |
| MRR               | _TBD_        |
| Faithfulness (1–5)| _TBD_        |
| Relevance (1–5)   | _TBD_        |

_The LLM-as-judge gives an indicative measure of answer quality, not an exact
ground truth: it reflects a model's assessment and carries some variability._

## Tech stack

- **Language:** Python 3.14
- **Orchestration:** LangChain
- **Embeddings:** BAAI/bge-large-en-v1.5 (Sentence-Transformers)
- **Vector store:** ChromaDB
- **Sparse retrieval:** BM25
- **Reranker:** cross-encoder (Sentence-Transformers)
- **LLM:** Llama 3.3 70B via Groq
- **Interface:** Gradio
- **Config:** OmegaConf + Pydantic
- **Tooling:** uv, ruff, pytest
- **Deployment:** Docker, Hugging Face Spaces, GitHub Actions CI

## Project structure

```
tech-news-rag-assistant/
├── src/tnra/
│   ├── ingestion/      # scraping, cleaning, chunking, embedding, indexing
│   ├── retrieval/      # dense + sparse search, RRF fusion, reranking
│   ├── generation/     # prompt, LLM client, guard, answer chain
│   ├── evaluation/     # golden set, metrics, LLM-as-judge
│   └── utils/          # shared helpers
├── scripts/            # ingest.py, app.py, evaluate.py
├── configs/            # YAML configuration
├── eval/golden_set/    # the curated evaluation set
├── tests/              # pytest suite
└── Dockerfile
```

## Getting started

### Prerequisites

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- A [Groq](https://console.groq.com/) API key

### Installation

```bash
git clone https://github.com/Nestallum/tech-news-rag-assistant.git
cd tech-news-rag-assistant
uv sync --extra dev
```

Create a `.env` file at the project root with your Groq key:

```
GROQ_API_KEY=your_key_here
```

### Usage

Ingest articles and build the index:

```bash
uv run python scripts/ingest.py
```

Launch the demo locally:

```bash
uv run python scripts/app.py
```

Run the evaluation on the golden set:

```bash
uv run python scripts/evaluate.py
```

## Limitations & future work

- **Demo latency** — the public demo runs on free CPU hardware, so each
  question takes a few seconds; the pipeline is much faster on a GPU.
- **Scope** — this is a question-answering system over individual articles.
  It is not designed for broad, corpus-wide requests like "summarize this
  week's news": summarizing an entire corpus is a different task from
  retrieval-augmented question answering.
- **Corpus freshness** — the index is a snapshot. A planned improvement is to
  purge articles older than one month on each ingestion run to keep the corpus
  current.
- **Multilingual support** — the system is English-only; multilingual
  question answering is a possible future extension.

## License

MIT
