# KB Semantic Search Benchmark — Claude Code Project Guide

## What this repo is

A Python benchmark harness + Flask web app for comparing embedding models on semantic search across three knowledge-base domains (tech, legal, medical). Built as part of a Ukrainian master's thesis (Нестеренко В.В., 2026).

## Key directories

```
src/                    Flask app + benchmark logic
  app.py                Flask web UI (model selector, search endpoint)
  build_index.py        Builds FAISS index + saves artifacts for a model
  evaluate_benchmark.py Runs nDCG@10/MRR@10/Recall@10/P@10 eval for a domain
  embedding_models.py   All embedding model classes (E5, BGE-M3, nomic, Qwen3, OpenAI…)

data/domains/{tech,legal,medical}/
  chunks.jsonl          {"doc_id":"…","text":"…"}
  benchmark/queries.jsonl   {"query_id":"q1","text":"…"}
  benchmark/qrels.jsonl     {"query_id":"q1","doc_id":"…","relevance":1}

artifacts/{tech,legal,medical}/{e5-base,bge-m3,nomic,qwen3}/
  faiss.index           FAISS IndexFlatIP (L2-normalised vectors)
  meta.jsonl            doc_id per vector row
  model.json            {"model_type":"e5","model_name":"…","params":{…}}

results/
  benchmark_{domain}.json   Full per-query metrics JSON
  benchmark_{domain}.txt    Human-readable summary
RESULTS.md              Auto-generated markdown summary (run_all_benchmarks.py)

scripts/                Benchmark-level scripts only
  run_all_benchmarks.py     Runs eval for all domains → RESULTS.md
  build_all_new_models.py   Builds FAISS indexes for all models

thesis/                 Master's thesis documents + editing scripts
  2026_M_PI_Nesterenko_VV.docx
  Nesterenko_Presentation.pptx
  unpacked_docx/            Working dir (docx unzipped for XML edits)
  scripts/                  One-off thesis editing scripts (docx/pptx)
```

## Models evaluated

| ID | Class | HF name | Notes |
|----|-------|---------|-------|
| `bm25` | `BM25RetrievalModel` | — (rank_bm25) | lexical baseline, no FAISS |
| `e5-base` | `E5EmbeddingModel` | intfloat/multilingual-e5-base | prefix query:/passage: |
| `bge-m3` | `SbertEmbeddingModel` | BAAI/bge-m3 | max_seq_length=256 |
| `nomic` | `NomicEmbeddingModel` | nomic-ai/nomic-embed-text-v1.5 | trust_remote_code, max_seq_length=512 |
| `qwen3` | `SbertEmbeddingModel` | Qwen/Qwen3-Embedding-0.6B | max_seq_length=256, batch_size=4 |
| `openai` | `OpenAIEmbeddingModel` | text-embedding-3-large | needs OPENAI_API_KEY |

## Common commands

```bash
# Run the Flask app
.venv/Scripts/python.exe src/app.py

# Build index for one model+domain
.venv/Scripts/python.exe src/build_index.py --domain tech --model-config artifacts/tech/e5-base/model.json

# Run full benchmark (all domains, all built models) → writes RESULTS.md
.venv/Scripts/python.exe scripts/run_all_benchmarks.py

# Run eval for one domain
.venv/Scripts/python.exe src/evaluate_benchmark.py \
  --queries data/domains/tech/benchmark/queries.jsonl \
  --qrels   data/domains/tech/benchmark/qrels.jsonl \
  --model-artifacts artifacts/tech/e5-base artifacts/tech/bge-m3 \
  --top-k 10 --output results/benchmark_tech.txt
```

## Thesis document

The Word document lives at `thesis/2026_M_PI_Nesterenko_VV.docx`. Edits are done by unpacking to `thesis/unpacked_docx/word/document.xml` (via zipfile), editing with ElementTree, then repacking. One-off editing scripts live in `thesis/scripts/`.

## Current benchmark results (as of 2026-04-27)

| Model | Tech nDCG | Legal nDCG | Medical nDCG |
|-------|-----------|------------|--------------|
| BGE-M3 | 0.6722 | 0.3065 | 0.4339 |
| Qwen3 | 0.6325 | 0.3199 | 0.3629 |
| E5-base | 0.6121 | 0.2567 | 0.3909 |
| nomic | 0.3765 | 0.0951 | 0.1668 |
| BM25 (baseline) | 0.4861 | 0.1875 | 0.3222 |
| OAI text-3-large | — | — | — (no API key) |

Bootstrap 95% CI (nDCG@10, n=2000): BGE-M3 Tech [0.6059, 0.7367] vs BM25 [0.4312, 0.5412] — non-overlapping (statistically significant).
