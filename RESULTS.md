# Benchmark Results

Real evaluation results for embedding models vs BM25 baseline across 3 domains.

## Metric Definitions
| Metric | Description |
|--------|-------------|
| **Recall@10** | Fraction of relevant items retrieved in top-10 |
| **MRR@10** | Mean Reciprocal Rank of the first relevant item |
| **nDCG@10** | Normalized Discounted Cumulative Gain at top-10 |
| **nDCG 95% CI** | Bootstrap confidence interval for nDCG@10 (n=2000 resamples) |
| **P@10** | Precision at top-10 |
| **ms/query** | Average query latency in milliseconds |

## Domain: Tech

| Model | Recall@10 | MRR@10 | nDCG@10 | nDCG 95% CI | P@10 | ms/query |
|-------|-----------|--------|---------|-------------|------|---------|
| BM25 (Okapi) — baseline | 0.7542 | 0.4808 | 0.4861 | [0.4312, 0.5412] | 0.1340 | 1.5 |
| E5-base (Microsoft multilingual-e5-base) | 0.7800 | 0.6340 | 0.6121 | [0.5464, 0.6766] | 0.1360 | 225.1 |
| Nomic (nomic-embed-text-v1.5) | 0.4725 | 0.4122 | 0.3765 | [0.3017, 0.4545] | 0.0750 | 407.4 |
| BGE-M3 (BAAI/bge-m3) | 0.8150 | 0.6993 | 0.6722 | [0.6059, 0.7367] | 0.1430 | 2959.8 |
| Qwen3-Embedding-0.6B | 0.7792 | 0.6640 | 0.6325 | [0.5674, 0.7002] | 0.1360 | 365.4 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — | — (API key not available) |

## Domain: Legal

| Model | Recall@10 | MRR@10 | nDCG@10 | nDCG 95% CI | P@10 | ms/query |
|-------|-----------|--------|---------|-------------|------|---------|
| BM25 (Okapi) — baseline | 0.2333 | 0.2533 | 0.1875 | [0.1419, 0.2357] | 0.0640 | 2.9 |
| E5-base (Microsoft multilingual-e5-base) | 0.3500 | 0.3145 | 0.2567 | [0.2003, 0.3126] | 0.0990 | 67.0 |
| Nomic (nomic-embed-text-v1.5) | 0.1250 | 0.1384 | 0.0951 | [0.0629, 0.1303] | 0.0370 | 131.6 |
| BGE-M3 (BAAI/bge-m3) | 0.3750 | 0.3831 | 0.3065 | [0.2490, 0.3665] | 0.1060 | 202.9 |
| Qwen3-Embedding-0.6B | 0.3750 | 0.3999 | 0.3199 | [0.2580, 0.3811] | 0.1070 | 399.1 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — | — (API key not available) |

## Domain: Medical

| Model | Recall@10 | MRR@10 | nDCG@10 | nDCG 95% CI | P@10 | ms/query |
|-------|-----------|--------|---------|-------------|------|---------|
| BM25 (Okapi) — baseline | 0.3983 | 0.4003 | 0.3222 | [0.2577, 0.3854] | 0.1000 | 0.6 |
| E5-base (Microsoft multilingual-e5-base) | 0.4583 | 0.4888 | 0.3909 | [0.3226, 0.4621] | 0.1140 | 80.3 |
| Nomic (nomic-embed-text-v1.5) | 0.1917 | 0.2110 | 0.1668 | [0.1193, 0.2185] | 0.0510 | 150.4 |
| BGE-M3 (BAAI/bge-m3) | 0.5450 | 0.5054 | 0.4339 | [0.3695, 0.5001] | 0.1400 | 236.3 |
| Qwen3-Embedding-0.6B | 0.4483 | 0.4184 | 0.3629 | [0.2997, 0.4285] | 0.1150 | 465.9 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — | — (API key not available) |

## Notes

- **BM25 (Okapi)**: lexical baseline, no neural embeddings. Tokenization: Unicode word tokens, lowercased.
- **text-embedding-3-large** (OpenAI): requires `OPENAI_API_KEY`, skipped — API key not available.
- **BGE-M3 / Qwen3-Embedding**: evaluated with `max_seq_length=256` truncation for CPU feasibility.
- **nomic-embed-text-v1.5**: evaluated with `max_seq_length=512`.
- **E5-base**: query prefix `query:`, document prefix `passage:` applied.
- **nomic**: task prefixes `search_query:` / `search_document:` applied.
- All FAISS indexes use `IndexFlatIP` with L2-normalized vectors (cosine similarity).
- Bootstrap CI: 2000 resamples with replacement, seed=42.
- Evaluation performed on CPU only (no CUDA).
