# Benchmark Results

Real evaluation results for new embedding models across 3 domains.

## Metric Definitions
| Metric | Description |
|--------|-------------|
| **Recall@10** | Fraction of relevant items retrieved in top-10 |
| **MRR@10** | Mean Reciprocal Rank of the first relevant item |
| **nDCG@10** | Normalized Discounted Cumulative Gain at top-10 |
| **P@10** | Precision at top-10 |
| **ms/query** | Average query latency in milliseconds |

## Domain: Tech

| Model | Recall@10 | MRR@10 | nDCG@10 | P@10 | ms/query |
|-------|-----------|--------|---------|------|---------|
| E5-base (Microsoft multilingual-e5-base) | 0.7800 | 0.6340 | 0.6121 | 0.1360 | 150.7 |
| Nomic (nomic-embed-text-v1.5) | 0.4725 | 0.4122 | 0.3765 | 0.0750 | 219.6 |
| BGE-M3 (BAAI/bge-m3) | 0.8150 | 0.6993 | 0.6722 | 0.1430 | 356.8 |
| Qwen3-Embedding-0.6B | 0.7792 | 0.6640 | 0.6325 | 0.1360 | 668.8 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — (API key not available) |

## Domain: Legal

| Model | Recall@10 | MRR@10 | nDCG@10 | P@10 | ms/query |
|-------|-----------|--------|---------|------|---------|
| E5-base (Microsoft multilingual-e5-base) | 0.3500 | 0.3145 | 0.2567 | 0.0990 | 110.4 |
| Nomic (nomic-embed-text-v1.5) | 0.1250 | 0.1384 | 0.0951 | 0.0370 | 244.6 |
| BGE-M3 (BAAI/bge-m3) | 0.3750 | 0.3831 | 0.3065 | 0.1060 | 322.1 |
| Qwen3-Embedding-0.6B | 0.3750 | 0.3999 | 0.3199 | 0.1070 | 710.6 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — (API key not available) |

## Domain: Medical

| Model | Recall@10 | MRR@10 | nDCG@10 | P@10 | ms/query |
|-------|-----------|--------|---------|------|---------|
| E5-base (Microsoft multilingual-e5-base) | 0.4583 | 0.4888 | 0.3909 | 0.1140 | 118.2 |
| Nomic (nomic-embed-text-v1.5) | 0.1917 | 0.2110 | 0.1668 | 0.0510 | 253.7 |
| BGE-M3 (BAAI/bge-m3) | 0.5450 | 0.5054 | 0.4339 | 0.1400 | 369.0 |
| Qwen3-Embedding-0.6B | 0.4483 | 0.4184 | 0.3629 | 0.1150 | 498.4 |
| text-embedding-3-large (OpenAI) | — | — | — | — | — (API key not available) |

## Notes

- **text-embedding-3-large** (OpenAI): requires `OPENAI_API_KEY`, skipped — API key not available in the build environment.
- **BGE-M3 / Qwen3-Embedding**: evaluated with `max_seq_length=256` truncation to make CPU inference feasible.
- **nomic-embed-text-v1.5**: evaluated with `max_seq_length=512`.
- All FAISS indexes use `IndexFlatIP` with L2-normalized vectors (cosine similarity).
- Evaluation performed on CPU only (no CUDA).
