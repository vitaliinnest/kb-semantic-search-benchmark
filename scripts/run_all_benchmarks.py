"""
Run evaluate_benchmark.py for all 3 domains and all available model artifacts.
Collects metrics and writes RESULTS.md.
"""
import sys
import json
import subprocess
import time
from pathlib import Path

# Project root is one level above this script (which lives in scripts/)
ROOT = Path(__file__).parent.parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "src" / "evaluate_benchmark.py"

DOMAINS = ["tech", "legal", "medical"]
# Dense models + BM25 baseline
DENSE_MODEL_IDS = ["bm25", "e5-base", "nomic", "bge-m3", "qwen3"]
RESULTS_DIR = ROOT / "results"


def _is_ready(domain: str, model_id: str) -> bool:
    path = ROOT / "artifacts" / domain / model_id
    return (
        ((path / "faiss.index").exists() and (path / "meta.jsonl").exists())
        or ((path / "bm25.pkl").exists() and (path / "meta.jsonl").exists())
    )


def run_eval(domain: str) -> dict | None:
    """Run evaluate_benchmark.py for a domain, return parsed JSON results."""
    queries = ROOT / "data" / "domains" / domain / "benchmark" / "queries.jsonl"
    qrels = ROOT / "data" / "domains" / domain / "benchmark" / "qrels.jsonl"
    artifacts_root = ROOT / "artifacts" / domain

    if not queries.exists():
        print(f"  SKIP {domain}: queries not found at {queries}")
        return None
    if not qrels.exists():
        print(f"  SKIP {domain}: qrels not found at {qrels}")
        return None

    # Collect all valid model dirs (FAISS or BM25)
    model_dirs = []
    for mid in DENSE_MODEL_IDS:
        p = artifacts_root / mid
        if (p / "faiss.index").exists() and (p / "meta.jsonl").exists():
            model_dirs.append(str(p))
        elif (p / "bm25.pkl").exists() and (p / "meta.jsonl").exists():
            model_dirs.append(str(p))

    if not model_dirs:
        print(f"  SKIP {domain}: no built model artifacts")
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = RESULTS_DIR / f"benchmark_{domain}.txt"

    cmd = [
        str(VENV_PYTHON), str(EVAL_SCRIPT),
        "--queries", str(queries),
        "--qrels", str(qrels),
        "--model-artifacts", *model_dirs,
        "--top-k", "10",
        "--output", str(out_txt),
    ]

    print(f"  CMD: evaluate {domain} ({len(model_dirs)} models: {', '.join(Path(d).name for d in model_dirs)})")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT / "src"),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAIL {domain} ({elapsed:.0f}s) exit={result.returncode}")
        err_lines = (result.stderr or result.stdout or "").strip().splitlines()[-10:]
        for line in err_lines:
            print(f"    {line}")
        return None

    json_candidate = out_txt.with_suffix(".json")
    if json_candidate.exists():
        print(f"  OK: {len(model_dirs)} models evaluated in {elapsed:.0f}s")
        return json.loads(json_candidate.read_text(encoding="utf-8"))
    print(f"  WARN {domain}: JSON output not found at {json_candidate}")
    return None


def _fmt_ci(row: dict) -> str:
    lo = row.get("ndcg_ci_lo")
    hi = row.get("ndcg_ci_hi")
    if lo is not None and hi is not None:
        return f"[{lo:.4f}, {hi:.4f}]"
    return "—"


def write_results_md(all_domain_data: dict):
    """Build RESULTS.md from collected metrics."""
    lines = []
    lines.append("# Benchmark Results")
    lines.append("")
    lines.append("Real evaluation results for embedding models vs BM25 baseline across 3 domains.")
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("| Metric | Description |")
    lines.append("|--------|-------------|")
    lines.append("| **Recall@10** | Fraction of relevant items retrieved in top-10 |")
    lines.append("| **MRR@10** | Mean Reciprocal Rank of the first relevant item |")
    lines.append("| **nDCG@10** | Normalized Discounted Cumulative Gain at top-10 |")
    lines.append("| **nDCG 95% CI** | Bootstrap confidence interval for nDCG@10 (n=2000 resamples) |")
    lines.append("| **P@10** | Precision at top-10 |")
    lines.append("| **ms/query** | Average query latency in milliseconds |")
    lines.append("")

    MODEL_ORDER = ["bm25", "e5-base", "nomic", "bge-m3", "qwen3", "openai"]
    DISPLAY = {
        "bm25":   "BM25 (Okapi) — baseline",
        "e5-base": "E5-base (Microsoft multilingual-e5-base)",
        "nomic":  "Nomic (nomic-embed-text-v1.5)",
        "bge-m3": "BGE-M3 (BAAI/bge-m3)",
        "qwen3":  "Qwen3-Embedding-0.6B",
        "openai": "text-embedding-3-large (OpenAI)",
    }

    for domain in DOMAINS:
        lines.append(f"## Domain: {domain.capitalize()}")
        lines.append("")
        data = all_domain_data.get(domain)
        if data is None:
            lines.append("_Benchmark not run or failed._")
            lines.append("")
            continue

        model_rows = data.get("models", [])
        if not model_rows:
            lines.append("_No model results._")
            lines.append("")
            continue

        by_model: dict[str, dict] = {}
        for row in model_rows:
            key = Path(row.get("artifacts_dir", "")).name
            by_model[key] = row

        lines.append("| Model | Recall@10 | MRR@10 | nDCG@10 | nDCG 95% CI | P@10 | ms/query |")
        lines.append("|-------|-----------|--------|---------|-------------|------|---------|")
        for mid in MODEL_ORDER:
            display = DISPLAY.get(mid, mid)
            if mid == "openai":
                lines.append(f"| {display} | — | — | — | — | — | — (API key not available) |")
                continue
            row = by_model.get(mid)
            if row is None:
                lines.append(f"| {display} | _(not built)_ | — | — | — | — | — |")
                continue
            r = row.get("recall_at_k", 0)
            m = row.get("mrr_at_k", 0)
            n = row.get("ndcg_at_k", 0)
            ci = _fmt_ci(row)
            p = row.get("precision_at_k", 0)
            lat = row.get("avg_latency_ms", 0)
            lines.append(f"| {display} | {r:.4f} | {m:.4f} | {n:.4f} | {ci} | {p:.4f} | {lat:.1f} |")

        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- **BM25 (Okapi)**: lexical baseline, no neural embeddings. Tokenization: Unicode word tokens, lowercased.")
    lines.append("- **text-embedding-3-large** (OpenAI): requires `OPENAI_API_KEY`, skipped — API key not available.")
    lines.append("- **BGE-M3 / Qwen3-Embedding**: evaluated with `max_seq_length=256` truncation for CPU feasibility.")
    lines.append("- **nomic-embed-text-v1.5**: evaluated with `max_seq_length=512`.")
    lines.append("- **E5-base**: query prefix `query:`, document prefix `passage:` applied.")
    lines.append("- **nomic**: task prefixes `search_query:` / `search_document:` applied.")
    lines.append("- All FAISS indexes use `IndexFlatIP` with L2-normalized vectors (cosine similarity).")
    lines.append("- Bootstrap CI: 2000 resamples with replacement, seed=42.")
    lines.append("- Evaluation performed on CPU only (no CUDA).")
    lines.append("")

    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {out}")


def main():
    print("=" * 60)
    print("Running benchmark evaluations")
    print("=" * 60)

    print("\nArtifact status:")
    for domain in DOMAINS:
        print(f"  Domain: {domain}")
        for mid in DENSE_MODEL_IDS:
            status = "READY" if _is_ready(domain, mid) else "missing"
            print(f"    {mid:12}: {status}")

    print("\nRunning evaluations...")
    all_domain_data = {}
    for domain in DOMAINS:
        print(f"\n[DOMAIN] {domain}")
        result = run_eval(domain)
        if result is not None:
            all_domain_data[domain] = result

    print("\nWriting RESULTS.md...")
    write_results_md(all_domain_data)
    print("\nDone!")


if __name__ == "__main__":
    main()
