"""
Converts doc-level qrels to chunk-level qrels.

For each query that has a doc-level qrel (chunk_id=null, doc_id=filename),
this script searches the SBERT FAISS index and selects the top-N chunks that
belong to the relevant document. Those chunks become the new chunk-level qrel
entries (relevance=2 for rank-1, relevance=1 for ranks 2..N).

Usage:
    python src/make_chunk_qrels.py --domain legal
    python src/make_chunk_qrels.py --domain medical
    python src/make_chunk_qrels.py --domain tech   # no-op if already chunk-level
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

# Add src/ to path so embedding_models / query imports work
import sys
sys.path.insert(0, str(Path(__file__).parent))

from embedding_models import load_model_from_artifacts


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chunk-level qrels from doc-level qrels.")
    parser.add_argument("--domain", required=True, help="Domain name (legal, medical, tech)")
    parser.add_argument(
        "--artifacts-root", default="artifacts", help="Root folder with model artifacts"
    )
    parser.add_argument(
        "--model-subdir", default="sbert", help="Which model's index to use for ranking (default: sbert)"
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="How many top chunks per relevant document to mark as relevant"
    )
    parser.add_argument(
        "--search-k", type=int, default=200,
        help="How many FAISS results to retrieve before filtering by doc (default: 200)"
    )
    args = parser.parse_args()

    domain = args.domain
    artifacts_dir = Path(args.artifacts_root) / domain / args.model_subdir
    benchmark_dir = Path("data") / "domains" / domain / "benchmark"

    queries_path = benchmark_dir / "queries.jsonl"
    qrels_path = benchmark_dir / "qrels.jsonl"

    queries = load_jsonl(queries_path)
    qrels = load_jsonl(qrels_path)

    # Group qrels by query_id
    qrels_by_query: dict[str, list[dict]] = {}
    for item in qrels:
        qrels_by_query.setdefault(item["query_id"], []).append(item)

    # Check if any qrels need conversion (have chunk_id=null)
    needs_conversion = any(q.get("chunk_id") is None for q in qrels)
    if not needs_conversion:
        print(f"[{domain}] All qrels already have chunk_id set — nothing to do.")
        return

    # Load meta
    meta_path = artifacts_dir / "meta.jsonl"
    meta: list[dict] = load_jsonl(meta_path)
    print(f"[{domain}] Loaded {len(meta)} chunks from {meta_path}")

    # Load FAISS index + model
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    model, config = load_model_from_artifacts(
        artifacts_dir=artifacts_dir,
        fallback_model_name="paraphrase-multilingual-MiniLM-L12-v2",
    )
    print(f"[{domain}] Loaded model: {config.model_name or config.model_type}")

    new_qrels: list[dict] = []
    converted = 0
    kept_chunk_level = 0

    for query_item in queries:
        qid = query_item["query_id"]
        query_text = query_item["query"]
        query_qrels = qrels_by_query.get(qid, [])

        if not query_qrels:
            continue

        # Separate doc-level from already-chunk-level entries
        doc_level = [q for q in query_qrels if q.get("chunk_id") is None]
        chunk_level = [q for q in query_qrels if q.get("chunk_id") is not None]

        # Keep existing chunk-level entries as-is
        new_qrels.extend(chunk_level)
        kept_chunk_level += len(chunk_level)

        if not doc_level:
            continue

        # For each doc-level entry, find top-N most relevant chunks
        # by searching the FAISS index and filtering by doc_id
        query_vec = model.encode_queries([query_text]).astype("float32")
        k = min(args.search_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        for qrel_entry in doc_level:
            relevant_doc = qrel_entry.get("doc_id", "")
            base_relevance = qrel_entry.get("relevance", 2)

            # Collect ranked chunks belonging to this document
            ranked_chunks: list[tuple[int, str]] = []  # (faiss_idx, chunk_id)
            for faiss_idx in indices[0]:
                if faiss_idx < 0 or faiss_idx >= len(meta):
                    continue
                record = meta[faiss_idx]
                if str(record.get("source", "")) == relevant_doc:
                    ranked_chunks.append((faiss_idx, str(record.get("chunk_id", ""))))

            if not ranked_chunks:
                # Fall back: keep the original doc-level entry unchanged
                new_qrels.append(qrel_entry)
                print(f"  [{qid}] WARNING: no chunks found for doc '{relevant_doc}', keeping doc-level entry")
                continue

            # Take top-N, assign graded relevance
            for rank, (_, chunk_id) in enumerate(ranked_chunks[: args.top_n]):
                rel = base_relevance if rank == 0 else max(1, base_relevance - 1)
                new_qrels.append({
                    "query_id": qid,
                    "chunk_id": chunk_id,
                    "doc_id": relevant_doc,
                    "relevance": rel,
                })

            converted += 1
            top_ids = [cid for _, cid in ranked_chunks[:args.top_n]]
            print(f"  [{qid}] doc '{relevant_doc}' → chunks: {top_ids}")

    print(
        f"\n[{domain}] Done. Converted {converted} doc-level qrels → chunk-level. "
        f"Kept {kept_chunk_level} existing chunk-level entries."
    )
    print(f"[{domain}] New qrels count: {len(new_qrels)} (was {len(qrels)})")

    # Backup original
    backup_path = qrels_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        backup_path.write_bytes(qrels_path.read_bytes())
        print(f"[{domain}] Backup saved: {backup_path}")

    # Sort by query_id for readability
    new_qrels.sort(key=lambda x: (x["query_id"], x.get("chunk_id") or ""))
    write_jsonl(qrels_path, new_qrels)
    print(f"[{domain}] Written: {qrels_path}")


if __name__ == "__main__":
    main()
